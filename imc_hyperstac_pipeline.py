#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
End-to-end HyPERSTAC-style pipeline for IMC folders.

Expected input layout:

    {image_folder}/{roi_name}/{channel_name}.tiff

All ROIs must contain the same channel names. Images are expected to already be
normalised to [0, 1].

The default patch size is 100 pixels. For 1 um/pixel IMC images, this gives
100 um x 100 um patches. Change this with --patch-size.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import tensorflow as tf
import tifffile
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from VICReg.ssl_base import VICReg


TIFF_SUFFIXES = {".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tile IMC channel folders, train a VICReg encoder, extract patch "
            "representations, and save an AnnData object."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--image-folder",
        type=Path,
        required=True,
        help="Folder organised as {image-folder}/{roi name}/{channel name}.tiff.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for patches, model weights, metadata, and AnnData output.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        help=(
            "Comma-separated channel order. If omitted, channel names are "
            "inferred alphabetically from the first ROI folder."
        ),
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=100,
        help=(
            "Patch width/height in pixels. With 1 um/pixel data, the default "
            "100 gives 100 um x 100 um patches."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride in pixels. Defaults to non-overlapping patches: stride = patch-size.",
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        default=1.0,
        help="Pixel size in microns, stored in AnnData metadata and spatial coordinates.",
    )
    parser.add_argument(
        "--mask-channel",
        type=str,
        default=None,
        help=(
            "Optional channel name used to filter empty/background patches. "
            "If omitted, every complete patch is kept."
        ),
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.05,
        help="Pixels above this value in --mask-channel are counted as tissue.",
    )
    parser.add_argument(
        "--min-mask-fraction",
        type=float,
        default=0.05,
        help=(
            "Minimum fraction of tissue pixels required when --mask-channel is set. "
            "Ignored when no mask channel is supplied."
        ),
    )
    parser.add_argument(
        "--min-patch-signal",
        type=float,
        default=0.0,
        help=(
            "Optional minimum mean signal across all channels for keeping a patch. "
            "Set to 0 to disable this filter."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate existing patch/model outputs in output-dir.",
    )
    parser.add_argument(
        "--reuse-patches",
        action="store_true",
        help="Reuse output-dir/patch_metadata.csv and output-dir/patches if present.",
    )

    parser.add_argument(
        "--encoder-weights",
        type=Path,
        default=None,
        help=(
            "Optional existing encoder weights (.weights.h5). If supplied, "
            "training is skipped and these weights are used for representations."
        ),
    )
    parser.add_argument("--epochs", type=int, default=30, help="VICReg training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument(
        "--representation-batch-size",
        type=int,
        default=None,
        help="Batch size for representation extraction. Defaults to --batch-size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Adam learning rate for VICReg training.",
    )
    parser.add_argument(
        "--projector-output-size",
        type=int,
        default=2048,
        help="Hidden/output width of the VICReg projector.",
    )
    parser.add_argument(
        "--encoder",
        choices=["resnet50", "small-cnn"],
        default="resnet50",
        help=(
            "Backbone encoder. resnet50 follows the original repository more closely; "
            "small-cnn is faster for quick tests."
        ),
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Optional cap on patches after tiling, useful for smoke tests.",
    )
    parser.add_argument(
        "--run-scanpy",
        action="store_true",
        help="Also compute neighbors, UMAP, and Leiden clusters in the AnnData object.",
    )
    parser.add_argument(
        "--leiden-resolution",
        type=float,
        default=1.0,
        help="Leiden resolution used when --run-scanpy is set.",
    )
    parser.add_argument(
        "--adata-name",
        type=str,
        default="imc_hyperstac_representations.h5ad",
        help="Name of the output AnnData file inside output-dir.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")

    return parser.parse_args()


def configure_runtime(seed: int) -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def channel_file_map(roi_dir: Path) -> dict[str, Path]:
    files = [p for p in roi_dir.iterdir() if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES]
    mapping: dict[str, Path] = {}
    for path in files:
        key = path.stem
        if key in mapping:
            raise ValueError(f"Duplicate files for channel '{key}' in {roi_dir}")
        mapping[key] = path
    return mapping


def discover_rois(image_folder: Path) -> list[Path]:
    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder does not exist: {image_folder}")
    rois = sorted([p for p in image_folder.iterdir() if p.is_dir()])
    if not rois:
        raise ValueError(f"No ROI folders found in {image_folder}")
    return rois


def resolve_channel_order(rois: list[Path], channels_arg: str | None) -> list[str]:
    if channels_arg:
        channels = [c.strip() for c in channels_arg.split(",") if c.strip()]
        if not channels:
            raise ValueError("--channels was supplied but no channel names were parsed.")
        return channels

    first_channels = sorted(channel_file_map(rois[0]).keys())
    if not first_channels:
        raise ValueError(f"No TIFF channels found in first ROI folder: {rois[0]}")
    return first_channels


def validate_roi_channels(rois: list[Path], channels: list[str]) -> None:
    expected = set(channels)
    for roi in rois:
        available = set(channel_file_map(roi))
        missing = sorted(expected - available)
        if missing:
            raise ValueError(f"ROI '{roi.name}' is missing channels: {missing}")


def read_channel(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D TIFF after squeeze, got shape {arr.shape}: {path}")
    arr = arr.astype(np.float32, copy=False)
    return np.clip(arr, 0.0, 1.0)


def load_roi_stack(roi_dir: Path, channels: list[str]) -> np.ndarray:
    mapping = channel_file_map(roi_dir)
    arrays = [read_channel(mapping[channel]) for channel in channels]
    shapes = {arr.shape for arr in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Channels in ROI '{roi_dir.name}' do not have matching shapes: {shapes}")
    return np.stack(arrays, axis=-1)


def ensure_clean_output(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for subdir in ["patches", "model"]:
            path = args.output_dir / subdir
            if path.exists():
                shutil.rmtree(path)
        for filename in ["patch_metadata.csv", args.adata_name, "channels.json", "run_config.json"]:
            path = args.output_dir / filename
            if path.exists():
                path.unlink()


def iter_patch_slices(height: int, width: int, patch_size: int, stride: int) -> Iterable[tuple[int, int, int, int]]:
    for row_start in range(0, height - patch_size + 1, stride):
        row_end = row_start + patch_size
        for col_start in range(0, width - patch_size + 1, stride):
            col_end = col_start + patch_size
            yield row_start, row_end, col_start, col_end


def tile_dataset(args: argparse.Namespace, rois: list[Path], channels: list[str]) -> pd.DataFrame:
    metadata_path = args.output_dir / "patch_metadata.csv"
    patches_dir = args.output_dir / "patches"

    if args.reuse_patches and metadata_path.exists() and patches_dir.exists():
        print(f"Reusing existing patches from {patches_dir}")
        return pd.read_csv(metadata_path)

    if patches_dir.exists() and any(patches_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{patches_dir} already exists and is not empty. Use --reuse-patches or --overwrite."
        )

    patches_dir.mkdir(parents=True, exist_ok=True)

    stride = args.stride or args.patch_size
    mask_index = channels.index(args.mask_channel) if args.mask_channel else None
    records: list[dict[str, object]] = []

    for roi_dir in tqdm(rois, desc="Tiling ROIs"):
        stack = load_roi_stack(roi_dir, channels)
        height, width, _ = stack.shape
        roi_patch_dir = patches_dir / roi_dir.name
        roi_patch_dir.mkdir(parents=True, exist_ok=True)

        for row_start, row_end, col_start, col_end in iter_patch_slices(
            height, width, args.patch_size, stride
        ):
            patch = stack[row_start:row_end, col_start:col_end, :]

            if mask_index is not None:
                mask = patch[:, :, mask_index] > args.mask_threshold
                if float(mask.mean()) < args.min_mask_fraction:
                    continue

            if args.min_patch_signal > 0 and float(patch.mean()) < args.min_patch_signal:
                continue

            patch_name = f"{row_start}_{row_end}_{col_start}_{col_end}.npy"
            patch_path = roi_patch_dir / patch_name
            np.save(patch_path, patch.astype(np.float32, copy=False))

            center_row_px = 0.5 * (row_start + row_end)
            center_col_px = 0.5 * (col_start + col_end)
            records.append(
                {
                    "roi": roi_dir.name,
                    "patch_path": str(patch_path),
                    "row_start": row_start,
                    "row_end": row_end,
                    "col_start": col_start,
                    "col_end": col_end,
                    "center_row_px": center_row_px,
                    "center_col_px": center_col_px,
                    "center_row_um": center_row_px * args.pixel_size_um,
                    "center_col_um": center_col_px * args.pixel_size_um,
                }
            )

    if not records:
        raise ValueError("No patches were created. Check patch size and filtering thresholds.")

    metadata = pd.DataFrame.from_records(records)

    if args.max_patches is not None and args.max_patches < len(metadata):
        metadata = metadata.sample(args.max_patches, random_state=args.seed).sort_index().reset_index(drop=True)

    metadata.to_csv(metadata_path, index=False)
    return metadata


def load_patch_tensor(path: tf.Tensor, patch_size: int, num_channels: int) -> tf.Tensor:
    def _load(path_value: bytes) -> np.ndarray:
        filename = path_value.decode("utf-8")
        return np.load(filename).astype(np.float32, copy=False)

    image = tf.numpy_function(_load, [path], tf.float32)
    image.set_shape((patch_size, patch_size, num_channels))
    return image


def make_augmenter(patch_size: int, num_channels: int):
    @tf.function
    def augment(image: tf.Tensor) -> tf.Tensor:
        image = tf.cast(image, tf.float32)
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.rot90(image, tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32))

        crop_scale = tf.random.uniform([], minval=0.75, maxval=1.0, dtype=tf.float32)
        crop_size = tf.cast(tf.round(tf.cast(patch_size, tf.float32) * crop_scale), tf.int32)
        crop_shape = tf.stack([crop_size, crop_size, tf.constant(num_channels, dtype=tf.int32)])
        image = tf.image.random_crop(image, size=crop_shape)
        image = tf.image.resize(image, (patch_size, patch_size), method="bilinear")

        image = tf.image.random_brightness(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)

        keep = tf.cast(tf.random.uniform((num_channels,), 0.0, 1.0) > 0.05, tf.float32)
        keep = tf.cond(tf.reduce_sum(keep) > 0, lambda: keep, lambda: tf.ones_like(keep))
        image = image * keep[tf.newaxis, tf.newaxis, :]

        image = image + tf.random.normal(tf.shape(image), mean=0.0, stddev=0.02, dtype=tf.float32)
        return tf.clip_by_value(image, 0.0, 1.0)

    return augment


def make_patch_dataset(
    patch_paths: list[str],
    patch_size: int,
    num_channels: int,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(np.asarray(patch_paths, dtype=str))
    if training:
        ds = ds.shuffle(buffer_size=min(len(patch_paths), 10000), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p: load_patch_tensor(p, patch_size=patch_size, num_channels=num_channels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        augment = make_augmenter(patch_size, num_channels)
        ds = ds.map(lambda x: (augment(x), augment(x)), num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_encoder(name: str, input_shape: tuple[int, int, int]) -> tf.keras.Model:
    if name == "resnet50":
        return tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_shape=input_shape,
        )

    if name == "small-cnn":
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        for filters in [32, 64, 128, 256]:
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        return tf.keras.Model(inputs, x, name="small_cnn_encoder")

    raise ValueError(f"Unsupported encoder: {name}")


def build_projector(input_shape: tuple[int, ...], output_size: int) -> tf.keras.Model:
    model = tf.keras.Sequential(name="vicreg_projector")
    model.add(tf.keras.Input(shape=input_shape))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    for _ in range(2):
        model.add(tf.keras.layers.Dense(output_size, use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dense(output_size))
    return model


def train_or_load_encoder(args: argparse.Namespace, metadata: pd.DataFrame, num_channels: int) -> tf.keras.Model:
    input_shape = (args.patch_size, args.patch_size, num_channels)
    encoder = build_encoder(args.encoder, input_shape)
    model_dir = args.output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.encoder_weights is not None:
        print(f"Loading encoder weights from {args.encoder_weights}")
        encoder.load_weights(args.encoder_weights)
        return encoder

    if len(metadata) < 2:
        raise ValueError("At least two patches are required for self-supervised training.")

    print(f"Training {args.encoder} VICReg encoder on {len(metadata)} patches")
    projector = build_projector(tuple(encoder.output_shape[1:]), args.projector_output_size)
    vicreg = VICReg(
        encoder_list=[encoder],
        projector_list=[projector],
        encoder_indices=[0, 0],
        projector_indices=[0, 0],
    )
    vicreg.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0))

    train_ds = make_patch_dataset(
        metadata["patch_path"].tolist(),
        patch_size=args.patch_size,
        num_channels=num_channels,
        batch_size=args.batch_size,
        training=True,
        seed=args.seed,
    )
    vicreg.fit(train_ds, epochs=args.epochs, verbose=2)

    weights_path = model_dir / "encoder.weights.h5"
    encoder.save_weights(weights_path)
    projector.save_weights(model_dir / "projector.weights.h5")
    print(f"Saved encoder weights to {weights_path}")
    return encoder


def extract_representations(
    args: argparse.Namespace,
    encoder: tf.keras.Model,
    metadata: pd.DataFrame,
    num_channels: int,
) -> np.ndarray:
    batch_size = args.representation_batch_size or args.batch_size
    ds = make_patch_dataset(
        metadata["patch_path"].tolist(),
        patch_size=args.patch_size,
        num_channels=num_channels,
        batch_size=batch_size,
        training=False,
        seed=args.seed,
    )

    representation_model = tf.keras.Sequential(
        [
            encoder,
            tf.keras.layers.GlobalAveragePooling2D(),
        ],
        name="representation_model",
    )
    features = representation_model.predict(ds, verbose=1)
    return features.astype(np.float32, copy=False)


def make_anndata(
    args: argparse.Namespace,
    metadata: pd.DataFrame,
    features: np.ndarray,
    channels: list[str],
) -> ad.AnnData:
    obs = metadata.copy()
    obs.index = [
        f"{row.roi}:{row.row_start}_{row.row_end}_{row.col_start}_{row.col_end}"
        for row in obs.itertuples(index=False)
    ]
    obs.index.name = "patch_id"

    adata = ad.AnnData(X=features, obs=obs)
    adata.obsm["spatial"] = obs[["center_col_um", "center_row_um"]].to_numpy(dtype=np.float32)
    adata.uns["channel_names"] = channels
    adata.uns["patch_size_px"] = args.patch_size
    adata.uns["patch_size_um"] = args.patch_size * args.pixel_size_um
    adata.uns["pixel_size_um"] = args.pixel_size_um
    adata.uns["encoder"] = args.encoder

    if args.run_scanpy:
        import scanpy as sc

        print("Running Scanpy neighbors, UMAP, and Leiden clustering")
        sc.pp.neighbors(adata, use_rep="X")
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=args.leiden_resolution)

    return adata


def save_run_config(args: argparse.Namespace, channels: list[str]) -> None:
    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)

    with open(args.output_dir / "channels.json", "w", encoding="utf-8") as f:
        json.dump(channels, f, indent=2)

    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.image_folder = args.image_folder.resolve()
    args.stride = args.stride or args.patch_size

    if args.patch_size <= 0:
        raise ValueError("--patch-size must be positive.")
    if args.stride <= 0:
        raise ValueError("--stride must be positive.")
    if not 0 <= args.min_mask_fraction <= 1:
        raise ValueError("--min-mask-fraction must be between 0 and 1.")

    configure_runtime(args.seed)
    ensure_clean_output(args)

    rois = discover_rois(args.image_folder)
    channels = resolve_channel_order(rois, args.channels)
    validate_roi_channels(rois, channels)
    if args.mask_channel is not None and args.mask_channel not in channels:
        raise ValueError(f"--mask-channel '{args.mask_channel}' is not in channel list.")

    print(f"Found {len(rois)} ROIs")
    print(f"Using {len(channels)} channels: {', '.join(channels)}")
    print(f"Patch size: {args.patch_size}px x {args.patch_size}px")

    save_run_config(args, channels)
    metadata = tile_dataset(args, rois, channels)
    print(f"Using {len(metadata)} patches")

    encoder = train_or_load_encoder(args, metadata, num_channels=len(channels))
    features = extract_representations(args, encoder, metadata, num_channels=len(channels))
    adata = make_anndata(args, metadata, features, channels)

    adata_path = args.output_dir / args.adata_name
    adata.write_h5ad(adata_path)
    print(f"Saved AnnData with shape {adata.shape} to {adata_path}")


if __name__ == "__main__":
    main()
