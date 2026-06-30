#!/bin/bash
#SBATCH --job-name=hyperstac-imc
#SBATCH --output=logs/hyperstac-imc-%j.out
#SBATCH --error=logs/hyperstac-imc-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

# Submit from the repository root:
#   sbatch run_imc_hyperstac.slurm
#
# Or override settings at submission time:
#   sbatch --export=ALL,IMAGE_FOLDER=/path/to/Images,OUTPUT_DIR=/path/to/output run_imc_hyperstac.slurm

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_DIR}"
mkdir -p logs

# Required: folder organised as {IMAGE_FOLDER}/{roi name}/{channel name}.tiff
IMAGE_FOLDER="${IMAGE_FOLDER:-/path/to/Images}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/hyperstac_imc_output}"

# Patch size is in pixels. For 1 um/pixel IMC, 100 means 100 um x 100 um patches.
PATCH_SIZE="${PATCH_SIZE:-100}"
STRIDE="${STRIDE:-${PATCH_SIZE}}"
PIXEL_SIZE_UM="${PIXEL_SIZE_UM:-1.0}"

# Optional comma-separated channel order, e.g. "DNA1,DNA2,CD45,PanCK".
# Leave empty to infer channels alphabetically from the first ROI.
CHANNELS="${CHANNELS:-}"

# Optional tissue/background filtering. Leave MASK_CHANNEL empty to keep all complete patches.
MASK_CHANNEL="${MASK_CHANNEL:-}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0.05}"
MIN_MASK_FRACTION="${MIN_MASK_FRACTION:-0.05}"
MIN_PATCH_SIGNAL="${MIN_PATCH_SIGNAL:-0.0}"

# Training defaults. Use ENCODER=small-cnn for a faster smoke test.
ENCODER="${ENCODER:-resnet50}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
PROJECTOR_OUTPUT_SIZE="${PROJECTOR_OUTPUT_SIZE:-2048}"

# Optional existing weights. If set, training is skipped.
ENCODER_WEIGHTS="${ENCODER_WEIGHTS:-}"

# Set RUN_SCANPY=1 to compute neighbors, UMAP, and Leiden in the output h5ad.
RUN_SCANPY="${RUN_SCANPY:-0}"
LEIDEN_RESOLUTION="${LEIDEN_RESOLUTION:-1.0}"

# Reuse previously created patches with REUSE_PATCHES=1, or replace outputs with OVERWRITE=1.
REUSE_PATCHES="${REUSE_PATCHES:-0}"
OVERWRITE="${OVERWRITE:-0}"

# If your cluster uses modules, uncomment/adapt these lines.
# module purge
# module load cuda/12.2
# module load anaconda3

# Conda activation. Create the environment once before submitting:
#   conda env create -f environment.yml
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hyperstac-imc

ARGS=(
  --image-folder "${IMAGE_FOLDER}"
  --output-dir "${OUTPUT_DIR}"
  --patch-size "${PATCH_SIZE}"
  --stride "${STRIDE}"
  --pixel-size-um "${PIXEL_SIZE_UM}"
  --encoder "${ENCODER}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --projector-output-size "${PROJECTOR_OUTPUT_SIZE}"
)

if [[ -n "${CHANNELS}" ]]; then
  ARGS+=(--channels "${CHANNELS}")
fi

if [[ -n "${MASK_CHANNEL}" ]]; then
  ARGS+=(
    --mask-channel "${MASK_CHANNEL}"
    --mask-threshold "${MASK_THRESHOLD}"
    --min-mask-fraction "${MIN_MASK_FRACTION}"
  )
fi

if [[ "${MIN_PATCH_SIGNAL}" != "0" && "${MIN_PATCH_SIGNAL}" != "0.0" ]]; then
  ARGS+=(--min-patch-signal "${MIN_PATCH_SIGNAL}")
fi

if [[ -n "${ENCODER_WEIGHTS}" ]]; then
  ARGS+=(--encoder-weights "${ENCODER_WEIGHTS}")
fi

if [[ "${RUN_SCANPY}" == "1" ]]; then
  ARGS+=(--run-scanpy --leiden-resolution "${LEIDEN_RESOLUTION}")
fi

if [[ "${REUSE_PATCHES}" == "1" ]]; then
  ARGS+=(--reuse-patches)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

echo "Running on host: $(hostname)"
echo "Repository: ${REPO_DIR}"
echo "Image folder: ${IMAGE_FOLDER}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Patch size: ${PATCH_SIZE}px"
echo "Encoder: ${ENCODER}"
echo "Epochs: ${EPOCHS}"

nvidia-smi || true

python imc_hyperstac_pipeline.py "${ARGS[@]}"
