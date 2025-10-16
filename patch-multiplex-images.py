#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Processes OME-TIFF images by creating a mask from the DAPI channel,
tiling the masked image, and saving the tiles as both NumPy arrays and JPEG images.
"""

import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import cv2
from scipy.ndimage import median_filter
import os
from tqdm.auto import tqdm
from PIL import Image
import argparse

def render_rgb_images(ome_tiff):
    """
    Converts a multi-channel OME-TIFF image into an RGB image for visualization.
    """
    if ome_tiff.shape[-1] > 10:
        print("Warning: More than 10 channels detected. Color mapping may be incomplete.")

    channels = [ome_tiff[:,:,i] for i in range(ome_tiff.shape[-1])]

    # A default color mapping for up to 10 channels
    color_mapping = {
        0: [0, 0, 1],       # Blue (e.g., DAPI)
        1: [1, 0, 1],       # Magenta (e.g., Ki67)
        2: [1, 1, 0],       # Yellow (e.g., CD8)
        3: [0, 1, 1],       # Cyan (e.g., CD4)
        4: [1, 0.5, 0],     # Orange (e.g., SMA)
        5: [0, 1, 0],       # Green (e.g., CK)
        6: [1, 0, 0],       # Red (e.g., CD68)
        7: [1, 1, 1],       # White
        8: [0.5, 0, 0.5],   # Purple
        9: [0, 0.5, 0.5]    # Teal
    }

    rgb_image = np.zeros((*channels[0].shape, 3), dtype=np.float32)

    # Map each channel to its corresponding color in the RGB image
    for i, channel in enumerate(channels):
        if i in color_mapping:
            color = color_mapping[i]
            for j in range(3):
                rgb_image[:, :, j] += channel.astype(np.float32) * color[j]

    # Normalize and scale for visibility
    max_val = rgb_image.max()
    if max_val > 0:
        # Scale of 2.5 amplifies signal for better visibility
        rgb_image = np.clip((255 * 2.5 * rgb_image / max_val), 0, 255).astype(np.uint8)
    else:
        rgb_image = np.zeros(rgb_image.shape, dtype=np.uint8)
        
    return rgb_image

def process_images(args):
    """
    Main function to find images, create masks, tile, and save results.
    """
    # Use the tile size for both height (M) and width (N)
    M = N = args.tile_size

    # Get a list of image files to process
    try:
        image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.tif', '.tiff'))]
    except FileNotFoundError:
        print(f"Error: Input directory not found at {args.input_dir}")
        return

    if not image_files:
        print(f"No TIFF files found in {args.input_dir}")
        return

    print(f"Found {len(image_files)} TIFF files to process.")

    for path in tqdm(image_files, desc="Processing Images"):
        base_name = os.path.splitext(path)[0]
        
        # Skip if already processed
        if os.path.exists(os.path.join(args.out_path, "jpeg", base_name)) or \
           os.path.exists(os.path.join(args.unwanted_path, "jpeg", base_name)):
            print(f"Skipping already processed file: {path}")
            continue

        try:
            full_path = os.path.join(args.input_dir, path)
            im = tifffile.imread(full_path)

            if args.move_axis:
                im = np.moveaxis(im, 0, 2)

            # Ensure image has enough channels
            if im.shape[2] < args.dapi_channel + 1 or im.shape[2] < args.num_channels:
                 print(f"Skipping {path}: Not enough channels.")
                 continue

            # Create an RGB representation for saving JPEGs
            jpeg_image = render_rgb_images(im[:,:,:args.num_channels])

            # --- Create mask from DAPI channel ---
            dapi_channel = im[:, :, args.dapi_channel]
            # Downscale for faster processing
            small_dapi = cv2.resize(dapi_channel, (dapi_channel.shape[1] // args.resize_factor, dapi_channel.shape[0] // args.resize_factor))
            dilated = cv2.dilate(small_dapi, kernel=np.ones((args.dilate_kernel, args.dilate_kernel), np.uint8), iterations=1)
            filtered = median_filter(dilated, size=args.filter_size)
            # Upscale mask back to original dimensions
            resized_image = cv2.resize(filtered, (dapi_channel.shape[1], dapi_channel.shape[0]))
            
            # Thresholding to get binary mask
            t = threshold_otsu(resized_image)
            mask = resized_image > t
            
            # Apply mask to the first num_channels
            output = mask[:, :, np.newaxis] * im[:,:,:args.num_channels]

            # --- Tiling ---
            tiles_np = {}
            tiles_jpeg = {}
            for x in range(0, im.shape[0], M):
                for y in range(0, im.shape[1], N):
                    if (x + M <= im.shape[0]) and (y + N <= im.shape[1]):
                        # Keep tile if more than half of it is covered by the mask
                        if mask[x:x+M, y:y+N].sum() > (M * N / 2):
                            tile_name = f"{x}_{x+M}_{y}_{y+N}"
                            tiles_np[tile_name] = output[x:x+M, y:y+N]
                            tiles_jpeg[tile_name] = jpeg_image[x:x+M, y:y+N]
            
            # --- Decide save path based on tile count ---
            # The original script had logic to check against a hardcoded 224px tile count.
            # This version uses the count of tiles generated with the specified --tile-size.
            if len(tiles_np) > args.min_tiles:
                save_path = args.out_path
            else:
                save_path = args.unwanted_path

            # --- Save the tiles ---
            if not tiles_np:
                print(f"No valid tiles found for {path}. Moving to unwanted.")
                # Create a placeholder directory to mark as processed
                os.makedirs(os.path.join(args.unwanted_path, "jpeg", base_name), exist_ok=True)
                continue

            numpy_save_dir = os.path.join(save_path, "numpy", base_name)
            jpeg_save_dir = os.path.join(save_path, "jpeg", base_name)
            os.makedirs(numpy_save_dir, exist_ok=True)
            os.makedirs(jpeg_save_dir, exist_ok=True)

            for name, tile in tqdm(tiles_np.items(), leave=False, desc=f"Saving {len(tiles_np)} tiles for {base_name}"):
                np.save(os.path.join(numpy_save_dir, name + '.npy'), tile)
                jpeg = Image.fromarray(tiles_jpeg[name], mode='RGB')
                jpeg.save(os.path.join(jpeg_save_dir, name + '.jpeg'))

        except Exception as e:
            print(f"Could not process file {path}. Error: {e}")

def main():
    """Parses command-line arguments and runs the image processing."""
    parser = argparse.ArgumentParser(
        description="Process OME-TIFF images, mask using DAPI channel, and save as tiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--input-dir", type=str, required=True, help="Path to the directory containing input OME-TIFF images.")
    parser.add_argument("--out-path", type=str, required=True, help="Path to the output directory for valid tiled images.")
    parser.add_argument("--unwanted-path", type=str, required=True, help="Path to the output directory for images with too few tiles.")

    # Optional arguments with defaults
    parser.add_argument("--dapi-channel", type=int, default=6, help="Index of the DAPI channel for creating the mask.")
    parser.add_argument("--num-channels", type=int, default=8, help="Number of channels to process and save in the output.")
    parser.add_argument("--tile-size", type=int, default=224, help="The height and width of the square tiles.")
    parser.add_argument("--min-tiles", type=int, default=20, help="Minimum number of tiles for an image to be considered 'wanted'.")
    parser.add_argument("--resize-factor", type=int, default=10, help="Factor to downscale DAPI channel for faster mask creation.")
    parser.add_argument("--dilate-kernel", type=int, default=50, help="Size of the dilation kernel for mask creation.")
    parser.add_argument("--filter-size", type=int, default=20, help="Size of the median filter for mask smoothing.")
    parser.add_argument("--move-axis", action="store_true", help="Set this flag to move image axis from (Channel, Height, Width) to (Height, Width, Channel).")

    args = parser.parse_args()
    process_images(args)

if __name__ == "__main__":
    main()
