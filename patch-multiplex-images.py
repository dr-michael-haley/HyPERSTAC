import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import cv2
from scipy.ndimage import median_filter
import os
from tqdm.auto import tqdm
from PIL import Image

def render_rgb_images(ome_tiff):
    channels = [ome_tiff[:,:,i] for i in range(ome_tiff.shape[-1])]

    # Create a mapping of channels to colors
    color_mapping = {
    0: [0, 0, 1],    # Blue (DAPI)
    1: [1, 0, 1],    # Magenta, Ki67
    2: [1, 1, 0],    # Yellow, CD8
    3: [0, 1, 1],    # Cyan, CD4
    4: [1, 0.5, 0],  # Orange, SMA
    5: [0, 1, 0],    # Green, CK
    6: [1, 0, 0],    # Red, CD68
    7: [1, 1, 1],    # White
    8: [0, 1, 0.5],  # Pink
    9: [0, 0, 0]     # Black (Autofluorescence)
    }

    # Create an empty RGB image
    rgb_image = np.zeros((*channels[0].shape, 3), dtype=np.float32)

    # Map each channel to its corresponding color in the RGB image
    for i, channel in enumerate(channels):
        color = color_mapping[i]
        for j in range(3):
            rgb_image[:, :, j] += channel.astype(np.float32) * color[j]

    rgb_image = (255 * 2.5 * rgb_image / rgb_image.max()).astype(np.uint8) ## Scale of 2.5 to amplify signal for visibility purposes only
    return rgb_image

DAPI_CHANNEL = 6
NUM_CHANNELS = 8

MOVE_AXIS = False

RESIZE_FACTOR = 10
DILATE_KERNEL = 50
FILTER_SIZE = 20
M = N = 224
MIN_TILES = 20

dir_path = "/path/to/images/"
out_path = f"/save/path/dapi_masked_{M}px_tiled_bioclavis_no_background"
unwanted_path = f"/save/path/unwanted_dapi_masked_{M}px_tiled_bioclavis_no_background"

for path in tqdm(os.listdir(dir_path)):
    if (not os.path.exists(os.path.join(out_path, "jpeg", path[:-4]))) & (not os.path.exists(os.path.join(unwanted_path, "jpeg", path[:-4]))) & (not os.path.isdir(os.path.join(dir_path,path))):
        im = tifffile.imread(os.path.join(dir_path,path))
        if MOVE_AXIS:
            im = np.moveaxis(im, 0, 2)
        jpeg_image = render_rgb_images(im)
        resized_image = cv2.resize(median_filter(cv2.resize(cv2.dilate(im[:,:,DAPI_CHANNEL], kernel = np.ones((DILATE_KERNEL,DILATE_KERNEL), np.uint8), iterations=1), (im[:,:,DAPI_CHANNEL].shape[0]//RESIZE_FACTOR, im[:,:,DAPI_CHANNEL].shape[1]//RESIZE_FACTOR)), (FILTER_SIZE,FILTER_SIZE)), (im[:,:,DAPI_CHANNEL].shape[1],im[:,:,DAPI_CHANNEL].shape[0]))
        t = threshold_otsu(resized_image)
        mask = resized_image>t
        output = mask[:,:,np.newaxis]*im

        tiles_np = {f"{x}_{x+M}_{y}_{y+N}": output[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N) if output[x:x+M,y:y+N].shape==(M,N,NUM_CHANNELS) if mask[x:x+M,y:y+N].sum()>M*N/2}
        tiles_jpeg = {f"{x}_{x+M}_{y}_{y+N}": jpeg_image[x:x+M,y:y+N] for x in range(0,jpeg_image.shape[0],M) for y in range(0,jpeg_image.shape[1],N) if jpeg_image[x:x+M,y:y+N].shape==(M,N,3) if mask[x:x+M,y:y+N].sum()>M*N/2}

        if M!=224:
            tiles_np_224 = {f"{x}_{x+224}_{y}_{y+224}": np.moveaxis(output[:,x:x+224,y:y+224], 0, 2) for x in range(0,im.shape[1],224) for y in range(0,im.shape[2],224) if output[:,x:x+224,y:y+224].shape==(8,224,224) if mask[x:x+224,y:y+224].sum()>224*224/2}
        else:
            tiles_np_224 = tiles_np

        if len(tiles_np_224.items())>MIN_TILES:
            save_path = out_path
        else:
            save_path = unwanted_path

        os.makedirs(os.path.join(save_path, "numpy", path[:-4]))
        os.makedirs(os.path.join(save_path, "jpeg", path[:-4]))
        for name, tile in tqdm(tiles_np.items()):
            np.save(os.path.join(save_path, "numpy", path[:-4], name+'.npy'), tile)
            jpeg = Image.fromarray(tiles_jpeg[name], mode='RGB')
            jpeg.save(os.path.join(save_path, "jpeg", path[:-4], name+'.jpeg'))
