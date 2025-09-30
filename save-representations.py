import os
import argparse

parser = argparse.ArgumentParser(description='SSL for multiplex data formatted like LATTICeA')
parser.add_argument('--data-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--encoder-name', type=str, default='resnet50')
parser.add_argument('--encoder-train-date', type=str, required=True)
parser.add_argument('--rep-save-path', type=str, required=True)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--image-size', type=int, default=224)
parser.add_argument('--num-channels', type=int, default=8)
parser.add_argument('--num-folds', type=int, default=2)
parser.add_argument('--file-type', type=str, default='numpy')
args = parser.parse_args()

import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
import glob
import h5py

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from classification_models.keras import Classifiers

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a', chunk_size=32):
    with h5py.File(output_path, mode) as file:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                chunk_shape = (chunk_size, ) + data_shape[1:]
                maxshape = (None, ) + data_shape[1:]
                dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
                dset[:] = val
                if attr_dict is not None:
                    if key in attr_dict.keys():
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0]:] = val
    return output_path

AUTO = tf.data.AUTOTUNE

NUM_FOLDS = args.num_folds

BATCH_SIZE = args.batch_size
IM_SIZE = args.image_size
NUM_CHANNELS = args.num_channels
input_shape = (IM_SIZE,IM_SIZE,NUM_CHANNELS)

record_path = os.path.join(args.data_path, '*')

def load_np_files_py(filename):
    return np.load(filename.numpy(), allow_pickle=True)

def load_np_file(filename):
    return tf.py_function(load_np_files_py, inp=[filename], Tout=[tf.float32])

@tf.function
def read_png_files(x):
    raw = tf.io.read_file(x)
    tensor = tf.io.decode_png(raw)
    return tensor

@tf.function
def read_jpeg_files(x):
    raw = tf.io.read_file(x)
    tensor = tf.io.decode_jpeg(raw)
    return tensor

if args.file_type=='numpy':
    load_func = load_np_file
elif args.file_type=='png':
    load_func = read_png_files
elif args.file_type=='jpeg':
    load_func = read_jpeg_files
else:
    raise ValueError("File type not recognised")

def get_patient_id(x):
    patient_label = x.split('/')[-1].split('_')[1]
    return patient_label

for fold in range(NUM_FOLDS):
    SAVE_PATH = os.path.join(args.save_path,args.encoder_name,f"{IM_SIZE}px-{NUM_FOLDS}-fold-{args.encoder_train_date}/fold-{fold}/encoder_weights_0.weights.h5")

    augment_im = lambda x: custom_augment_multiplex(tf.squeeze(x), input_shape=input_shape, output_shape=input_shape)

    paths = glob.glob(record_path)

    # Extracts patient IDs to make sure two slides from the same patient are assigned to the same fold
    case_numbers = []
    for path in paths:
        patient_label = int(get_patient_id(path))
        case_numbers.append(patient_label)
    case_numbers = sorted(list(set(case_numbers)))

    train_cases = [case_id for case_id in case_numbers if case_id%NUM_FOLDS != fold]
    valid_cases = [case_id for case_id in case_numbers if case_id%NUM_FOLDS == fold]

    train_paths = [path for path in paths if int(get_patient_id(path)) in train_cases]
    valid_paths = [path for path in paths if int(get_patient_id(path)) in valid_cases]
    print('train num paths', len(train_paths))
    print('valid num paths', len(valid_paths))
    ResNet50, _ = Classifiers.get('resnet50')
    # with strategy.scope():
    backbone = ResNet50(input_shape=input_shape, weights=None, include_top=False)
    backbone.load_weights(SAVE_PATH)
    head = tf.keras.Sequential([tf.keras.Input(shape=backbone.output_shape[1:]),
                                tf.keras.layers.GlobalAveragePooling2D()
                                ])
    encoder = tf.keras.Sequential([backbone, head])

    for SET in ['train','valid']:
        record_path_full = os.path.join(args.rep_save_path, f"{IM_SIZE}px-{NUM_FOLDS}-fold/siamese_unprivileged_multiplex_k_{fold}/feats_h5")

        if not os.path.exists(record_path_full):
            os.makedirs(record_path_full)

        if SET=='train':
            path_list = train_paths
        else:
            path_list = valid_paths

        for tma_path in (pbar:=tqdm(path_list)):
            print("TMA path", tma_path)
            pbar.set_description(f"Processing {SET} set for fold {fold}")
            tma_paths = sorted([os.path.join(tma_path, i) for i in os.listdir(tma_path)])
            tma_path_name = tma_path.split('/')[-1]

            print("TMA paths lengths", len(tma_paths))
            ds = tf.data.Dataset.from_tensor_slices(tma_paths)
            ds = ds.map(load_func, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            coords_list = np.array([name.split('/')[-1].split('.')[0].split('_') for name in tma_paths]).astype('int')

            # try:
            out = encoder.predict(ds, verbose=0)
            asset_dict = {'features': out, 'coords': coords_list, 'paths': np.array(tma_paths, dtype=np.string_)}
            save_hdf5(os.path.join(record_path_full, tma_path_name+'.h5'), asset_dict, attr_dict= None, mode='w')
            del out
            # except:
            #     print(tma_path_name)
#                 bad_paths.append(tma_path_name)
print("*********BAD_PATHS************")
print(bad_paths)
print("******************************")
