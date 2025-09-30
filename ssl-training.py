#################################################################################
# Kernel and Imports
#################################################################################

import os, sys
import argparse

CNNs = ['resnet18',
        'resnet34',
        'resnet50',
        'resnet101',
        'resnet152',
        'seresnet18',
        'seresnet34',
        'seresnet50',
        'seresnet101',
        'seresnet152',
        'seresnext50',
        'seresnext101',
        'senet154',
        'resnet50v2'
        'resnet101v2',
        'resnet152v2',
        'resnext50',
        'resnext101',
        'vgg16',
        'vgg19',
        'densenet121',
        'densenet169',
        'densenet201',
        'inceptionresnetv2',
        'inceptionv3',
        'xception',
        'nasnetlarge',
        'nasnetmobile',
        'mobilenet',
        'mobilenetv2'
    ]

parser = argparse.ArgumentParser(description='SSL for multiplex data formatted like LATTICeA')
parser.add_argument('--data-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--encoder-name', type=str, default='resnet50')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--image-size', type=int, default=224)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--num-channels', type=int, default=8)
parser.add_argument('--num-folds', type=int, default=2)
parser.add_argument('--lr-base', type=float, default=1e-4)
parser.add_argument('--warmup-fraction', type=float, default=0.1)
parser.add_argument('--warmup-lr', type=float, default=0.0)
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--file-type', type=str, default='numpy')
parser.add_argument('--projector-output-size', type=int, default=8192)
parser.add_argument('--verbosity', type=int, default=2)
args = parser.parse_args()

import tensorflow as tf
import glob
from tqdm import tqdm
import datetime
t = datetime.datetime.now()

# You need to update this to include wherever the VICReg and SSL_Base directories are saved
sys.path.insert(0,'/home/')

from VICReg.vicreg_utils  import create_adam_opt, save_vicreg_weights, load_vicreg_weights
from VICReg.dataset_utils import preprocess_ds, load_datasets
from VICReg.augmentations import *
from VICReg.analysis_utils import *
from VICReg.warmupcosine import WarmUpCosine

from SSL_Base.ssl_models import VICReg

from classification_models.keras import Classifiers

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

# Only for ditributed training across multiple GPUs
strategy = tf.distribute.MirroredStrategy() #may need cross_device_ops=tf.distribute.ReductionToOneDevice()
print(physical_devices)

#################################################################################
# Hyperparameters
#################################################################################

NUM_FOLDS = args.num_folds

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
IM_SIZE = args.image_size
NUM_CHANNELS = args.num_channels
input_shape_multiplex = (IM_SIZE,IM_SIZE,NUM_CHANNELS)

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
    SAVE_PATH = os.path.join(args.save_path,args.encoder_name,f"{IM_SIZE}px-{NUM_FOLDS}-fold-{t.strftime('%Y-%m-%d')}/fold-{fold}")

    augment_im = lambda x: custom_augment_multiplex(tf.squeeze(x), input_shape=input_shape_multiplex, output_shape=input_shape_multiplex)

    paths = glob.glob(record_path)

    # Extracts patient IDs to make sure two slides from the same patient are assigned to the same fold
    case_numbers = []
    for path in paths:
        patient_label = int(get_patient_id(path))
        case_numbers.append(patient_label)
    case_numbers = sorted(list(set(case_numbers)))

    train_cases = [case_id for case_id in case_numbers if case_id%NUM_FOLDS != fold]
    valid_cases = [case_id for case_id in case_numbers if case_id%NUM_FOLDS == fold]

    train_paths = [ext_path for path in paths if int(get_patient_id(path)) in train_cases for ext_path in glob.glob(os.path.join(path,'*'))]
    valid_paths = [ext_path for path in paths if int(get_patient_id(path)) in valid_cases for ext_path in glob.glob(os.path.join(path,'*'))]

    DATASET_SIZE = np.round(len(train_paths)*(NUM_FOLDS-1)/NUM_FOLDS, 0)
    SHUFFLE_BUFFER = DATASET_SIZE
    WARMUP_FRACTION = args.warmup_fraction
    STEPS_PER_EPOCH = DATASET_SIZE//BATCH_SIZE
    WARMUP_EPOCHS = EPOCHS * WARMUP_FRACTION
    WARMUP_STEPS = int(WARMUP_EPOCHS * STEPS_PER_EPOCH)

    lr_decayed_fn = WarmUpCosine(learning_rate_base=args.lr_base,
                                total_steps=EPOCHS*STEPS_PER_EPOCH,
                                warmup_learning_rate=args.warmup_lr,
                                warmup_steps=WARMUP_STEPS,
                                start_step=args.start_epoch*STEPS_PER_EPOCH
                                )

    # Construct Dataset
    train_ds = tf.data.Dataset.from_tensor_slices(train_paths).shuffle(SHUFFLE_BUFFER)#.shard(10,0)
    valid_ds = tf.data.Dataset.from_tensor_slices(valid_paths)#.shard(10,0)

    train_ds = train_ds.map(load_func, num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds = valid_ds.map(load_func, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply augmentations
    train_ds = train_ds.map(lambda x: (augment_im(x), augment_im(x)), num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds = valid_ds.map(lambda x: (augment_im(x), augment_im(x)), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


    #################################################################################
    # Model & Training
    #################################################################################

    def create_projector(input_shape=(7,7,2048), output_size=8192):
        projector = tf.keras.models.Sequential()
        projector.add(tf.keras.Input(shape=input_shape))
        projector.add(tf.keras.layers.GlobalAveragePooling2D())
        projector.add(tf.keras.layers.Dense(output_size))
        projector.add(tf.keras.layers.BatchNormalization())
        projector.add(tf.keras.layers.Activation('relu'))

        projector.add(tf.keras.layers.Dense(output_size))
        projector.add(tf.keras.layers.BatchNormalization())
        projector.add(tf.keras.layers.Activation('relu'))

        projector.add(tf.keras.layers.Dense(output_size))
        return projector

    with strategy.scope():
        if args.start_epoch>0:
            vicreg = tf.keras.models.load_model(os.path.join(SAVE_PATH,f"{args.start_epoch}.keras"))
        else:
            if args.encoder_name in CNNs:
                ModelClass, _ = Classifiers.get(args.encoder_name)
                encoder1 = ModelClass(input_shape=input_shape_multiplex, weights=None, include_top=False)
            elif args.encoder_name.startswith('transformers:'):
                from transformers import AutoImageProcessor
                encoder1 = AutoImageProcessor.from_pretrained(args.encoder_name.removeprefix('transformers:'))
            else:
                raise ValueError("Model name not recognised")

            projector1 = create_projector(input_shape=encoder1.output_shape[1:], output_size=args.projector_output_size)

            # enc_list: list of all encoders
            # proj_list: list of all projectors

            enc_list = [encoder1]
            proj_list = [projector1]

            # Indice of each model in enc_list/proj_list: for single model training with 2 branches input [0,0],
            # for multiple models input the number of each model, e.g. [0,0,1,2] for two branches of model 0,
            # 1 branch of model 1 and 1 branch of model 2
            encoder_indices = [0,0]
            projector_indices = [0,0]
            optimiser = create_adam_opt(lr_decayed_fn)

            vicreg = VICReg(encoder_list=enc_list, projector_list=proj_list, encoder_indices=encoder_indices, projector_indices=projector_indices)
            vicreg.compile(optimizer=optimiser)
            vicreg.built=True

        # checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(SAVE_PATH,"{epoch}.keras"))
        vicreg.fit(train_ds,
                epochs=EPOCHS,
                validation_data=valid_ds,
                verbose=args.verbosity,
                # callbacks=[checkpoint]
                )
        save_vicreg_weights(SAVE_PATH, vicreg.encoder_list, vicreg.projector_list)
        vicreg.save(os.path.join(SAVE_PATH,"final_model.keras"))
