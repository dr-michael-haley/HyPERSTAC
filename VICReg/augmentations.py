import numpy as np
import tensorflow as tf
# import tensorflow_addons as tfa
# import tensorflow_models as tfm
from VICReg.rotate import rotate
from VICReg.gaussian_filter import gaussian_filter2d

@tf.function
def random_resize_crop(image, image_shape=(256,256,3), output_shape=(256,256,3), scale=[0.75, 1.0]):
    #output_shape[2] is redundant
    size_0         = tf.random.uniform(
                                    shape  = (1,),
                                    minval = scale[0] * image_shape[0],
                                    maxval = scale[1] * image_shape[0],
                                    dtype  = tf.float32,
                                    )
    size_1         = tf.random.uniform(
                                    shape  = (1,),
                                    minval = scale[0] * image_shape[1],
                                    maxval = scale[1] * image_shape[1],
                                    dtype  = tf.float32,
                                    )
    size_0         = tf.cast(size_0, tf.int32)[0]
    size_1         = tf.cast(size_1, tf.int32)[0]
    crop         = tf.image.random_crop(image, (size_0, size_1, image_shape[2]))
    crop_resize  = tf.image.resize(crop, (output_shape[0], output_shape[1]))
    return crop_resize

@tf.function
def flip_random_crop(image, input_shape, output_shape):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = random_resize_crop(image, input_shape, output_shape)
    return image


@tf.function
def float_parameter(level, maxval):
    return tf.cast(level * maxval / 10.0, tf.float32)

@tf.function
def sample_level(n):
    return tf.random.uniform(shape=[1], minval=0.1, maxval=n, dtype=tf.float32)

@tf.function
def random_brightness(x):
    return tf.image.random_brightness(x,256)

@tf.function
def random_contrast(x):
    return tf.image.random_contrast(x,0.5,2.)

@tf.function
def random_hue(x):
    return tf.image.random_hue(x, 0.3)
    
@tf.function
def random_saturation(x):
    return tf.image.random_saturation(x, 0.5, 2)

# @tf.function
# def random_rotate(x):
#     return tfa.image.rotate(x, tf.random.uniform(shape=(1,),minval=0,maxval=np.pi,dtype=tf.dtypes.float32))

@tf.function
def random_rotate(x):
    return rotate(x, tf.random.uniform(shape=(1,),minval=0,maxval=np.pi,dtype=tf.dtypes.float32))


@tf.function
def solarize(image, level=6):
    threshold = float_parameter(sample_level(level), 1)
    return tf.where(image < threshold, image, 255 - image)

@tf.function
def color_jitter(x, strength=0.5):
    #Needs revision
    x = tf.image.random_brightness(x, max_delta=0.8 * strength)
    x = tf.image.random_contrast(
        x, lower=1 - 0.8 * strength, upper=1 + 0.8 * strength
    )
    x = tf.image.random_saturation(
        x, lower=1 - 0.8 * strength, upper=1 + 0.8 * strength
    )
    x = tf.image.random_hue(x, max_delta=0.2 * strength)
    x = tf.clip_by_value(x, 0, 255)
    return x

@tf.function
def color_jitter_multiplex(x, strength=0.5):
    #Needs revision
    x = tf.image.random_brightness(x, max_delta=0.8 * strength)
    x = tf.image.random_contrast(
        x, lower=1 - 0.8 * strength, upper=1 + 0.8 * strength
    )
#     x = tf.image.random_saturation(
#         x, lower=1 - 0.8 * strength, upper=1 + 0.8 * strength
#     )
    x = tf.image.random_hue(x, max_delta=0.2 * strength)
    x = tf.clip_by_value(x, 0, 255)
    return x

@tf.function
def color_drop(x):
    x = tf.image.rgb_to_grayscale(x)
    x = tf.tile(x, [1, 1, 3])
    return x

@tf.function
def random_background(x, shape):
    rand = tf.random.uniform(shape,minval=0,maxval=256)
    x = tf.where(x==0,rand,x)
    return x

@tf.function
def random_fashion_background(x, f_x_train, size):
    rand = tf.random.uniform([],minval=0,maxval=60000,dtype=tf.dtypes.int32)
    im   = tf.gather(f_x_train,rand)
    x    = tf.image.resize(x, (shape[0], shape[1]))
    im   = tf.image.resize(im, (shape[0], shape[1]))
    x    = tf.where(x==0,tf.cast(im, tf.dtypes.float32),x)
    return x

@tf.function
def random_number_background(x, x_train, shape):
    rand = tf.random.uniform([],minval=0,maxval=60000,dtype=tf.dtypes.int32)
    im   = tf.gather(x_train,rand)
    x    = tf.image.resize(x, (shape[0], shape[1]))
    im   = tf.image.resize(im, (shape[0], shape[1]))
    x    = tf.where(x==0,tf.cast(im, tf.dtypes.float32),x)
    return x

@tf.function
def gaussian_noise(x, std=1):
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=std, dtype=tf.float32) 
    return x + noise

@tf.function
def gaussian_blur(x, shape):
    return gaussian_filter2d(x, filter_shape=(shape[0]//10, shape[1]//10), sigma=[1.9*tf.random.uniform(shape=(), dtype=tf.float32)+0.1 for _ in range(2)])

@tf.function
def mask_genes(x, frac_masked_range=[0,0.2]):
    prob = tf.random.uniform(shape=[], minval=frac_masked_range[0], maxval=frac_masked_range[1], dtype=tf.float32)
    return tf.nn.dropout(x, rate=prob)*(1-prob)

@tf.function
def random_updates(x, shape, prob_range=[0,0.1]):
    prob = tf.random.uniform(shape=[], minval=prob_range[0], maxval=prob_range[1], dtype=tf.float32)
    num_values = tf.cast(tf.math.floor(shape[0]*prob), tf.int32)
    idxs = tf.range(shape[0])
    sampled_idxs = tf.random.shuffle(idxs)[:num_values]
    values = tf.gather(x, sampled_idxs)
    return tf.tensor_scatter_nd_update(x, tf.expand_dims(sampled_idxs,1), values)
    
@tf.function
def random_apply(func, x, p, *args, **kwargs):
    if tf.random.uniform([], minval=0, maxval=1) < p:
        return func(x, *args, **kwargs)
    else:
        return x

@tf.function
def custom_augment_x_ray(image, input_shape, output_shape, x_train=None, f_x_train=None):
    image = tf.cast(image, tf.float32)
    #image = tf.image.resize(image,(output_size, output_size))
    image = flip_random_crop(image, input_shape, output_shape)
    image = random_apply(random_background, image, p=0.3, shape=output_shape)
    if f_x_train is not None:
        image = random_apply(random_fashion_background, image, p=0.2, f_x_train=f_x_train, shape=output_shape)
    if x_train is not None:
        image = random_apply(random_number_background, image, p=0.2, x_train=x_train, shape=output_shape)
    image = random_apply(random_rotate,image,p=0.4)
    #image = random_apply(solarize, image, p=0.3)
    #image = color_jitter(image)

    return image

@tf.function
def custom_augment_image(image, input_shape, output_shape, x_train=None, f_x_train=None):
    image = tf.cast(image, tf.float32)
    #image = tf.image.resize(image,(output_size, output_size))
    if f_x_train is not None:
        image = random_apply(random_fashion_background, image, p=0.2, f_x_train=f_x_train, shape=output_shape)
    if x_train is not None:
        image = random_apply(random_number_background, image, p=0.2, x_train=x_train, shape=output_shape)
    image = flip_random_crop(image, input_shape, output_shape)
    # image = random_apply(random_background, image, p=0.3, shape=output_shape)
    image = random_apply(color_jitter, image, p=0.8)
    image = random_apply(color_drop, image, p=0.2)
    image = random_apply(gaussian_blur, image, p=0.5, shape=output_shape)
    # image = random_apply(random_rotate,image,p=0.4)
    image = random_apply(solarize, image, p=0.1)
    
    return image

@tf.function
def custom_augment_multiplex(image, input_shape, output_shape, x_train=None, f_x_train=None):
    image = tf.cast(image, tf.float32)
    #image = tf.image.resize(image,(output_size, output_size))
    image = flip_random_crop(image, input_shape, output_shape)
    image = random_apply(random_background, image, p=0.3, shape=output_shape)
    if f_x_train is not None:
        image = random_apply(random_fashion_background, image, p=0.2, f_x_train=f_x_train, shape=output_shape)
    if x_train is not None:
        image = random_apply(random_number_background, image, p=0.2, x_train=x_train, shape=output_shape)
    image = random_apply(random_rotate,image,p=0.4)
    image = random_apply(solarize, image, p=0.3)
    # image = color_jitter(image)
    image = random_apply(gaussian_noise, image, p=1)
    return image

@tf.function
def custom_augment_genes(x, input_shape, output_shape):
    x = tf.cast(x, tf.float32)
    x = mask_genes(x)
    x = random_updates(x, shape=input_shape)
    x = gaussian_noise(x)
    x = tf.reshape(x, output_shape)
    return x
