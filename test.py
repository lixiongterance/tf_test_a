import os
import re

import tensorflow as tf
import tensorflow.keras as keras

import matplotlib.pyplot as plt

"""
image: (800, 600, 3)
mask: (800, 600, 1)
"""


def decode_and_normalize(filename_image, filename_mask):
    image_string = tf.io.read_file(filename_image)
    mask_string = tf.io.read_file(filename_mask)
    image_decode = tf.image.decode_png(image_string)
    mask_decode = tf.image.decode_png(mask_string)
    image_decode = tf.cast(image_decode, tf.float32) / 255.0
    mask_decode = tf.cast(mask_decode, tf.float32)

    return image_decode, mask_decode


data_dir = 'D:/dev/tensorflow_test/dataset/Automatic_Portrait_Matting/'
train_data_dir = data_dir + 'training/'
test_data_dir = data_dir + 'testing/'

train_image_filenames = tf.constant([train_data_dir + filename
                                     for filename in os.listdir(train_data_dir)
                                     if re.match(r'\d+\.png', filename)])
train_mask_filenames = tf.constant([train_data_dir + filename
                                    for filename in os.listdir(train_data_dir)
                                    if re.match(r'\d+_matte\.png', filename)])
test_image_filenames = tf.constant([test_data_dir + filename
                                    for filename in os.listdir(test_data_dir)
                                    if re.match(r'\d+\.png', filename)])
test_mask_filenames = tf.constant([test_data_dir + filename
                                   for filename in os.listdir(test_data_dir)
                                   if re.match(r'\d+_matte\.png', filename)])

train_images = tf.data.Dataset.from_tensor_slices((train_image_filenames, train_mask_filenames))
train_images = train_images.map(decode_and_normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_images = tf.data.Dataset.from_tensor_slices((test_image_filenames, test_mask_filenames))
test_images = test_images.map(decode_and_normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# for image, mask in train_images.as_numpy_iterator():
#     print(mask)
#     for i in range(len(mask)):
#         for j in range(len(mask[i])):
#             if mask[i][j] not in (0., 255.):
#                 print(mask[i][j])
#                 print(i, j)
#                 plt.figure()
#                 plt.subplot(1, 2, 1)
#                 plt.imshow(image)
#                 plt.subplot(1, 2, 2)
#                 plt.imshow(mask)
#                 plt.show()
#                 exit(1)

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 1000

train_dataset = train_images.cache().shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test_images.batch(BATCH_SIZE)


def display(display_list):
    plt.figure()

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(keras.preprocessing.image.array_to_img(display_list[i]))
        plt.colorbar()
        plt.axis('off')
    plt.show()


for image, mask in train_images.take(1):
    display([image, mask])
