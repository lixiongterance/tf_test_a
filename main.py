import os
import re
import datetime

import tensorflow as tf
import tensorflow.keras as keras

import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', help='train or test')
args = parser.parse_args()

# 防止 GPU OOM
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     for i in range(len(physical_devices)):
#         tf.config.experimental.set_memory_growth(physical_devices[i], True)
#         print('memory growth: ', tf.config.experimental.get_memory_growth(physical_devices[i]))
# else:
#     print('No GPU device')

# 防止内存（显存） allocate 失败
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        print(e)

# 禁用 GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 原始数据
data_dir = './dataset/Automatic_Portrait_Matting/'
train_data_dir = data_dir + 'training/'
test_data_dir = data_dir + 'testing/'

# 参数定义
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
SHUFFLE_BUFFER_SIZE = 1000
OUTPUT_CHANNELS = 2  # 输出结果类别个数

checkpoint_dir = './checkpoints/'
checkpoint_path = checkpoint_dir + 'cp-{epoch:04d}.ckpt'
tensorboard_dir = './tensorboard/'


def display(display_list):
    plt.figure()

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


class DataLoader:
    """
    real_image: (800, 600, 3)
    real_mask: (800, 600, 1)
    output_image: (800, 608, 3)
    output_mask: (800, 608, 1)
    """

    def __init__(self, load_train=True, load_test=True):
        if load_train:
            self.train_image_filenames = tf.constant([train_data_dir + filename
                                                      for filename in os.listdir(train_data_dir)
                                                      if re.match(r'\d+\.png', filename)])
            self.train_mask_filenames = tf.constant([train_data_dir + filename
                                                     for filename in os.listdir(train_data_dir)
                                                     if re.match(r'\d+_matte\.png', filename)])

            self.train_images = tf.data.Dataset.from_tensor_slices(
                (self.train_image_filenames, self.train_mask_filenames))
            self.train_images = self.train_images.map(DataLoader.read_and_convert,
                                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

            # TODO cache 函数导致 CPU 内存持续增长，导致 OOM(why?而且数据量总共 1 GB?)，去掉之后，一轮训练结束时会逐渐释放
            # self.train_dataset = self.train_images.cache().shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).repeat()
            self.train_dataset = self.train_images.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).repeat()
            self.train_dataset = self.train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            self.num_train_data = self.train_image_filenames.shape[0]
        if load_test:
            self.test_image_filenames = tf.constant([test_data_dir + filename
                                                     for filename in os.listdir(test_data_dir)
                                                     if re.match(r'\d+\.png', filename)])
            self.test_mask_filenames = tf.constant([test_data_dir + filename
                                                    for filename in os.listdir(test_data_dir)
                                                    if re.match(r'\d+_matte\.png', filename)])

            self.test_images = tf.data.Dataset.from_tensor_slices((self.test_image_filenames,
                                                                   self.test_mask_filenames))
            self.test_images = self.test_images.map(DataLoader.read_and_convert,
                                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

            self.test_dataset = self.test_images.batch(BATCH_SIZE)

            self.num_test_data = self.test_image_filenames.shape[0]

    @staticmethod
    def show_sample(data, num=1):
        for image, mask in data.take(num):
            display([image, mask])

    @staticmethod
    def read_and_convert(filename_image, filename_mask):
        # read and decode
        image_string = tf.io.read_file(filename_image)
        mask_string = tf.io.read_file(filename_mask)
        image_decode = tf.image.decode_png(image_string)
        mask_decode = tf.image.decode_png(mask_string)
        # resize
        image_resized = tf.image.resize(image_decode, (800, 608))
        mask_resized = tf.image.resize(mask_decode, (800, 608))
        # normalize
        image_nor = tf.cast(image_resized, tf.float32) / 255.0
        mask_nor = tf.cast(mask_resized, tf.float32) / 255.0

        return image_nor, mask_nor


def down_sample():
    """编码器 - 特征提取模型
    """
    base_model = keras.applications.MobileNetV2(input_shape=(800, 608, 3), include_top=False)

    layer_names = [
        'block_1_expand_relu',  # 400x304
        'block_3_expand_relu',  # 200x152
        'block_6_expand_relu',  # 100x76
        'block_13_expand_relu',  # 50x38
        'block_16_project',  # 25x19
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    down_stack = keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False

    return down_stack


def upsample(filters, kernel_size):
    """Upsamples an input.

    Conv2DTranspose => Batchnorm => Dropout => Relu
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = keras.Sequential()
    result.add(
        keras.layers.Conv2DTranspose(filters, kernel_size, strides=2,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     use_bias=False))
    result.add(keras.layers.BatchNormalization())
    result.add(keras.layers.Dropout(0.5))
    result.add(keras.layers.ReLU())
    return result


def unet_model(output_channels):
    """unet 模型

    :param output_channels: 输出通道数
    :return: keras model
    """
    # 输入
    inputs = keras.Input(shape=(800, 608, 3))
    x = inputs

    # 降频取样
    down_stack = down_sample()
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # 升频取样并建立跳跃连接
    up_stack = [
        upsample(128, 3),  # 25x19 -> 50x38
        upsample(64, 3),  # 50x38 -> 100x76
        upsample(32, 3),  # 100x76 -> 200x152
        upsample(16, 3),  # 200x152 -> 400x304
    ]
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = keras.layers.Concatenate()
        x = concat([x, skip])

    # 输出，模型最后一层
    last = keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same')  # 400x304 -> 800x608

    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)


def create_mask(pred_mask):
    """预测结果转换为图片
    """
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def train():
    # 创建模型
    model = unet_model(OUTPUT_CHANNELS)
    # Configures the model for training
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  # from_logits=True 表示 y_pred 未经过 softmax 层处理
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # 模型结构
    # model.summary()
    # keras.utils.plot_model(model, show_shapes=True)

    # 定义 checkpoint 和 tensorboard 回调
    ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    log_dir = tensorboard_dir + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    data_loader = DataLoader()
    train_length = data_loader.num_train_data
    steps_per_epoch = train_length // BATCH_SIZE
    validation_steps = data_loader.num_test_data // BATCH_SIZE

    # model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.fit(data_loader.train_dataset, epochs=EPOCHS,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              validation_data=data_loader.test_dataset,
              callbacks=[ckpt_callback, tb_callback])


def test():
    # 创建模型，载入权值
    model = unet_model(OUTPUT_CHANNELS)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    # model.load_weights(checkpoint_path.format(epoch=1))

    # 取出一张图片进行预测
    data_loader = DataLoader(load_train=False, load_test=True)
    # sample_image, sample_mask = next(data_loader.test_images.take(1).as_numpy_iterator())
    for sample_image, sample_mask in data_loader.test_images.take(1):
        display([sample_image, sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        print('Wrong args')
        exit(1)
