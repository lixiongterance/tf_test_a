import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', help='train or test')
args = parser.parse_args()

# 参数定义
EPOCHS = 10
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 1000
OUTPUT_CHANNELS = 3  # 输出结果类别个数
checkpoint_path = './checkpoints/cp.ckpt'


def display(display_list):
    # plt.figure(figsize=(15, 15))
    plt.figure()

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


class DataLoader:
    def __init__(self, load_train=True, load_test=True):
        self.dataset, self.info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
        if load_train:
            self.train_images = self.dataset['train'].map(DataLoader.load_image_train,
                                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
            self.train_dataset = self.train_images.cache().shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).repeat()
            self.train_dataset = self.train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        if load_test:
            self.test_images = self.dataset['test'].map(DataLoader.load_image_test,
                                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
            self.test_dataset = self.test_images.batch(BATCH_SIZE)

    @staticmethod
    def show_sample(data, num=1):
        for image, mask in data.take(num):
            sample_image, sample_mask = image, mask
            display([sample_image, sample_mask])

    @staticmethod
    def normalize(input_image, input_mask):
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask -= 1
        return input_image, input_mask

    @staticmethod
    def load_image_train(datapoint):
        input_image = tf.image.resize(datapoint['image'], (128, 128))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        input_image, input_mask = DataLoader.normalize(input_image, input_mask)

        return input_image, input_mask

    @staticmethod
    def load_image_test(datapoint):
        input_image = tf.image.resize(datapoint['image'], (128, 128))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

        input_image, input_mask = DataLoader.normalize(input_image, input_mask)

        return input_image, input_mask


def down_sample():
    """编码器 - 特征提取模型
    """

    base_model = keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False)

    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    down_stack = keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False

    return down_stack


def upsample(filters, size):
    """Upsamples an input.

    参考 pix2pix upsample 方法:
      https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py

    Conv2DTranspose => Batchnorm => Dropout => Relu

    Args:
      filters: number of filters
      size: filter size

    Returns:
      Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = keras.Sequential()
    result.add(
        keras.layers.Conv2DTranspose(filters, size, strides=2,
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
    inputs = keras.Input(shape=(128, 128, 3))
    x = inputs

    # 降频取样
    down_stack = down_sample()
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # 升频取样并建立跳跃连接
    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),  # 32x32 -> 64x64
    ]
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = keras.layers.Concatenate()
        x = concat([x, skip])

    # 输出，模型最后一层
    last = keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same')  # 64x64 -> 128x128

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
    model.compile(optimizer='adam',
                  # from_logits=True 表示 y_pred 未经过 softmax 层处理
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # 模型结构保存为图片
    model.summary()
    # keras.utils.plot_model(model, show_shapes=True)

    # 定义回调，保存训练权值
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)
    data_loader = DataLoader()
    train_length = data_loader.info.splits['train'].num_examples
    steps_per_epoch = train_length // BATCH_SIZE
    val_subsplits = 5
    validation_steps = data_loader.info.splits['test'].num_examples // BATCH_SIZE // val_subsplits

    model_history = model.fit(data_loader.train_dataset, epochs=EPOCHS,
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=validation_steps,
                              validation_data=data_loader.test_dataset,
                              callbacks=[cp_callback])


def test():
    # 创建模型，载入权值
    model = unet_model(OUTPUT_CHANNELS)
    model.load_weights(checkpoint_path)

    # 取出一张图片进行预测
    data_loader = DataLoader(load_train=False, load_test=True)
    sample_image, sample_mask = next(data_loader.test_images.take(1).as_numpy_iterator())
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
