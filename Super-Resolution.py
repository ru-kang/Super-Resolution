import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt
import random
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, PReLU, Add, Conv2DTranspose

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def preprocess_image(hr_image_path, scale_factor=4, noise_factor=0.05, contrast_factor=1.5):
    hr_image = tf.io.read_file(hr_image_path)
    hr_image = tf.image.decode_image(hr_image, channels=3, expand_animations=False)
    hr_image.set_shape([None, None, 3])
    hr_image = tf.image.convert_image_dtype(hr_image, tf.float32)

    # Resize images to a consistent size (example size: [256, 256])
    hr_image = tf.image.resize(hr_image, [256, 256])

    # Create the low-resolution image by reducing the size and scaling back up.
    lr_size = [hr_image.shape[0] // scale_factor, hr_image.shape[1] // scale_factor]
    lr_image = tf.image.resize(hr_image, lr_size, method='area')
    lr_image = tf.image.resize(lr_image, [256, 256], method='bicubic')

    # Add noise
    noise = tf.random.normal(shape=tf.shape(lr_image), mean=0.0, stddev=noise_factor)
    lr_image = lr_image + noise
    lr_image = tf.clip_by_value(lr_image, 0.0, 1.0)

    # Increase contrast
    lr_image = tf.image.adjust_contrast(lr_image, contrast_factor)

    return lr_image, hr_image

def load_dataset(dataset_path, subset, scale_factor=4):
    # Load all image files from the subset path.
    image_files = glob.glob(os.path.join(dataset_path, subset, '*.[jp][pn]g'))
    image_files = [f for f in image_files if not f.lower().endswith('.db')]
    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    dataset = dataset.map(lambda x: preprocess_image(x, scale_factor))

    return dataset

# Paths to the dataset.
bsd500_path = 'Data/BSD500/images'
div2k_train_path = 'Data/DIV2K/DIV2K_train_HR/DIV2K_train_HR'
div2k_valid_path = 'Data/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR'

# Loading the datasets.
scale_factor = 4
div2k_dataset_train = glob.glob(os.path.join(div2k_train_path, '*.png'))
random.seed(123)
div2k_dataset_test = random.sample(div2k_dataset_train, 100)
div2k_dataset_train = [img for img in div2k_dataset_train if img not in div2k_dataset_test]
div2k_dataset_test = tf.data.Dataset.from_tensor_slices(div2k_dataset_test)
div2k_dataset_test = div2k_dataset_test.map(lambda x: preprocess_image(x, scale_factor))
div2k_dataset_train = tf.data.Dataset.from_tensor_slices(div2k_dataset_train)
div2k_dataset_train = div2k_dataset_train.map(lambda x: preprocess_image(x, scale_factor))

bsd500_dataset_train = load_dataset(bsd500_path, 'train')
bsd500_dataset_val = load_dataset(bsd500_path, 'val')
bsd500_dataset_test = load_dataset(bsd500_path, 'test')
div2k_dataset_valid = load_dataset(div2k_valid_path, '')

# Combine the DIV2K and BSD500 datasets.
combined_dataset_train = bsd500_dataset_train.concatenate(div2k_dataset_train)
combined_dataset_val = bsd500_dataset_val.concatenate(div2k_dataset_valid)
combined_dataset_test = bsd500_dataset_test.concatenate(div2k_dataset_test)

# show images
# for lr_image, hr_image in combined_dataset_test.take(3):
#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))

#     # show low-resolution image
#     axes[0].imshow(np.clip(lr_image.numpy() * 255, 0, 255).astype(np.uint8))
#     axes[0].set_title("Low Resolution Image")
#     axes[0].axis('off')

#     # show high-resolution image
#     axes[1].imshow(np.clip(hr_image.numpy() * 255, 0, 255).astype(np.uint8))
#     axes[1].set_title("High Resolution Image")
#     axes[1].axis('off')

#     plt.show()
def build_residual_block(input_tensor):
    residual = input_tensor

    x = Conv2D(128, kernel_size=(3, 3), padding='same')(residual)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    output_tensor = Add()([x, residual])

    return output_tensor

def build_generator(input_shape):
    input_layer = Input(shape=input_shape)

    x = Conv2D(128, kernel_size=(3, 3), padding='same')(input_layer)
    x = PReLU()(x)

    for _ in range(8):
        x = build_residual_block(x)

    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = PReLU()(x)

    x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = PReLU()(x)

    output_layer = Conv2D(3, kernel_size=(3, 3), activation='tanh', padding='same')(x)

    model = Model(inputs=input_layer, outputs=output_layer, name='generator')

    return model

input_shape = (256, 256, 3)
generator = build_generator(input_shape)
# 取出組合訓練數據集中的低解析度圖像
for lr_image, _ in combined_dataset_train.take(3):
    # 將低解析度圖像輸入到生成器中
    generated_hr_image = generator.predict(tf.expand_dims(lr_image, 0))

    # 展示生成的高解析度圖像
    plt.imshow(np.clip(generated_hr_image[0] * 255, 0, 255).astype(np.uint8))
    plt.title("Generated High Resolution Image")
    plt.axis('off')
    plt.show()
generator.summary()

