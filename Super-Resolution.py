import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt
import random
import numpy as np

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
