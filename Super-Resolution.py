import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt

def preprocess_image(hr_image_path, scale_factor=4):
    hr_image = tf.io.read_file(hr_image_path)
    hr_image = tf.image.decode_image(hr_image, channels=3)
    hr_image.set_shape([None, None, 3])
    hr_image = tf.image.convert_image_dtype(hr_image, tf.float32)

    # Get the size of the image.
    hr_size = tf.shape(hr_image)[:2]

    # Create the low-resolution image by reducing the size and then scaling back up.
    lr_size = hr_size // scale_factor
    lr_image = tf.image.resize(hr_image, lr_size, method='area')
    
    # Resize the low-resolution image to match the original high-resolution size.
    lr_image = tf.image.resize(lr_image, hr_size, method='bicubic')
    
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
div2k_train_path = 'Data/DIV2K/DIV2K_train_HR'
div2k_valid_path = 'Data/DIV2K/DIV2K_valid_HR'

# Loading the datasets.
bsd500_dataset_train = load_dataset(bsd500_path, 'train')
bsd500_dataset_val = load_dataset(bsd500_path, 'val')
bsd500_dataset_test = load_dataset(bsd500_path, 'test')
div2k_dataset_train = load_dataset(div2k_train_path, '')
div2k_dataset_valid = load_dataset(div2k_valid_path, '')

# Combine the DIV2K and BSD500 datasets.
combined_dataset_train = bsd500_dataset_train.concatenate(div2k_dataset_train)
combined_dataset_val = bsd500_dataset_val.concatenate(div2k_dataset_valid)

for lr_image, hr_image in combined_dataset_train.take(2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # show low-resolution image
    axes[0].imshow(lr_image.numpy())
    axes[0].set_title("Low Resolution Image")
    axes[0].axis('off')

    # show high-resolution image
    axes[1].imshow(hr_image.numpy())
    axes[1].set_title("High Resolution Image")
    axes[1].axis('off')

    plt.show()
