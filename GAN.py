import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt
import random
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D, LeakyReLU, Dense, Flatten, AveragePooling2D, BatchNormalization, PReLU, Add, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from keras.utils.generic_utils import Progbar

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
    lr_image = tf.image.resize(lr_image, [64, 64], method='bicubic')

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

def build_residual_block(input_tensor):
    # residual = input_tensor

    # x = Conv2D(128, kernel_size=(3, 3), padding='same')(residual)
    # x = BatchNormalization()(x)
    # x = PReLU()(x)

    # x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    # x = BatchNormalization()(x)

    # output_tensor = Add()([x, residual])

    # return output_tensor
    residual = input_tensor

    x = Conv2D(64, kernel_size=(3, 3), padding='same')(residual)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    output_tensor = Add()([x, residual])

    return output_tensor

def build_generator(input_shape):
    # input_layer = Input(shape=input_shape)

    # x = Conv2D(128, kernel_size=(3, 3), padding='same')(input_layer)
    # x = PReLU()(x)

    # for _ in range(8):
    #     x = build_residual_block(x)

    # x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    # x = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    # x = PReLU()(x)

    # x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    # x = PReLU()(x)

    # output_layer = Conv2D(3, kernel_size=(3, 3), activation='tanh', padding='same')(x)

    # model = Model(inputs=input_layer, outputs=output_layer, name='generator')

    # return model
    input_layer = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=(9, 9), padding='same')(input_layer)
    x = PReLU()(x)

    for _ in range(8):
        x = build_residual_block(x)

    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = PReLU()(x)

    x = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = PReLU()(x)

    output_layer = Conv2D(3, kernel_size=(9, 9), activation='tanh', padding='same')(x)

    model = Model(inputs=input_layer, outputs=output_layer, name='generator')

    return model

def DownSampling(input_, unit, kernel_size, strides=1, bn=True):
    x = Conv2D(unit, kernel_size=kernel_size, strides=strides, padding='same')(input_)
    if bn:
        x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def build_discriminator(input_shape):
    input_ = Input(input_shape)
    model = DownSampling(input_, unit= 64, kernel_size=3, bn=False)
    #model = DownSampling(model, unit=64, kernel_size=3, strides=2)
    model = DownSampling(model, unit=128, kernel_size=3, strides=2)
    #model = DownSampling(model, unit=128, kernel_size=3, strides=2)
    model = DownSampling(model, unit=256, kernel_size=3, strides=2)
    #model = DownSampling(model, unit=256, kernel_size=3, strides=2)
    #model = DownSampling(model, unit=512, kernel_size=3, strides=2)
    feature = DownSampling(model, unit=512, kernel_size=3, strides=2)
    """
    model = Sequential()

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    feature = model
    feature.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    feature.add()
    """
    """
    model = Sequential([
        Conv2D(128, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape),
        LeakyReLU(alpha=0.2),
        Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        AveragePooling2D(),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    """
    # 光玄大帥哥
    # model.add(Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.2))
    
    #model.add(Conv2D(1024, (3, 3), strides=(2, 2), padding='same'))
    #model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha=0.2))

    #model = AveragePooling2D()(feature)

    model = Flatten()(feature)
    # model = Dense(1024)(feature)
    model = LeakyReLU(alpha=0.2)(model)
    output = Dense(1, activation='sigmoid')(model)
    model = Model(inputs=input_, outputs=output, name='discriminator')

    fn_model = Model(inputs=input_, outputs=feature, name='fn')

    return model, fn_model

def compile_models(generator,discriminator,fn):
    discriminator.compile(optimizer=Adam(),loss='mse', metrics = ['accuracy'])
    discriminator.trainable = False
    gan_input = Input(shape=(64, 64, 3))
    print(generator(gan_input).shape)
    gen = generator(gan_input)
    gan_output = discriminator(gen)
    generator_sample_features = fn(gen)
    gan = Model(gan_input, [gan_output,generator_sample_features])
    gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5),loss=['binary_crossentropy','mse'],loss_weights=[0.001,1], metrics = ['accuracy'])
    return gan

def train_gan(generator, discriminator, gan, fn, epochs=10000,batch_size =8):
    for epoch in range(epochs):
        for lr_image, hr_image in combined_dataset_train.take(50):
            #getnum = np.random.randint(0,train_hr.shape[0],batch_size)
            #print(lr_image.shape)
            #print(hr_image.shape)
            gen_images = generator.predict(lr_image)
            high_images = hr_image 
            labels_high = np.ones((batch_size,1))
            labels_low = np.zeros((batch_size,1))
            
            #high_images=np.reshape(high_images,(1,high_images.shape[0],high_images.shape[1],high_images.shape[2]))
            d_loss_high = discriminator.train_on_batch(high_images, labels_high)
            d_loss_low = discriminator.train_on_batch(gen_images, labels_low)
            d_loss = 0.5*np.add(d_loss_high,d_loss_low)

            #geninput = train_lr[getnum]
            labels_gan = np.ones((batch_size,1))
            #lr_image=np.reshape(lr_image,(1,lr_image.shape[0],lr_image.shape[1],lr_image.shape[2]))
            image_features = fn(high_images)
            #print(type(image_features))
            g_loss = gan.train_on_batch(lr_image,[labels_gan, image_features])

        print(f"{epoch} [D Loss: {d_loss}] [G Loss: {g_loss}]")

        count_test = 0
        
        # 取出組合訓練數據集中的低解析度圖像
        for lr_image, hr_image in combined_dataset_test.take(5):
            # 將低解析度圖像輸入到生成器中
            generated_hr_image = generator.predict(lr_image)
            fig, axes = plt.subplots(1, 3, figsize=(10, 5))
            # show gen-resolution image
            axes[0].imshow(np.clip(generated_hr_image[0] * 255, 0, 255).astype(np.uint8))
            axes[0].set_title("Gen Resolution Image")
            axes[0].axis('off')
            # show low-resolution image
            axes[1].imshow(np.clip(lr_image[0].numpy() * 255, 0, 255).astype(np.uint8))
            axes[1].set_title("Low Resolution Image")
            axes[1].axis('off')
            # show high-resolution image
            axes[2].imshow(np.clip(hr_image[0].numpy() * 255, 0, 255).astype(np.uint8))
            axes[2].set_title("High Resolution Image")
            axes[2].axis('off')

            img_file = os.path.join('deepLearningClassGAN/test', 'test_'+str(count_test)+".png")
            plt.savefig(img_file)
            plt.close()
            count_test = count_test+1
        
        count_train = 0
        
        # 取出組合訓練數據集中的低解析度圖像
        for lr_image, hr_image in combined_dataset_train.take(5):
            # 將低解析度圖像輸入到生成器中
            generated_hr_image = generator.predict(lr_image)
            fig, axes = plt.subplots(1, 3, figsize=(10, 5))
            # show gen-resolution image
            axes[0].imshow(np.clip(generated_hr_image[0] * 255, 0, 255).astype(np.uint8))
            axes[0].set_title("Gen Resolution Image")
            axes[0].axis('off')
            # show low-resolution image
            axes[1].imshow(np.clip(lr_image[0].numpy() * 255, 0, 255).astype(np.uint8))
            axes[1].set_title("Low Resolution Image")
            axes[1].axis('off')
            # show high-resolution image
            axes[2].imshow(np.clip(hr_image[0].numpy() * 255, 0, 255).astype(np.uint8))
            axes[2].set_title("High Resolution Image")
            axes[2].axis('off')

            img_file = os.path.join('deepLearningClassGAN/test', 'train_'+str(count_train)+".png")
            plt.savefig(img_file)
            plt.close()
            count_train = count_train+1


# Paths to the dataset.
bsd500_path = './deepLearningClassGAN/Data/BSD500/images'
div2k_train_path = './deepLearningClassGAN/Data/DIV2K/DIV2K_train_HR/DIV2K_train_HR'
div2k_valid_path = './deepLearningClassGAN/Data/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR'
# Loading the datasets.
scale_factor = 4
div2k_dataset_train = glob.glob(os.path.join(div2k_train_path,'*.png'))
random.seed(123)
num_samples = min(100, len(div2k_dataset_train))
#print(div2k_dataset_train)
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

combined_dataset_train = combined_dataset_train.batch(8)
combined_dataset_val = combined_dataset_val.batch(8)
combined_dataset_test = combined_dataset_test.batch(8)

#print(combined_dataset_train.take(3))
#train_lr = []
#train_hr = []
#for lr_image, hr_image in combined_dataset_train:
    
#    train_lr.append(lr_image)
#    train_hr.append(hr_image)

#train_lr = np.asarray(train_lr)
#train_hr = np.asarray(train_hr)

g_input_shape = (64, 64, 3)
generator = build_generator(g_input_shape)

# 取出組合訓練數據集中的低解析度圖像


#generator.summary()

d_input_shape = (256,256, 3)
discriminator, fn = build_discriminator(d_input_shape)

gan = compile_models(generator,discriminator,fn)

train_gan(generator, discriminator, gan, fn)
"""
# 取出組合訓練數據集中的低解析度圖像
for lr_image, _ in combined_dataset_train.take(3):
    # 將低解析度圖像輸入到生成器中
    generated_hr_image = generator.predict(tf.expand_dims(lr_image, 0))

    # 展示生成的高解析度圖像
    plt.imshow(np.clip(generated_hr_image[0] * 255, 0, 255).astype(np.uint8))
    plt.title("Generated High Resolution Image")
    plt.axis('off')
    img_file = os.path.join('deepLearningClassGAN/test', '1'+".png")
    plt.savefig(img_file)
    plt.close()
"""



"""
 #show images
for lr_image, hr_image in combined_dataset_test.take(3):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # show low-resolution image
    axes[0].imshow(np.clip(lr_image.numpy() * 255, 0, 255).astype(np.uint8))
    axes[0].set_title("Low Resolution Image")
    axes[0].axis('off')
    # show high-resolution image
    axes[1].imshow(np.clip(hr_image.numpy() * 255, 0, 255).astype(np.uint8))
    axes[1].set_title("High Resolution Image")
    axes[1].axis('off')
    plt.show()

"""
