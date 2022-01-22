import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
# import os
import random
from volumentations import *
import tensorflow.keras.backend as K

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

slice_num = 128
img_row = 128
img_col = 128
channels = 1
batch_size = 8

mask_shape = (slice_num, img_row, img_col, channels)

img_shape = (slice_num, img_row, img_col, 1)

root = '/home/mdevic31/scratch/data/'

def weighted_bce(y_true, y_pred):
  weights = (y_true * 59.) + 1.
  bce = K.binary_crossentropy(y_true, y_pred)
  weighted_bce = K.mean(bce * weights)
  return weighted_bce


def get_3D_UNet_generator():
    input = tf.keras.layers.Input(img_shape, batch_size=batch_size)

    # go from 128x128x128 to 64x64x64
    # filters go from 1 to 32
    conv0 = tf.keras.layers.Conv3D(16, (3, 3, 3), padding='same')(input)
    norm0 = tf.keras.layers.BatchNormalization()(conv0)
    leaky0 = tf.keras.layers.LeakyReLU()(norm0)  # (128, 128, 128, 16)
    conv05 = tf.keras.layers.Conv3D(32, (3, 3, 3), padding='same')(leaky0)
    norm05 = tf.keras.layers.BatchNormalization()(conv05) # si b=2, instance normalization ai lieu de batch normalization
    leaky05 = tf.keras.layers.LeakyReLU()(norm05)  # (128, 128, 128, 32) --> stays 128 bc of padding
    max0 = tf.keras.layers.Conv3D(32, (3, 3, 3), strides=(2, 2, 2))(leaky05) # TODO: filtre (3,3,3)

    # from 64x64x64 to 32x32x32
    conv1 = tf.keras.layers.Conv3D(32, (3, 3, 3), padding='same')(max0)
    norm1 = tf.keras.layers.BatchNormalization()(conv1)
    leaky1 = tf.keras.layers.LeakyReLU()(norm1)  # (64, 64, 64, 32)
    conv2 = tf.keras.layers.Conv3D(64, (3, 3, 3), padding='same')(leaky1)
    norm2 = tf.keras.layers.BatchNormalization()(conv2)
    leaky2 = tf.keras.layers.LeakyReLU()(norm2)  # (64, 64, 64, 64) --> stays 64 bc of padding
    max1 = tf.keras.layers.Conv3D(64, (3, 3, 3), strides=(2, 2, 2))(leaky2)  # (32, 32, 32, 64)

    # bottom layer of U-net
    conv3 = tf.keras.layers.Conv3D(128, (3, 3, 3), padding='same')(max1)
    norm3 = tf.keras.layers.BatchNormalization()(conv3)
    leaky3 = tf.keras.layers.LeakyReLU()(norm3)  # (32, 32, 32, 128)
    conv4 = tf.keras.layers.Conv3D(128, (3, 3, 3), padding='same')(leaky3)
    norm4 = tf.keras.layers.BatchNormalization()(conv4)
    leaky4 = tf.keras.layers.LeakyReLU()(norm4)  # (32, 32, 32, 128)
    convt1 = tf.keras.layers.Conv3DTranspose(64, (3, 3, 3), strides=(2, 2, 2))(leaky4)  # (64, 64, 64, 64)

    conc1 = tf.keras.layers.Concatenate(axis=-1)([convt1, leaky2])  # (64, 64, 64, 64) + (64, 64, 64, 64)
    conv5 = tf.keras.layers.Conv3D(64, (3, 3, 3), padding='same')(conc1)
    norm5 = tf.keras.layers.BatchNormalization()(conv5)
    leaky5 = tf.keras.layers.LeakyReLU()(norm5)  # (64, 64, 64, 64)
    conv6 = tf.keras.layers.Conv3D(64, (3, 3, 3), padding='same')(leaky5)
    norm6 = tf.keras.layers.BatchNormalization()(conv6)
    leaky6 = tf.keras.layers.LeakyReLU()(norm6)  # (64, 64, 64, 64)
    convt2 = tf.keras.layers.Conv3DTranspose(32, (3, 3, 3), strides=(2, 2, 2))(leaky6)  # (128, 128, 128, 32)

    conc2 = tf.keras.layers.Concatenate(axis=-1)([convt2, leaky05])  # (128, 128, 128, 32) + (128, 128, 128, 32)
    conv7 = tf.keras.layers.Conv3D(32, (3, 3, 3), padding='same')(conc2)
    norm7 = tf.keras.layers.BatchNormalization()(conv7)
    leaky7 = tf.keras.layers.LeakyReLU()(norm7)  # (128, 128, 128, 32)
    conv8 = tf.keras.layers.Conv3D(32, (3, 3, 3), padding='same')(leaky7)
    norm8 = tf.keras.layers.BatchNormalization()(conv8)
    leaky8 = tf.keras.layers.LeakyReLU()(norm8)  # (128, 128, 128, 32)

    output = tf.keras.layers.Conv3D(channels, (1, 1, 1), activation='sigmoid')(leaky8)

    model = tf.keras.Model(inputs=[input], outputs=[output])

    model.summary()

    return model


# def get_augmentations():
#     return v.Compose([v.Rotate(x_limit=(-2, 2), y_limit=(0, 0), z_limit=(0, 0), p=0.25, border_mode="nearest"),
#                       v.RandomGamma(gamma_limit=(0.9, 1.1), p=0.25)])
def augmentor(img, mask):
    aug = get_augmentation()
    data = {'image': img, 'mask': mask}
    aug_data = aug(**data)
    img, mask = aug_data['image'], aug_data['mask']
    # plt.imshow(img[60])
    # plt.show()
    # plt.imshow(mask[60])
    # plt.show()
    return img, mask

    # random data augmentation
def get_augmentation():
    return Compose([  # Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
        # Flip(0, p=0.5),
        Flip(1, p=0.5),
        Flip(2, p=0.5),
        # ElasticTransform((0, 0.25), interpolation=2, p=0.1),
        RandomRotate90((1, 2), p=0.6),
        GaussianNoise(var_limit=(0, 5), p=0.4),
        RandomGamma(gamma_limit=(0.5, 3), p=0.4)], p=1.0)

optimizer = tf.keras.optimizers.Adam(0.001, 0.9)  # 0.001 is the learning rate

generator = get_3D_UNet_generator()
generator.compile(loss=weighted_bce, optimizer=optimizer)
# patient_numbers = []


def workout(epochs, batch_size_workout):
    for epoch in range(epochs):
        mask_batch = np.zeros(shape=(batch_size_workout, slice_num, img_row, img_col, channels))
        image_batch = np.zeros(shape=(batch_size_workout, slice_num, img_row, img_col, 1))
        # fill the array with the bath for training during epoch
        past_ids = []
        for i in range(batch_size_workout):
            idx = random.randint(0, 80)
            if len(past_ids) == 0:
                pass
            else:
                while idx in past_ids:
                    print("while")
                    idx = random.randint(0, 80)
            past_ids.append(idx)
            image = np.load(root + "images/" + str(idx) + ".npy")
            # normalize
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = np.expand_dims(np.expand_dims(image, axis=-1), axis=0)
            mask = np.load(root + "masks/" + str(idx) + ".npy")
            mask = np.expand_dims(np.expand_dims(mask, axis=-1), axis=0)
            print("image:", image.shape)
            print("mask:", mask.shape)
            # aug = get_augmentations()
            # data = {'image': image, 'mask': mask}
            # aug_data = aug(**data)
            # image_batch[i], mask_batch[i] = np.reshape(np.expand_dims(aug_data['image'], axis=0),
            #                                            newshape=(1, 128, 128, 128, 1)), aug_data['mask']
            # image, mask = augmentor(image, mask)
            image_batch[i], mask_batch[i] = image, np.reshape(mask, newshape=(1, img_col, img_row, slice_num, 1))
            # patient_numbers.append(idx)
        print("before train on batch")
        g_loss = generator.train_on_batch(image_batch, mask_batch)

        # print statement to keep track of what is going on during training
        print("%d [G loss: %f]" % (epoch, g_loss))

    return


workout(epochs=1000, batch_size_workout=batch_size)  # , validation_data=)

generator.save('generator_test_1_no_augmentation.h5')