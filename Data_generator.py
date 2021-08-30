import copy
import numpy as np
import tensorflow.keras
import tensorflow as tf
import os
# import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt

from skimage.transform import *
from scipy.ndimage import interpolation
from volumentations import *
from tensorflow.keras.utils import Sequence, to_categorical


class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, dim=(128, 128, 128), n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        img, msk = self.__data_generation(list_IDs_temp)

        return img, msk

    def augmentor(self, img, mask):
        aug = self.get_augmentation()
        data = {'image': img, 'mask': mask}
        aug_data = aug(**data)
        img, mask = aug_data['image'], aug_data['mask']
        # plt.imshow(img[60])
        # plt.show()
        # plt.imshow(mask[60])
        # plt.show()
        return img, mask

    # random data augmentation TODO: choose probabilities
    def get_augmentation(self):
        return Compose([  # Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
            # Flip(0, p=0.5),
            Flip(1, p=0.5),
            Flip(2, p=0.5),
            # ElasticTransform((0, 0.25), interpolation=2, p=0.1),
            RandomRotate90((1, 2), p=0.5),
            GaussianNoise(var_limit=(0, 5), p=0.2),
            RandomGamma(gamma_limit=(0.5, 3), p=0.4)], p=1.0)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((self.batch_size, *self.dim))  # , self.n_channels
        y = np.empty((self.batch_size, *self.dim))

        # Generate data
        # for i, ID in enumerate(list_IDs_temp):
        #     # Store sample
        #     x[i, ] = np.load('train_data/1013990/images/' + ID + '.npy')
        #     # resize(x[i], (128, 128, 128), anti_aliasing=True)
        #     x[i] = zoom(x[i], np.array((128, 128, 128)) / x[i].shape, order=0)
        #     x[i] = to_categorical(x[i], 2)
        #
        #     # Store class
        #     y[i] = np.load('train_data/1013990/masks/' + ID + '.npy')  # self.labels[ID]
        #     # resize(y[i], (128, 128, 128), anti_aliasing=True)
        #     y[i] = zoom(y[i], np.array((1, 128, 128, 128)) / y[i].shape, order=0)

        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            pre_resize_x = np.load('train_data/all_imgs/' + ID)
            pre_resize_x = interpolation.zoom(pre_resize_x, [64 / pre_resize_x.shape[0], 0.5, 0.5], order=5)
            x[i] = pre_resize_x

            # Store class
            pre_resize_y = np.load('train_data/all_msks/' + ID)
            # plt.imshow(pre_resize_y[250])
            # plt.show()
            pre_resize_y = interpolation.zoom(pre_resize_y, [64 / pre_resize_y.shape[0], 0.5, 0.5], order=0)
            y[i] = pre_resize_y
            # print(pre_resize_y.shape)
            # plt.imshow(pre_resize_y[31])
            # plt.show()
            img_augmented_np, mask_augmented_np = self.augmentor(pre_resize_x, pre_resize_y)

            # convert numpy arrays to tensors
            img_tf = tf.convert_to_tensor(img_augmented_np)
            x[i] = img_tf
            msk_tf = tf.convert_to_tensor(mask_augmented_np)
            y[i] = msk_tf

            print('patient file:', ID)

            # augment the images
            # img_np = np.load('train_data/all_imgs/44.npy')  # + ID)
            # msk_np = np.load('train_data/all_msks/44.npy')  # + ID)
            # # plt.imshow(img_np[60])
            # # plt.show()
            # # plt.imshow(msk_np[60])
            # # plt.show()
            # img_augmented_np, mask_augmented_np = self.augmentor(img_np, msk_np)

            # convert numpy arrays to tensors
            # img_tf = tf.convert_to_tensor(img_augmented_np)
            # x[i] = img_tf
            # msk_tf = tf.convert_to_tensor(mask_augmented_np)
            # y[i] = msk_tf

        return x, y  # tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)
