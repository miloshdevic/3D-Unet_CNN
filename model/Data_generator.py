import tensorflow as tf
import tensorflow.keras
from scipy.ndimage import interpolation
from volumentations import *


class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=8, dim=(128, 128, 128), n_channels=1,
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
        return img, mask

    # random data augmentation
    def get_augmentation(self):
        return Compose([  # Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
            # Flip(0, p=0.5),
            Flip(1, p=0.5),
            Flip(2, p=0.5),
            # ElasticTransform((0, 0.25), interpolation=2, p=0.1),
            RandomRotate90((1, 2), p=0.6),
            GaussianNoise(var_limit=(0, 5), p=0.4),
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
        print("\nIN THE DATA GENERATION\n")

        for i, ID in enumerate(list_IDs_temp):

            print('patient file:', ID)

            # augment the images
            img_np = np.load('/home/mdevic31/scratch/data/images/' + ID)
            msk_np = np.load('/home/mdevic31/scratch/data/masks/' + ID)

            # normalization
            img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))
            x[i] = img_np
            y[i] = msk_np

        return x, y
