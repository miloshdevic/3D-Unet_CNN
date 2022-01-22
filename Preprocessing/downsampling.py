import numpy as np
import tensorflow as tf
from scipy.ndimage import interpolation

path_to_img = '/Volumes/LaCie_SSD/patients_data_training/images/'
path_to_msk = '/Volumes/LaCie_SSD/patients_data_training/masks/'
saving_path = '/Volumes/LaCie_SSD/downsampled_patients_data_training/'
list_IDs = ['0.npy', '1.npy', '2.npy', '3.npy', '4.npy', '5.npy', '6.npy', '7.npy', '8.npy', '9.npy', '10.npy',
            '11.npy',
            '12.npy', '13.npy', '14.npy', '15.npy', '16.npy', '17.npy', '18.npy', '19.npy', '20.npy', '21.npy',
            '22.npy', '23.npy',
            '24.npy', '25.npy', '26.npy', '27.npy', '28.npy', '29.npy', '30.npy', '31.npy', '32.npy', '33.npy',
            '34.npy', '35.npy',
            '36.npy', '37.npy', '38.npy', '39.npy', '40.npy', '41.npy', '42.npy', '43.npy', '44.npy', '45.npy',
            '46.npy', '47.npy',
            '48.npy', '49.npy', '50.npy', '51.npy', '52.npy', '53.npy', '54.npy', '55.npy', '56.npy', '57.npy',
            '58.npy', '59.npy',
            '60.npy', '61.npy', '62.npy', '63.npy', '64.npy', '65.npy', '66.npy', '67.npy', '68.npy', '69.npy',
            '70.npy', '71.npy',
            '72.npy', '73.npy', '74.npy', '75.npy', '76.npy', '77.npy', '78.npy', '79.npy', '80.npy', '81.npy',
            '82.npy', '83.npy',
            '84.npy', '85.npy', '86.npy', '87.npy', '88.npy', '89.npy', '90.npy', '91.npy', '92.npy', '93.npy',
            '94.npy', '95.npy',
            '96.npy', '97.npy', '98.npy', '99.npy', '100.npy', '101.npy', '102.npy', '103.npy', '104.npy', '105.npy',
            '106.npy', '107.npy',
            '108.npy', '109.npy', '110.npy', '111.npy', '112.npy', '113.npy', '114.npy', '115.npy', '116.npy',
            '117.npy', '118.npy', '119.npy',
            '120.npy', '121.npy', '122.npy', '123.npy', '124.npy', '125.npy', '126.npy', '127.npy', '128.npy',
            '129.npy', '130.npy', '131.npy', '132.npy']


def downsampling(list_ids):
    """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
    # Initialization
    # x = np.empty((32, (512, 512, 512)))  # , self.n_channels
    # y = np.empty((32, (512, 512, 512)))

    for i, ID in enumerate(list_ids):
        temp = np.load(saving_path + 'images/' + ID)
        print('\nPatient file: ', ID, temp.shape)
        #cinput()
        # downscale CT images
        pre_resize_x = np.load(path_to_img + ID)
        print(pre_resize_x.shape)
        pre_resize_x = interpolation.zoom(pre_resize_x, [128 / pre_resize_x.shape[0], 0.25, 0.25], order=5)
        print("downsample of image done")
        # x[i] = pre_resize_x
        np.save(saving_path + 'images/' + ID, pre_resize_x)

        # downscale masks
        pre_resize_y = np.load(path_to_msk + ID)
        pre_resize_y = interpolation.zoom(pre_resize_y, [128 / pre_resize_y.shape[0], 0.25, 0.25], order=0)
        print("downsample of mask done")
        # y[i] = pre_resize_y
        np.save(saving_path + 'masks/' + ID, pre_resize_y.astype('uint8'))

        print('patient ' + ID + ' saved')

        # # convert numpy arrays into tensors
        # img_tf = tf.convert_to_tensor(pre_resize_x)
        # msk_tf = tf.convert_to_tensor(pre_resize_y)
        #
        # # save tensors
        # tf.data.experimental.save(img_tf, saving_path + 'images/')
        # tf.data.experimental.save(msk_tf, saving_path + 'masks/')


if __name__ == '__main__':
    downsampling(list_IDs)
