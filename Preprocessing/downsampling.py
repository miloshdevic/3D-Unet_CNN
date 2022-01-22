import numpy as np
from scipy.ndimage import interpolation

path_to_img = '/Volumes/LaCie_SSD/patients_data_training/images/'
path_to_msk = '/Volumes/LaCie_SSD/patients_data_training/masks/'
saving_path = '/Volumes/LaCie_SSD/downsampled_patients_data_training/'
nbr_patients = 132


def downsampling(nbr_patients):
    """Generates data containing batch_size samples"""

    ID = 0
    for i in range(0, nbr_patients):
        temp = np.load(saving_path + 'images/' + str(ID) + '.npy')
        print('\nPatient file: ', ID, temp.shape)
        
        # downscale CT images
        pre_resize_x = np.load(path_to_img + str(ID))
        pre_resize_x = interpolation.zoom(pre_resize_x, [128 / pre_resize_x.shape[0], 0.25, 0.25], order=5)
        print("downsample of image done")
        np.save(saving_path + 'images/' + str(ID), pre_resize_x)

        # downscale masks
        pre_resize_y = np.load(path_to_msk + str(ID))
        pre_resize_y = interpolation.zoom(pre_resize_y, [128 / pre_resize_y.shape[0], 0.25, 0.25], order=0)
        print("downsample of mask done")
        np.save(saving_path + 'masks/' + str(ID), pre_resize_y.astype('uint8'))

        print('patient ' + str(ID) + ' saved')
        ID += 1


if __name__ == '__main__':
    downsampling(nbr_patients)
