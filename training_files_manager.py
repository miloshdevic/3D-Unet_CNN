import os
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

TRAIN_PATH = '/Users/miloshdevic/Documents/Internship Summer 2021/2D_U-Net_CNN/train_data/'
TEST_PATH = '/Users/miloshdevic/Documents/Internship Summer 2021/2D_U-Net_CNN/test_data/'
VAL_PATH = '/Users/miloshdevic/Documents/Internship Summer 2021/2D_U-Net_CNN/validation_data/'

os.makedirs('my_train', exist_ok=True)
os.makedirs('my_train/images', exist_ok=True)
# os.makedirs('my_train/images/applicators', exist_ok=True)
os.makedirs('my_val/images', exist_ok=True)
os.makedirs('my_train/masks', exist_ok=True)
# os.makedirs('my_train/masks/applicators', exist_ok=True)
os.makedirs('my_val/masks', exist_ok=True)

train_ids = next(os.walk(TRAIN_PATH))[1]
val_ids = next(os.walk(VAL_PATH))[1]

# ID_img = 0
# ID_msk = 0
# counter = 0
# for patient in train_ids:
#     # replace the . with your starting directory
#     for root, dirs, files in os.walk('/Users/miloshdevic/Documents/Internship Summer 2021/2D_U-Net_CNN/train_data/' + \
#                                      patient + '/images/'):
#
#         for file in sorted(files):
#             print(file)
#             path_file = os.path.join(root, file)
#             if not file[-4:] == '.npy':
#                 continue
#             if not file.__contains__('.npy'):
#                 continue
#             shutil.copy(path_file, "validation_data/images/" + str(ID_img) + ".npy")
#             # data = np.load(path_file, allow_pickle=True)
#             # # im = Image.fromarray(data)
#             # # im.convert('RGB').save("my_train/images/applicators/" + str(ID) + ".jpeg")
#             # plt.imsave("my_train/images/applicators/" + str(ID_img) + ".jpeg", data, cmap='gray')
#             ID_img += 1
#             # shutil.copy2(path_file, 'my_train/images/')  # change you destination dir
#
#     for root, dirs, files in os.walk('/Users/miloshdevic/Documents/Internship Summer 2021/2D_U-Net_CNN/train_data/' + \
#                                      patient + '/masks/'):
#         for file in sorted(files):
#             path_file = os.path.join(root, file)
#             if not file.__contains__('.npy'):
#                 continue
#             shutil.copy(path_file, "validation_data/masks/" + str(ID_msk) + ".npy")
#             # data = np.load(path_file, allow_pickle=True)
#             # # im = Image.fromarray(data)
#             # # im.convert('RGB').save("my_train/masks/applicators/" + str(ID) + ".jpeg")
#             # plt.imsave("my_train/masks/applicators/" + str(ID_msk) + ".jpeg", data, cmap='gray')
#             ID_msk += 1
#             # shutil.copy2(path_file, 'my_train/masks/')  # change you destination dir
#     counter += 1
#     if counter == 2:
#         break

ID_img = 0
ID_msk = 0
counter = 0
for patient in val_ids:
    # replace the . with your starting directory
    for root, dirs, files in os.walk('/Users/miloshdevic/Documents/Internship Summer 2021/2D_U-Net_CNN/validation_data/' + \
                                     patient + '/images/'):

        for file in sorted(files):
            print(file)
            path_file = os.path.join(root, file)
            if not file[-4:] == '.npy':
                continue
            if not file.__contains__('.npy'):
                continue
            shutil.copy(path_file, "my_val/images/" + str(ID_img) + ".npy")
            # data = np.load(path_file, allow_pickle=True)
            # # im = Image.fromarray(data)
            # # im.convert('RGB').save("my_train/images/applicators/" + str(ID) + ".jpeg")
            # plt.imsave("my_train/images/applicators/" + str(ID_img) + ".jpeg", data, cmap='gray')
            ID_img += 1
            # shutil.copy2(path_file, 'my_train/images/')  # change you destination dir

    for root, dirs, files in os.walk('/Users/miloshdevic/Documents/Internship Summer 2021/2D_U-Net_CNN/validation_data/' + \
                                     patient + '/masks/'):
        for file in sorted(files):
            path_file = os.path.join(root, file)
            if not file.__contains__('.npy'):
                continue
            shutil.copy(path_file, "my_val/masks/" + str(ID_msk) + ".npy")
            # data = np.load(path_file, allow_pickle=True)
            # # im = Image.fromarray(data)
            # # im.convert('RGB').save("my_train/masks/applicators/" + str(ID) + ".jpeg")
            # plt.imsave("my_train/masks/applicators/" + str(ID_msk) + ".jpeg", data, cmap='gray')
            ID_msk += 1
            # shutil.copy2(path_file, 'my_train/masks/')  # change you destination dir
    counter += 1
    if counter == 2:
        break
