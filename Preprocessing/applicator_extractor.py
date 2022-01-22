import tqdm as tqdm
# from xlwt import Workbook

from src.Structure import *
from src.CT import *
import glob
import numpy
import matplotlib.pyplot as plt
from nibabel.nicom.dicomwrappers import *
import scipy.ndimage as ndi
import shutil
from pydicom import dcmread
import datetime
import pandas as pd
# import albumentations as A
os.chdir('//')


# User defined variables
########################################################################################################################
# folder that contains the patient folders (CHANGE)
parent_folder = '../patient_data/preprocessed_already/'
# location to save the images (CHANGE)
path = '../data/resampled_val/'
# path where data for training can be found (CHANGE)
path_for_training = '../2D_U-Net_CNN/train_data/'
# resolution that images get resampled to in vox/mm
resolution = 0.6
# window size of interest, in mm origin at center
window_in_mm = 512 * 0.6
# skip duplicate patients ?
skip_duplicates = False
# keep only area above lowest dwell ?
crop_dwells = True
# number of sanity check images to plot (one per patient)
num_to_plot = 1
# make folders for every patient in image and label folders ?
make_folders = True
########################################################################################################################


# variables used for debugging, feel free to delete them if not needed
window = int(window_in_mm/resolution)
print("WINDOW:", window)
patient_counter = 0  # I used this to continue from where I had stopped previously
patient_names = []


def get_voxel_positions_list_version(coord):

    # Luca used it, left it here just in case
    # Create a mesh grid with CT coordinates
    # Mesh grid consists of an array of x coordinates and an array of y coordinates
    # e.g.:x=[[0.5, 1, 1.5, 2, ... , 50, 50.5, 51, 51.5],
    #       [0.5, 1, 1.5, 2, ... , 50, 50.5, 51, 51.5],
    #       (...),
    #       [0.5, 1, 1.5, 2, ... , 50, 50.5, 51, 51.5],
    #       [0.5, 1, 1.5, 2, ... , 50, 50.5, 51, 51.5]]

    # these corrections need to be manually input to correct for interpolation errors.
    # if orig_spacing[1] == 1.25:
    #     correction = 1/3
    # elif orig_spacing[1] == 0.976562:
    #     correction = - 1/4
    # else:
    #     correction = 0

    # Different ways to crop the images
    # voxel positions (sometimes a correction is needed)
    x_positions = numpy.arange(coord.num_voxels[0]) * coord.spacing[0] * coord.orient[0] + coord.img_pos[0] #- correction
    y_positions = numpy.arange(coord.num_voxels[1]) * coord.spacing[1] * coord.orient[1] + coord.img_pos[1] #- correction

    # crop in center
    x_positions = x_positions[int(len(x_positions) / 2 - window / 2): int(len(x_positions) / 2 + window / 2)]
    y_positions = y_positions[int(len(y_positions) / 2 - window / 2): int(len(y_positions) / 2 + window / 2)]

    # crop top left (wasn't working properly last time I tried)
    # x_positions = x_positions[:window]
    # y_positions = y_positions[-1*window:]

    # crop top right (wasn't working properly last time I tried)
    # x_positions = x_positions[-1*window:]
    # y_positions = y_positions[-1*window:]

    position_grid = numpy.meshgrid(x_positions, y_positions)

    # flatten mesh grid (e.g. shape: (262144, 2), 262144 [x,y] coordinates)
    position_flat = numpy.array(list(zip(position_grid[0].flatten(), position_grid[1].flatten())))
    return position_flat


for folder in sorted(os.listdir(parent_folder)):

    # used because of a hidden file that was causing me some problems, you might not need this 'if' statement
    if not os.path.isdir(str(parent_folder)+"/"+str(folder)):
        continue

    if "CT" in str(os.listdir(str(parent_folder)+"/"+str(folder))):
        # read in the dicom files and print to know where it is while running the code
        dicom_folder = str(parent_folder) + "/" + str(folder)
        print(dicom_folder)
        rs_filename_list = glob.glob(dicom_folder + '/RS*')
        print(dicom_folder + '/RS*')
        rp_filename_list = glob.glob(dicom_folder + '/RP*')

        assert len(rs_filename_list) == 1, \
            "There must be one RS file, with the file name starts with RS: e.g. RSxxxxxx.dicom"
        rs_filename = rs_filename_list[0]
        rp_filename = rp_filename_list[0]
        print('rp_filename:', rp_filename)
        ct_dir = os.path.dirname(rs_filename)

        ct = CT({'ct_folder': ct_dir})
        rs = dicom.read_file(rs_filename)
        rp = dicom.read_file(rp_filename)

        zoom = (ct.spacing[0] / resolution, ct.spacing[1] / resolution,
                ct.spacing[2] / resolution)

        orig_spacing = ct.spacing

        print("RESOLUTION:", ct.spacing)  # you might need to know the resolution

        # adjust parameters for new resolution
        if not zoom == (1.0, 1.0, 1.0):
            ct.coords.slice_coordinates = ndi.interpolation.zoom(ct.coords.slice_coordinates, zoom[2])
            ct.coords.spacing = np.divide(ct.coords.spacing, zoom)
            ct.coords.num_voxels = np.multiply(ct.coords.num_voxels, zoom).astype(int)

        if skip_duplicates:
            # skip duplicate patients
            print(rs.PatientName)
            if rs.PatientName in patient_names:
                print("SKIPPING DUPLICATE PATIENT \n")
                continue
            else:
                patient_names.append(str(rs.PatientName))

        # get only the digits of the file name, used to save patients, you can save them differently
        patient_id = []
        for i in range(len(folder)):
            if folder[i].isdigit():
                patient_id.append(int(folder[i]))
        patient_id = int(''.join([str(item) for item in patient_id]))
        print("patient_id:", patient_id)

        app_grid_mask_list = []
        dwell_zs = []

        if crop_dwells:
            for index, item in enumerate(rp.ApplicationSetupSequence):
                for ind, obj in enumerate(item.ChannelSequence):
                    for thing in obj.BrachyControlPointSequence:
                        try:
                            dwell_zs.append(thing.ControlPoint3DPosition[2])  # for cropping along the z axis
                            break
                        except AttributeError:
                            print("\n", folder, "has no attribute\n")

            z_positions = numpy.arange(ct.coords.num_voxels[2]) * ct.coords.spacing[2] * ct.coords.orient[2] + \
                          ct.coords.img_pos[2]
            min_dwell_z = min(dwell_zs)

        skip = False
        if skip:
            grid_mask = numpy.zeros((ct.coords.num_voxels[2], window, window), dtype=numpy.integer)

            # loop through all structures in structure file (Structure.py)
            # for index, item in enumerate(rp[0x300f, 0x1000].value):
            for index, item in enumerate(rp[0x300f, 0x1000].value):
                first_point = True
                for ind, sequence in tqdm.tqdm(enumerate(item.ROIContourSequence)):
                    print("catheter")
                    slice_data = my_get_mask_slices(sequence, ct.coords)  # reading traced points made by physician
                    if slice_data is None:
                        print("Not a catheter")
                        continue

                    # get flattened mesh grid with coordinates
                    positions = get_voxel_positions_list_version(ct.coords)

                    # go through every slice and draw catheter paths on voxels to create a 3D reconstruction
                    # ct_voxels = ct.get_unscaled_grid(zoom=zoom, window=window, idx=idx)

                    for slice_num, paths in slice_data.items():
                        if slice_num < idx:
                            continue
                        for route in paths:
                            for contour in route:
                                contour_mask = contour.contains_points(
                                    positions).reshape((int(np.sqrt(len(positions))), int(np.sqrt(len(positions)))))

                                grid_mask[slice_num] = np.logical_or(contour_mask, grid_mask[slice_num])

                                # used for visuals
                                # if slice_num == 250:
                                #     fig, ax = plt.subplots()
                                #     # ax.imshow(ct_voxels[slice_num])
                                #     ax.imshow(grid_mask[slice_num])
                                #     # y, x = grid_mask[slice_num].nonzero()
                                #     # ax.scatter(x, y, c='red')
                                #     plt.show()

                                    # fig, ax = plt.subplots()
                                    # plt.imshow(grid_mask[slice_num])
                                    # plt.show()
                                    # fig, ax = plt.subplots(2)  # creates canvas, 2 plots and 1 figure


            # TODO: take only relevant part, needs adjusting
            grid_mask = grid_mask[idx:]  # to avoid going through unnecesseary part of the grid
                                         # aka before and after where the catheters are

            # TODO: crop top left or top right
            # ideally here I would crop depending if it is for the left or right breast

            z_values, xph, yph = grid_mask.nonzero()
            max_z_value = max(z_values)

            # The following section is how I saved the files, masks and CT images separately
            if make_folders:
                print("\ncreating folders...")
                # make a patient folder in image and in label folder
                temp_path = path + 'image/' + folder
                if not os.path.exists(temp_path):  # create only if it hasn't been done before
                    os.mkdir(path + 'image/' + folder)
                    os.mkdir(path + 'label/' + folder)
                else:
                    # remove files that were pre-processed before bc I would run the code many times and stop in the
                    # middle of it so some files were saved and instead of manually deleting them I do it here
                    for file in os.listdir(path + 'image/' + folder):
                        os.remove(os.path.join(path + 'image/' + folder, file))
                    for file in os.listdir(path + 'label/' + folder):
                        os.remove(os.path.join(path + 'label/' + folder, file))

                temp_path_training = path_for_training + str(patient_id) + '/images/'
                if not os.path.exists(temp_path_training):  # create only if it hasn't been done before
                    os.mkdir(path_for_training + str(patient_id) + '/')
                    os.mkdir(path_for_training + str(patient_id) + '/images/')
                    os.mkdir(path_for_training + str(patient_id) + '/masks/')
                else:
                    # remove files that were pre-processed before
                    for file in os.listdir(path_for_training + str(patient_id) + '/images/'):
                        os.remove(os.path.join(path_for_training + str(patient_id) + '/images/', file))
                    for file in os.listdir(path_for_training + str(patient_id) + '/masks/'):
                        os.remove(os.path.join(path_for_training + str(patient_id) + '/masks/', file))

            print("folders created")
            print("\ndoing interpolation for grid...\n")
            ct_voxels = ct.get_unscaled_grid(zoom=zoom, window=window, idx=idx)  # it's normal if it takes a lot of time

            # choose where to save them
            np.save(path_for_training + str(patient_id) + '/images/' + str(patient_counter), ct_voxels)
            np.save(path_for_training + str(patient_id) + '/masks/' + str(patient_counter), grid_mask)

            print('\n' + str(patient_counter) + ' patients saved successfully \n')
            patient_counter += 1
