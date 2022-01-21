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

# GOOD FILE APPLICATOR EXTRACTOR

# test_file_ct = dcmread("/Users/miloshdevic/Documents/Internship Summer 2021/Breast_catheter_digitization-main/patient_data/val_patients/Manual_Anonymized1013990/RP1.3.6.1.4.1.2452.6.3585239483.1311726841.2018464430.3441996844.dcm")
# print(test_file_ct)

# input()

# User defined variables
########################################################################################################################
# folder that contains the patient folders
parent_folder = '/Users/miloshdevic/Documents/Internship Summer 2021/Breast_catheter_digitization-main/patient_data/preprocessed_already/'
# location to save the images
path = '/Users/miloshdevic/Documents/Internship Summer 2021/Breast_catheter_digitization-main/data/resampled_val/'
# path where data for training can be found
path_for_training = '/Users/miloshdevic/Documents/Internship Summer 2021/2D_U-Net_CNN/train_data/'
# resolution that images get resampled to in vox/mm
resolution = 0.6  # 0.7421875  # 0.9765625
# window size of interest, in mm origin at center
window_in_mm = 512 * 0.6  # 0.7421875  # 0.9765625
# skip duplicate patients ?
skip_duplicates = False
# keep only area above lowest dwell ?
crop_dwells = True  # changed
# number of sanity check images to plot (one per patient)
num_to_plot = 1
# make folders for every patient in image and label folders ?
make_folders = True
########################################################################################################################


# init variables
window = int(window_in_mm/resolution)
print("WINDOW:", window)
patient_counter = 97
patient_names = []


def get_voxel_positions_list_version(coord):

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

    # voxel positions
    x_positions = numpy.arange(coord.num_voxels[0]) * coord.spacing[0] * coord.orient[0] + coord.img_pos[0] #- correction
    y_positions = numpy.arange(coord.num_voxels[1]) * coord.spacing[1] * coord.orient[1] + coord.img_pos[1] #- correction

    # crop in center
    x_positions = x_positions[int(len(x_positions) / 2 - window / 2): int(len(x_positions) / 2 + window / 2)]
    y_positions = y_positions[int(len(y_positions) / 2 - window / 2): int(len(y_positions) / 2 + window / 2)]

    # crop top left
    # x_positions = x_positions[:window]
    # y_positions = y_positions[-1*window:]

    # crop top right
    # x_positions = x_positions[-1*window:]
    # y_positions = y_positions[-1*window:]

    position_grid = numpy.meshgrid(x_positions, y_positions)

    # flatten mesh grid (e.g. shape: (262144, 2), 262144 [x,y] coordinates)
    position_flat = numpy.array(list(zip(position_grid[0].flatten(), position_grid[1].flatten())))
    return position_flat


jsp = -1
idx_tab = []
for folder in sorted(os.listdir(parent_folder)):
    print("FOLDER PATIENT:", folder, " --- ", jsp)
    jsp += 1
    # print(str(os.listdir(str(parent_folder)+"/"+str(folder))))
    # if folder == "Patient1123625_breast" or folder == "Patient1150395_breast" or folder == "Patient1129426_breast" or \
    #         folder == "Patient1163303_breast" or folder == "Patient1167439_breast" or folder == "Patient1194868_breast" \
    #         or folder == "Patient1220091_breast" or folder == "Patient1220982_breast" or \
    #         folder == "Patient1225321_breast" or folder == "Patient1248358_breast" or folder == "Patient1252849_breast" \
    #         or folder == "Patient1289500_breast" or folder == "Patient1294955_breast" or \
    #         folder == "Patient1295004_breast" or folder == "Patient13380_breast" or folder == "Patient167205_breast" or\
    #         folder == "Patient19804_breast" or folder == "Patient1_breast" or folder == "Patient200371_breast" or \
    #         folder == "Patient296641_breast" or folder == "Patient304821_breast" or folder == "Patient343160_breast" or\
    #         folder == "Patient346867_breast" or folder == "Patient358193_breast" or folder == "Patient466823_breast" or\
    #         folder == "Patient630860_breast" or folder == "Patient638532_breast" or folder == "Patient718830_breast" or\
    #         folder == "Patient776327_breast" or folder == "Patient815838_breast" or folder == "Patient839776_breast" or\
    #         folder == "Patient840481_breast" or folder == "Patient852608_breast" or folder == "Patient854988_breast" or\
    #         folder == "Patient922445_breast" or folder == "Patient997647_breast":
    #     # patients with problems: breast1, 3, 4, 6 (prostate)
    #     continue

    # if folder.startswith("Patient1") or folder.startswith("Patient2") or folder.startswith("Patient3") or \
    #         folder.startswith("Patient4") or folder.startswith("Patient5"):
    #     continue
    # if folder == "Patient1123625_breast":  # doesn't work   or folder == "Patient1129426_breast"
    #     continue
    # if folder == "Patient1167439_breast":  # patient with 2 RS files
    #     continue
    # if folder == "Patient1123625_breast":  # patient with 2 RS files
    #     continue
    # # if folder == "Patient1129426_breast":  # patient with 2 RS files
    # #     continue
    # if folder == "Patient1_breast":  # patient with 2 RS files
    #     continue

    # if folder == '273000':
    #     patient_counter = 44
    # else:
    #     continue

    if not os.path.isdir(str(parent_folder)+"/"+str(folder)):
        continue
    if "CT" in str(os.listdir(str(parent_folder)+"/"+str(folder))):
        # read in the dicom files
        dicom_folder = str(parent_folder) + "/" + str(folder)
        print(dicom_folder)
        rs_filename_list = glob.glob(dicom_folder + '/RS*')
        print(dicom_folder + '/RS*')
        rp_filename_list = glob.glob(dicom_folder + '/RP*')
        print("length rs_filename_list: ", len(rs_filename_list))
        print("length rp_filename_list: ", len(rp_filename_list))
        print(rs_filename_list)
        print(rp_filename_list)
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

        print("RESOLUTION:", ct.spacing)

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

        # get only the digits of the file name
        patient_id = []
        for i in range(len(folder)):
            if folder[i].isdigit():
                patient_id.append(int(folder[i]))
        print("patient_id:", patient_id)
        patient_id = int(''.join([str(item) for item in patient_id]))
        print("patient_id:", patient_id)

        app_grid_mask_list = []
        dwell_zs = []

        if crop_dwells:
            # ROINum_list = []
            for index, item in enumerate(rp.ApplicationSetupSequence):
                for ind, obj in enumerate(item.ChannelSequence):
                    for thing in obj.BrachyControlPointSequence:
                        # print("length:", len(obj.BrachyControlPointSequence))
                        # print("control points tables:", thing.ControlPoint3DPosition)
                        try:
                            dwell_zs.append(thing.ControlPoint3DPosition[2])  # for cropping along the z axis
                            break
                        except AttributeError:
                            print("\n", folder, "has no attribute\n")
                            idx = 241

                    # ROINum_list.append(str(obj.ReferencedROINumber))  # used to avoid duplicates, error: always 'None'

            z_positions = numpy.arange(ct.coords.num_voxels[2]) * ct.coords.spacing[2] * ct.coords.orient[2] + \
                          ct.coords.img_pos[2]

            min_dwell_z = min(dwell_zs)
            idx = (np.abs(z_positions - min_dwell_z)).argmin()
        else:
            idx = 0
        print("\n", folder, " -- " , idx, "\n")
        idx_tab.append(idx)
        skip = False
        if skip:
            grid_mask = numpy.zeros((ct.coords.num_voxels[2], window, window), dtype=numpy.integer)

            # loop through all structures in structure file
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


            # TODO: take only relevant part
            grid_mask = grid_mask[idx:]  # to avoid going through unnecesseary part of the grid

            # TODO: crop top left or top right

            z_values, xph, yph = grid_mask.nonzero()
            max_z_value = max(z_values)

            if make_folders:
                print("\ncreating folders...")
                # make a patient folder in image and in label folder
                temp_path = path + 'image/' + folder
                print('temp_path:', temp_path)
                if not os.path.exists(temp_path):  # create only if it hasn't been done before
                    os.mkdir(path + 'image/' + folder)
                    os.mkdir(path + 'label/' + folder)
                else:
                    # remove files that were pre-processed before
                    for file in os.listdir(path + 'image/' + folder):
                        os.remove(os.path.join(path + 'image/' + folder, file))
                    for file in os.listdir(path + 'label/' + folder):
                        os.remove(os.path.join(path + 'label/' + folder, file))

                temp_path_training = path_for_training + str(patient_id) + '/images/'
                print('temp_path_training:', temp_path_training)
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
            ct_voxels = ct.get_unscaled_grid(zoom=zoom, window=window, idx=idx)

            # TODO
            #np.save(path_for_training + str(patient_id) + '/images/' + str(patient_counter), ct_voxels)
            #np.save(path_for_training + str(patient_id) + '/masks/' + str(patient_counter), grid_mask)

            np.save(path_for_training + '/all_imgs/' + str(patient_counter), ct_voxels)
            np.save(path_for_training + '/all_msks/' + str(patient_counter), grid_mask)


            # ct_voxels = ct.get_whole_grid()
            # ID_num = 0
            # for num_slice in range(0, max_z_value + 10):
            #     print("in loop for visualization")
            #     ct_img = ct_voxels[num_slice]  # possible IndexError: index 122 is out of bounds for axis 0 with size 122
            #     label_img = grid_mask[num_slice].astype(int)
            #     fig, ax = plt.subplots()
            #     ax.imshow(ct_img)
            #     y, x = label_img.nonzero()
            #     ax.scatter(x, y, c='red')
            #     fig.set_size_inches(19, 10.5)
            #     # plt.savefig("grid_images//img{}.png".format(num_slice))
            #     # plt.show()
            #
            # # for num_slice in range(0, max_z_value + 10):
            # #     print("in num_slice for loop")
            # #     ct_img = ct_voxels[num_slice]
            # #     label_img = grid_mask[num_slice].astype(int)
            # #
            # #     if patient_number < num_to_plot:
            # #         print("first if")
            # #         if num_slice == 50:
            # #             fig, ax = plt.subplots()
            # #             ax.imshow(ct_img)
            # #             y, x = label_img.nonzero()
            # #             ax.scatter(x, y, c='red')
            # #             fig.set_size_inches(19, 10.5)
            # #             # plt.show()
            # #             if folder == "Patient1_breast":
            # #                 plt.show()
            #
            #     if make_folders:
            #         print("seconf if")
            #         np.save(path + 'image/' + folder + '/' + str(ID_num), ct_img)
            #         np.save(path + 'label/' + folder + '/' + str(ID_num), label_img)
            #         np.save(path_for_training + str(patient_id) + '/images/' + str(ID_num), ct_img)
            #         np.save(path_for_training + str(patient_id) + '/masks/' + str(ID_num), label_img)
            #     else:
            #         print("second else")
            #         np.save(path + 'image/' + str(ID_num), ct_img)
            #         np.save(path + 'label/' + str(ID_num), label_img)
            #         np.save(path_for_training + str(patient_id) + '/images/' + str(ID_num), ct_img)
            #         np.save(path_for_training + str(patient_id) + '/masks/' + str(ID_num), label_img)
            #
            #     ID_num += 1

            patient_counter += 1
            print('\n' + str(patient_counter) + ' patients saved successfully \n')

            source = str(parent_folder + folder)
            dest = '/Users/miloshdevic/Documents/Internship Summer 2021/Breast_catheter_digitization-main/patient_data/' \
                   'preprocessed_already'
            shutil.move(source, dest)
            print('Folder moved')

print("\n\nmax idx", max(idx_tab))
