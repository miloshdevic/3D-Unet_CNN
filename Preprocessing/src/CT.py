"""
CT module.

Copyright Marc-Andre Renaud, 2017
"""
import os

import pydicom as dicom

import numpy

import scipy.ndimage as ndi

from src.CoordinateSystem import CoordinateSystem




class CT(object):
    """
    DICOM CT dataset wrapper.

    Optional:
    coords (CoordinateSystem): Coordinate system of the CT.
    num_voxels (list): Number of voxels in (x, y, z)
    img_pos (list): Coordinate of the center of the first pixel in (x, y, z)
    spacing (list): Distance between voxels in (x, y, z)
    (Optional) slice_coordinates (list): sorted z values of each slice in case of variable slice thickness
    files (list): List of CT DICOM files in the appropriate slice ordering.

    If the optional attributes are not included, then:
    ct_folder (string): Folder where CT DICOM files are saved
    """
    ct_folder = 'C:/Luca/testing'
    preprocessed_attributes = {"num_voxels", "spacing", "img_pos",
                               "orientation", "rescale_slope", "rescale_intercept",
                               "slice_coordinates", "coords", "files", "uids"}

    def __init__(self, attrs):
        """Constructor."""
        for k, v in attrs.items():
            setattr(self, k, v)

        # Preprocess CT dataset if any of these attributes are not defined.
        if not all([hasattr(self, attr) for attr in self.preprocessed_attributes]):
            # ct_folder must always be defined
            assert hasattr(self, "ct_folder")
            self._preprocess()

    def _get_file_list(self):
        file_list = os.listdir(self.ct_folder)

        return file_list

    def _preprocess(self):
        slice_ordering = []
        for ct_filename in self._get_file_list():

            ct_filepath = os.path.join(self.ct_folder, ct_filename)

            try:
                ct_file = dicom.read_file(ct_filepath, force=True)
                if ct_file.SOPClassUID.name == "MR Image Storage": #ct_file.SOPClassUID.name == "CT Image Storage": ##changed CLD 6/3/2019 to MR
                    slice_ordering.append((ct_filename, ct_file.ImagePositionPatient[2], ct_file.SOPInstanceUID))
                elif ct_file.SOPClassUID.name == "CT Image Storage": ##second if added 6/7/2019 LW
                    slice_ordering.append((ct_filename, ct_file.ImagePositionPatient[2], ct_file.SOPInstanceUID))

            except:
                pass

        slice_ordering.sort(key=lambda x: x[1])


        self.files = [x[0] for x in slice_ordering]
        self.uids = [x[2] for x in slice_ordering]

        first_image = dicom.read_file(os.path.join(self.ct_folder,
                                      slice_ordering[0][0]), force=True)
        first_z = float(slice_ordering[0][1])
        slice_thickness = slice_ordering[1][1] - first_z

        self.img_pos = [float(first_image.ImagePositionPatient[0]),
                        float(first_image.ImagePositionPatient[1]),
                        first_z]

        self.orientation = [float(x) for x in first_image.ImageOrientationPatient]
        self.spacing = [float(first_image.PixelSpacing[0]),
                        float(first_image.PixelSpacing[1]),
                        slice_thickness]
        self.num_voxels = [first_image.Columns,
                           first_image.Rows,
                           len(slice_ordering)]

        try: # try and except added by CLD 6/3/2019 to add US
            self.rescale_slope = int(first_image.RescaleSlope)
        except:
            self.rescale_slope =1
        try: # try and except added by CLD 6/3/2019 to add US
            self.rescale_intercept = int(first_image.RescaleIntercept)
        except:
            self.rescale_intercept =0

        self.slice_coordinates = [round(float(z[1]), 4) for z in slice_ordering]

        coord_dict = {
            "img_pos": self.img_pos,
            "spacing": self.spacing,
            "num_voxels": self.num_voxels,
            "orient": [self.orientation[0], self.orientation[4], 1],
            "slice_coordinates": self.slice_coordinates
        }

        self.coords = CoordinateSystem(coord_dict)

    def downsample_ct(self, factor):
        """Downsample CT grid."""
        import scipy.ndimage

        downsampled_folder = os.path.join(self.ct_folder, "downsampled")
        try:
            os.mkdir(downsampled_folder)
        except OSError:
            pass
        first_file = None
        ds_files = []
        for ct_file in self.files:
            path = os.path.join(self.ct_folder, ct_file)
            dcm = dicom.read_file(path, force=True)
            downsampled = scipy.ndimage.zoom(dcm.pixel_array, factor, order=1)
            dcm.Rows = downsampled.shape[0]
            dcm.Columns = downsampled.shape[1]
            dcm.PixelSpacing = [float(x) / factor for x in dcm.PixelSpacing]
            dcm.PixelData = downsampled.tostring()
            base_name = os.path.splitext(ct_file)[0]
            downsampled_name = base_name + "_downsampled.dcm"
            final_path = os.path.join(downsampled_folder, downsampled_name)
            dcm.save_as(final_path)

            if not first_file:
                first_file = dcm

            ds_files.append(downsampled_name)

        ds_meta = {}
        ds_meta["img_pos"] = self.img_pos
        ds_meta["spacing"] = [self.spacing[0] / factor, self.spacing[1] / factor, self.spacing[2]]
        ds_meta["num_voxels"] = [first_file.Columns, first_file.Rows, self.num_voxels[2]]
        ds_meta["slice_coordinates"] = self.slice_coordinates
        ds_meta["orientation"] = self.orientation
        ds_meta["rescale_slope"] = self.rescale_slope
        ds_meta["rescale_intercept"] = self.rescale_intercept
        ds_meta["files"] = ds_files
        ds_meta["uids"] = self.uids

        return ds_meta

    def get_slice(self, slice_num):
        """Return pixel data for CT slice."""
        ctfile_path = os.path.join(self.ct_folder, self.files[slice_num])
        ct_dicom = dicom.read_file(ctfile_path, force=True)
        pixels = ct_dicom.pixel_array.astype(numpy.float32).tostring()

        return pixels

    def get_whole_grid(self):
        """Return whole CT grid pixel data."""
        ct_grid = numpy.empty((self.num_voxels[2],
                               self.num_voxels[1],
                               self.num_voxels[0]), dtype=numpy.int16)

        for slice_num in range(self.num_voxels[2]):
            ctfile_path = os.path.join(self.ct_folder, self.files[slice_num])
            ct_dicom = dicom.read_file(ctfile_path, force=True)
            ct_grid[slice_num] = ct_dicom.pixel_array

        ct_grid = ct_grid * self.rescale_slope + self.rescale_intercept

        return ct_grid




    # TODO: add parameter to know if left or right
    def get_unscaled_grid(self, zoom=(1.0,1.0,1.0), window = None, idx = None):
        """Return raw pixel data for CT grid."""
        ct_grid = numpy.empty((self.num_voxels[2],
                               self.num_voxels[1],
                               self.num_voxels[0]), dtype=numpy.float32)

        for slice_num in range(self.num_voxels[2]):
            ctfile_path = os.path.join(self.ct_folder, self.files[slice_num])
            ct_dicom = dicom.read_file(ctfile_path, force=True)
            ct_grid[slice_num] = ct_dicom.pixel_array

        if not zoom == (1.0,1.0,1.0):
            ct_grid = ndi.interpolation.zoom(ct_grid, numpy.flip(zoom))

        if not window == None:
            ct_grid = ct_grid[:, int(ct_grid.shape[1]/2-window/2):int(ct_grid.shape[1]/2+window/2), int(ct_grid.shape[2]/2-window/2):int(ct_grid.shape[2]/2+window/2)]
            # TODO: add if statement for left/right
            # ct_grid = ct_grid[:, :window, -1*window:]  # left
            # ct_grid = ct_grid[:, -1*window:, -1 * window:]  # right

        if not idx == None:
            ct_grid = ct_grid[idx:]
        return ct_grid

    def slice_from_z(self, z):
        """Return the slice number from z coordinate."""
        slice_num = numpy.searchsorted(self.slice_coordinates, z)

        if slice_num >= len(self.slice_coordinates) - 1:
            slice_num = len(self.slice_coordinates) - 1
        elif slice_num == 0:
            if abs(z - self.slice_coordinates[slice_num]) < abs(self.slice_coordinates[slice_num + 1] - z):
                pass
            else:
                slice_num += 1
        else:
            possibilities = numpy.array([abs(z - self.slice_coordinates[slice_num - 1]),
                                         abs(z - self.slice_coordinates[slice_num]),
                                         abs(z - self.slice_coordinates[slice_num + 1])])
            slice_num = slice_num + (numpy.argmin(possibilities) - 1)

        return slice_num

    def as_dict(self):
        """Serialize as dict."""
        return {
            "name": self.uid,
            "uid": self.uid,
            "num_voxels": self.num_voxels,
            "spacing": self.spacing,
            "img_pos": self.img_pos,
            "orientation": self.orientation,
            "rescale_slope": self.rescale_slope,
            "rescale_intercept": self.rescale_intercept,
            "slice_coordinates": self.slice_coordinates,
            "files": self.files,
            "uids": self.uids
        }
