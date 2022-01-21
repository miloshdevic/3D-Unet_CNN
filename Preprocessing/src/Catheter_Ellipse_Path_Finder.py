import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import copy


def get_catheter_angle(start_point, end_point):
    # angle measured between the catheter and the z-plane in deg
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    dz = end_point[2] - start_point[2]
    dxy = np.sqrt(np.square(dx) + np.square(dy))
    if not dxy == 0:
        angle = np.rad2deg(np.arctan(abs(dz / dxy)))
    else:
        angle = 90
    return angle


def get_points_between_catheter_points(start_point, end_point, slice_thickness):
    # not including the endpoint (to avoid repetition)
    points = []
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    dz = end_point[2] - start_point[2]
    slices_in_between = int(np.abs(dz) / slice_thickness)
    for slice_number in range(slices_in_between):
        points.append([start_point[0] + dx / slices_in_between * slice_number,
                       start_point[1] + dy / slices_in_between * slice_number,
                       start_point[2] + dz / slices_in_between * slice_number])
    return points  # matrix


# points_start_end matrix (n x 2)
def get_ellipse_paths(points_start_end, slice_thickness):
    path_dict = {}

    # Extend the catheter by 4mm
    # select most superior pair of points (assuming that it is the last pair)
    start_point = points_start_end[0][0]
    end_point = points_start_end[0][1]
    # azimuthal_angle (rad)
    azimuthal_angle = np.deg2rad(90.0 - get_catheter_angle(start_point, end_point))

    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]

    # polar angle on the z_plane/slice
    if dx == 0:
        if dy < 0:
            polar_angle = -np.pi / 2
        else:
            polar_angle = np.pi / 2
    else:
        polar_angle = np.arctan2(dy, dx)

    # find the x, y and z coordinates of the extended catheter tip
    # tip_x = end_point[0] + np.cos(polar_angle) * np.sin(azimuthal_angle) * 4
    # tip_y = end_point[1] + np.sin(polar_angle) * np.sin(azimuthal_angle) * 4
    # tip_z = end_point[2] + np.cos(azimuthal_angle) * 4

    # the z_range over which the axis of the ellipse taper to 0
    # taper_range = (tip_z-end_point[2])/2.0
    # height at which the radius begins to taper (i.e. when the diameter increases at the tip)
    # taper_height = (tip_z-end_point[2])/2.0+end_point[2]

    # add point to the pairs of points (in a copied list so we do not alter the original points)
    segments = copy.deepcopy(points_start_end)
    # segments.append([start_point, [tip_x, tip_y, tip_z]])  # append in the beginning bc tip drawn first for breast

    # first, find angle of the ellipse on the z-plane as measured from the y-axis counterclockwise
    for pair in segments:
        start_point = pair[0]
        end_point = pair[1]
        ellipse_centers = get_points_between_catheter_points(start_point, end_point, slice_thickness)
        # angle measured between the catheter and the z-plane in deg
        angle = get_catheter_angle(start_point, end_point)

        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]

        # xy_angle is the polar angle on the z_plane/slice
        if dy == 0:
            if dx < 0:
                xy_angle = 90
            elif dx > 0:
                xy_angle = -90
            else:
                xy_angle = 0
        else:
            xy_angle = -np.rad2deg(np.arctan(dx / dy))

        for point in ellipse_centers:

            z_value = point[2]

            # the semi-minor and -major axes taper at the tip of the catheter
            # if z_value > taper_height:
            #     diameter = 3.0 * (1-(z_value-taper_height)/taper_range)
            # else:
            #     diameter = 3.0
            diameter = 3.0

            scaled_height = diameter / np.sin(np.deg2rad(angle))

            # x = 0
            # y = 0
            # i = 1
            max_iter = 16
            path_list = []

            for i in range(max_iter):
                x = diameter / 2 * np.cos(i * 2 * np.pi / max_iter)
                y = scaled_height / 2 * np.sin(i * 2 * np.pi / max_iter)

                x_rot = x * np.cos(np.deg2rad(xy_angle)) - y * np.sin(np.deg2rad(xy_angle))
                y_rot = x * np.sin(np.deg2rad(xy_angle)) + y * np.cos(np.deg2rad(xy_angle))

                x_fin = x_rot + point[0]
                y_fin = y_rot + point[1]
                path_list.append([x_fin, y_fin])
            path = Path(path_list)  # TODO: check type input

            if z_value not in path_dict:
                path_dict[z_value] = []
            path_dict[z_value].append(path)

    return path_dict


def get_z_values(start_point, end_point, slice_thickness):
    ellipse_centers = get_points_between_catheter_points(start_point, end_point, slice_thickness)
    z_values = []
    for point in ellipse_centers:
        z_values.append(point[2])
    return z_values

# # test data:
# start_point = [0, 0, 0]
# end_point = [1, 1, 1]
# # second_end = [-2, -2, 4]
# point_start_end = []
# point_start_end.append([start_point, end_point])
# # point_start_end.append([end_point, second_end])
# paths=get_ellipse_paths(point_start_end, 0.2)
# print(paths.keys())
# 
# fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
# angle = get_catheter_angle(start_point, end_point)
# for i in paths.keys():
#     path = paths[i][0]
#     patch = patches.PathPatch(path, facecolor='orange', lw=2)
#     ax.add_patch(patch)
# 
# ax.set_xlim(-5, 5)
# ax.set_ylim(-5, 5)
# plt.show()

