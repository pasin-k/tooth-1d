import os
import numpy as np
from stl import mesh
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from stlSlicer import getSlicer
from open_save_file import get_file_name

# TODO; Check if all file is created

augment_config = [0, 5, 10, 15, 20, 25, 30, 35, 40]
degree = [0, 45, 90, 135]


def find_slope(point1, point2):  # function to find top line
    if point1[0] == point2[0]:
        return 0
    else:
        return (point1[1] - point2[1]) / (point1[0] - point2[0])


def get_segment(points, mode=None, margin=0, file_name=None):
    """
    Get a segments of cross section
    :param points: ndarray of N*2
    :param mode: String, [left, right, top, debug] -> select which part to return
    :param margin: Float, amount of margin from minimum line
    :return: ndarray of segmented image or saved image (Not sure)
    """
    # Turn into numpy
    points = np.asarray(points)

    # Find max and min x to find sharp point of tooth
    x_min_index = np.argmin(points, axis=0)[0]
    x_max_index = np.argmax(points, axis=0)[0]

    left_point = points[x_min_index, :]
    right_point = points[x_max_index, :]

    # Initial slope, (+20 points to avoid corner of the curve which could provide a unusually large slope)
    top_left_index = x_min_index + 20
    top_right_index = x_max_index - 20
    left_slope = find_slope(left_point, points[top_left_index])
    right_slope = find_slope(right_point, points[top_right_index])
    slope_track = []
    # Find largest slope point, choose as top_left and top_right
    for i in range(top_left_index, top_right_index):
        # For left side, find the steepest slope
        new_left_slope = find_slope(points[x_min_index, :], points[i, :])
        if new_left_slope > left_slope and new_left_slope > 0:
            left_slope = new_left_slope
            top_left_index = i
        # For right side, find the steepest negative slope
        new_right_slope = find_slope(points[x_max_index, :], points[i, :])
        slope_track.append([new_right_slope, points[i, :]])
        if new_right_slope < right_slope and new_right_slope < 0:
            right_slope = new_right_slope
            top_right_index = i

    # Current coordinates: x_min_index, top_left_index, x_max_index, top_right_index

    #     # saves the points from condition above
    #     FigureCoords = np.asarray(B)
    #     np.save(file=save_path + '1dfigs/' + directory[1] + '_' + str(cross_sections[ind]), arr=FigureCoords)

    if mode == "left":
        # If there is margin, choose points slightly below x_min_index
        if margin > 0:
            segmented_points = points[0:top_left_index + 1, :]
            segmented_points = segmented_points[segmented_points[:, 1] > points[x_min_index, 1] - margin]
        else:
            segmented_points = points[x_min_index:top_left_index + 1, :]
        x = segmented_points[:, 0]
        y = segmented_points[:, 1]
    elif mode == "right":
        if margin > 0:
            segmented_points = points[top_right_index:, :]
            segmented_points = segmented_points[segmented_points[:, 1] > points[x_max_index, 1] - margin]
        else:
            segmented_points = points[top_right_index:x_max_index + 1, :]
        x = segmented_points[:, 0]
        y = segmented_points[:, 1]
    elif mode == "top":
        segmented_points = points[top_left_index:top_right_index + 1, :]
        x = segmented_points[:, 0]
        y = segmented_points[:, 1]
    else:
        # Display entire tooth
        mode = None
        segmented_points = points
        x = points[:, 0]
        y = points[:, 1]

    # print(np.shape(x))

    # Plotting
    dpi = 100
    img_size = 800
    fig = plt.figure(figsize=(img_size / dpi, img_size / dpi), dpi=dpi)
    ax = fig.gca()
    min_x, max_x, min_y, max_y = -5, 5, -6, 6
    if mode is None:
        ax.axis([min_x, max_x, min_y, max_y])
        ax.set_autoscale_on(False)  # allows us to define scale
    ax.plot(x, y, linewidth=1.0)

    if file_name is not None:
        # plot lines for viewing
        x1 = range(-5, 5)
        yleft = np.full((10,), points[top_left_index, :][1])
        yright = np.full((10,), points[top_right_index, :][1])
        ax.plot(x1, yleft, '-c')
        ax.plot(x1, yright, '-r')
        if (mode is None) or (mode == "left"):
            ybottom_left = np.full((10,), points[x_min_index, :][1] - margin)
            ax.plot(x1, ybottom_left, '-py')
            ybottom_left = np.full((10,), points[x_min_index, :][1])
            ax.plot(x1, ybottom_left, '-y')
        if (mode is None) or (mode == "right"):
            ybottom_right = np.full((10,), points[x_max_index, :][1] - margin)
            ax.plot(x1, ybottom_right, '-pb')
            ybottom_right = np.full((10,), points[x_max_index, :][1])
            ax.plot(x1, ybottom_right, '-b')
        fig.savefig(file_name, bbox_inches='tight')
        plt.close('all')
    return segmented_points


def get_segment_multiple(name, margin=0,
                         base_dir="/home/pasin/Documents/Google_Drive/Aa_TIT_LAB_Comp/Library/Tooth/Tooth/Model/my2DCNN/data/segment_2"):
    """
    Get a segments of cross section from multiple files
    :param points: ndarray of N*2
    :param mode: String, [left, right, top, debug] -> select which part to return
    :param margin: Float, amount of margin from minimum line
    :return: ndarray of segmented image or saved image (Not sure)
    """
    name_dir, image_name = get_file_name(folder_name=name, file_name="PreparationScan.stl")
    name_dir = name_dir[160:]
    image_name = image_name[160:]
    cnt = 0
    for n_name, im_name in zip(name_dir, image_name):
        cnt += 1
        points_all = getSlicer(n_name, 0, degree, augment=augment_config, axis=1)
        stl_points = []
        error_file_names = []  # Names of file that cannot get cross-section image

        for index, point in enumerate(points_all):  # Enumerate over all augmentation points
            if point is None:
                error_file_names.append("PreparationScan_%s_%d" % (im_name, augment_config[index]))
            else:
                for d in range(len(degree)):

                    file_name = "PreparationScan_%s_%s_%d.png" % (im_name, augment_config[index], degree[d])
                    file_name_point = "PreparationScan_%s_%s_%d.npy" % (im_name, augment_config[index], degree[d])
                    # Turn into numpy
                    points = np.asarray(point[d])

                    # Find max and min x to find sharp point of tooth
                    x_min_index = np.argmin(points, axis=0)[0]
                    x_max_index = np.argmax(points, axis=0)[0]

                    left_point = points[x_min_index, :]
                    right_point = points[x_max_index, :]

                    # Initial slope, (+20 points to avoid corner of the curve which could provide a unusually large slope)
                    top_left_index = x_min_index + 20
                    top_right_index = x_max_index - 20
                    left_slope = find_slope(left_point, points[top_left_index])
                    right_slope = find_slope(right_point, points[top_right_index])
                    slope_track = []
                    # Find largest slope point, choose as top_left and top_right
                    for i in range(top_left_index, top_right_index):
                        # For left side, find the steepest slope
                        new_left_slope = find_slope(points[x_min_index, :], points[i, :])
                        if new_left_slope > left_slope and new_left_slope > 0:
                            left_slope = new_left_slope
                            top_left_index = i
                        # For right side, find the steepest negative slope
                        new_right_slope = find_slope(points[x_max_index, :], points[i, :])
                        slope_track.append([new_right_slope, points[i, :]])
                        if new_right_slope < right_slope and new_right_slope < 0:
                            right_slope = new_right_slope
                            top_right_index = i

                    # Current coordinates: x_min_index, top_left_index, x_max_index, top_right_index

                    #     # saves the points from condition above
                    #     FigureCoords = np.asarray(B)
                    #     np.save(file=save_path + '1dfigs/' + directory[1] + '_' + str(cross_sections[ind]), arr=FigureCoords)

                    # Plotting
                    dpi = 100
                    img_size = 800
                    fig = plt.figure(figsize=(img_size / dpi, img_size / dpi), dpi=dpi)
                    ax = fig.gca()
                    min_x, max_x, min_y, max_y = -5, 5, -6, 6
                    ax.axis([min_x, max_x, min_y, max_y])
                    ax.set_autoscale_on(False)  # allows us to define scale
                    # plot lines for viewing
                    x1 = range(-5, 5)
                    yleft = np.full((10,), points[top_left_index, :][1])
                    yright = np.full((10,), points[top_right_index, :][1])
                    ybottom_left_margin = np.full((10,), points[x_min_index, :][1] - margin)
                    ybottom_right_margin = np.full((10,), points[x_max_index, :][1] - margin)
                    ybottom_left = np.full((10,), points[x_min_index, :][1])
                    ybottom_right = np.full((10,), points[x_max_index, :][1])

                    # Display entire tooth
                    segmented_points = points
                    x = points[:, 0]
                    y = points[:, 1]

                    ax.plot(x, y, linewidth=1.0)
                    ax.plot(x1, yleft, '-c')
                    ax.plot(x1, yright, '-r')
                    ax.plot(x1, ybottom_left_margin, '-py')
                    ax.plot(x1, ybottom_left, '-y')
                    ax.plot(x1, ybottom_right_margin, '-pb')
                    ax.plot(x1, ybottom_right, '-b')

                    fig.savefig(base_dir + "/full/" + file_name, bbox_inches='tight')
                    plt.close()

                    ax.relim()
                    ax.autoscale()

                    # Left
                    # If there is margin, choose points slightly below x_min_index
                    if margin > 0:
                        segmented_points = points[0:top_left_index + 1, :]
                        segmented_points = segmented_points[segmented_points[:, 1] > points[x_min_index, 1] - margin]
                    else:
                        segmented_points = points[x_min_index:top_left_index + 1, :]
                    x = segmented_points[:, 0]
                    y = segmented_points[:, 1]

                    ax.plot(x, y, linewidth=1.0)
                    ax.plot(x1, yleft, '-c')
                    ax.plot(x1, yright, '-r')
                    ax.plot(x1, ybottom_left_margin, '-py')
                    ax.plot(x1, ybottom_left, '-y')

                    fig.savefig(base_dir + "/left/" + file_name, bbox_inches='tight')
                    np.save(base_dir + "/left_point/" + file_name_point, segmented_points)
                    plt.close()

                    # Right
                    if margin > 0:
                        segmented_points = points[top_right_index:, :]
                        segmented_points = segmented_points[segmented_points[:, 1] > points[x_max_index, 1] - margin]
                    else:
                        segmented_points = points[top_right_index:x_max_index + 1, :]
                    x = segmented_points[:, 0]
                    y = segmented_points[:, 1]

                    ax.plot(x, y, linewidth=1.0)
                    ax.plot(x1, yleft, '-c')
                    ax.plot(x1, yright, '-r')
                    ax.plot(x1, ybottom_right_margin, '-pb')
                    ax.plot(x1, ybottom_right, '-b')

                    fig.savefig(base_dir + "/right/" + file_name, bbox_inches='tight')
                    np.save(base_dir + "/right_point/" + file_name_point, segmented_points)
                    plt.close()

                    # Top
                    segmented_points = points[top_left_index:top_right_index + 1, :]
                    x = segmented_points[:, 0]
                    y = segmented_points[:, 1]

                    ax.plot(x, y, linewidth=1.0)
                    ax.plot(x1, yleft, '-c')
                    ax.plot(x1, yright, '-r')
                    ax.plot(x1, ybottom_left_margin, '-py')
                    ax.plot(x1, ybottom_left, '-y')

                    fig.savefig(base_dir + "/top/" + file_name, bbox_inches='tight')
                    np.save(base_dir + "/top_point/" + file_name_point, segmented_points)
                    plt.close()
            # plt.close()
        if cnt % 10 == 0:
            print("Progress: %s, current image: %s" % (cnt, im_name))


if __name__ == '__main__':
    #NOTE: Run on jupyter notebook
    get_segment_multiple(name='../global_data/stl_data', margin=0,
                         base_dir="/home/pasin/Documents/Google_Drive/Aa_TIT_LAB_Comp/Library/Tooth/Tooth/Model/my2DCNN/data/segment_2")
