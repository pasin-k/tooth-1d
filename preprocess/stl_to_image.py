"""
This file is to convert stl file and csv file into cross_section image
After running this, use image_to_tfrecord.py to turn images into tfrecord
"""

# Import Libraries
import os
from utils.open_save_file import save_plot, save_coordinate, save_file, get_cross_section
import numpy as np

augment_config = [0, -1, -2, -3, 1, 2, 3, 180, 179, 178, 177, 181, 182, 183]
degree = [0, 45, 90, 135]
numdeg = len(degree)


def get_coor_distance(stl_points, mode_remove):
    """
    Only can be used for 'normalize_coor' function, calculate distance between points
    :param stl_points:  List of points
    :param mode_remove: True: Undersampling, False: Oversampling
    :return:            List of distance
    """
    distance = []
    if mode_remove:  # Find distance between points that point and the next two point
        for i in range((np.shape(stl_points)[0]) - 2):
            distance.append(np.linalg.norm(stl_points[i, :] - stl_points[i + 1, :]) +
                            np.linalg.norm(stl_points[i + 1, :] - stl_points[i + 2, :]))
    else:
        for i in range((np.shape(stl_points)[0]) - 1):  # Find distance between that point and next point
            distance.append(np.linalg.norm(stl_points[i, :] - stl_points[i + 1, :]))
    return distance


# stl_points are expected to be numpy arrays of coordinate of a single image
def point_sampling(stl_points, coor_amount):
    """
    Sampling points from abitrary value into a fixed amount
    :param stl_points: ndarray (N,2)
    :param coor_amount: int, amount of point
    :return: ndarray
    """
    # Check if stl_points is numpy
    if type(stl_points).__module__ != np.__name__:
        raise ValueError("Input is not numpy array, currently %s" % type(stl_points))
    stl_points = stl_points.astype(float)

    # Determine if undersampling or oversampling
    if np.shape(stl_points)[0] > coor_amount:  # Undersampling
        # Remove index of the closest distance until satisfy
        for j in range(np.shape(stl_points)[0] - coor_amount):
            # Calculate distance between two points which are two space away (Actually focus on i+1, behind and forward)
            distance = get_coor_distance(stl_points, True)
            distance_index = np.argsort(distance)
            remove_index = distance_index[0]  # Choose min distance as the index to remove
            stl_points = np.delete(stl_points, remove_index + 1, axis=0)  # Remove from the points
    else:  # Oversampling
        # Add point between two points with most distance
        for j in range(coor_amount - np.shape(stl_points)[0]):
            # Find distance between each point
            distance = get_coor_distance(stl_points, False)
            distance_index = np.argsort(distance)
            add_index = distance_index[-1]  # Choose max distance as the index to add coordinate
            new_point = (stl_points[add_index, :] + stl_points[add_index + 1, :]) / 2
            stl_points = np.insert(stl_points, add_index + 1, new_point, axis=0)
    return stl_points


def save_image(stl_points, label_name, error_file_names, out_directory="./data/cross_section"):
    """
    Save all cross-section into an png image
    :param stl_points: List of cross-section images from stlslicer, can be used directly from 'get_cross_section' function
    :param label_name: List of score associate to each image
    :param error_file_names: List of file with error
    :param out_directory: Directory to save images
    :return: File saved: Images, error_file.txt, config.txt
    """
    # Save data as png image
    png_name = "PreparationScan"
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    # Overwrite if same file exist
    open(out_directory + '/error_file.txt', 'w').close()
    open(out_directory + '/config.txt', 'w').close()
    for j in range(len(label_name)):
        save_plot(stl_points[j], out_directory, "%s_%s" % (png_name, label_name[j]), degree)
        if j % 50 == 0:
            print("Saved %s out of %s" % (j, len(label_name)))
    # Save names which has defect on it, use when convert to tfrecord
    with open(out_directory + '/error_file.txt', 'a') as filehandle:
        for listitem in error_file_names:
            filehandle.write('%s\n' % listitem)
    with open(out_directory + '/config.txt', 'a') as filehandle:
        filehandle.write('%s\n' % len(degree))
        filehandle.write('%s\n' % len(augment_config))
    print("Finished saving data")


def stl_point_to_movement(stl_points):  # stl_points is list of all file (all examples)
    """
    Get difference between each coordinates instead
    :param stl_points: list of points
    :return:
    """
    new_stl_points = []
    for stl_point_sample in stl_points:  # stl_point_sample is one example of stl_points
        new_points_sample = []
        for stl_point_image in stl_point_sample:  # stl_point_image is one degree of cross-section
            difference = stl_point_image[1:, :] - stl_point_image[0:-1, :]  # Find difference between each position
            new_points_sample.append(difference)
        new_stl_points.append(new_points_sample)
    return new_stl_points


def save_stl_point(stl_points, label_name, error_file_names, out_directory="./data/coordinates"):
    """
    Save all cross-section into an .npy
    :param stl_points: List of cross-section images from stlslicer, can be used directly from 'get_cross_section' function
    :param label_name: List of score associate to each image
    :param error_file_names: List of file with error
    :param out_directory: Directory to save images
    :return: File saved: npy, error_file.txt, config.txt
        """
    # Save data as .npy file
    coor_name = "PreparationScan"
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    # Overwrite if same file exist
    open(out_directory + '/error_file.txt', 'w').close()
    open(out_directory + '/config.txt', 'w').close()

    # This convert coordinates into vector between each coordinate
    stl_points = stl_point_to_movement(stl_points)
    for j in range(len(label_name)):
        save_coordinate(stl_points[j], out_directory, "%s_%s" % (coor_name, label_name[j]), degree)

    # Save names which has defect on it, use when convert to tfrecord
    with open(out_directory + '/error_file.txt', 'a') as filehandle:
        for listitem in error_file_names:
            filehandle.write('%s\n' % listitem)
    print("Finished saving data")
    # Save some useful data
    with open(out_directory + '/config.txt', 'a') as filehandle:
        filehandle.write('%s\n' % len(degree))
        filehandle.write('%s\n' % len(augment_config))


# Fetch stl file and save as either image or .npy file of coordinates
if __name__ == '__main__':
    # Output 'points' as list[list[numpy]] (example_data, degrees, points)
    save_img = True
    save_coor = True
    is_fix_amount = True
    fix_amount = 300  # Sampling coordinates to specified amount

    # data_type, stat_type will not be used unless you want to look at lbl value
    points_all, lbl_all, lbl_name_all, err_name_all, header = get_cross_section(degree=degree,
                                                                                augment_config=augment_config,
                                                                                # folder_name='../../global_data/stl_data_debug',
                                                                                # csv_dir='../../global_data/Ground Truth Score_debug.csv',
                                                                                )
    if is_fix_amount:
        fix_amount = fix_amount + 1  # Compensate for the missing data
        print("Adjusting number of coordinates... Takes a long time")
        for i in range(len(points_all)):
            for d_index in range(len(degree)):
                points_all[i][d_index] = point_sampling(points_all[i][d_index], fix_amount)
            if i % 50 == 0:
                print("Done %s out of %s" % (i + 1, len(points_all)))

    if save_img:
        print("Start saving images...")
        image_dir = "../data/cross_section_14augment"
        save_image(points_all, lbl_name_all, err_name_all, out_directory=image_dir)
        save_file(os.path.join(image_dir, "score.csv"), lbl_all, data_format="dict_list", field_name=header)

    if save_coor:
        print("Start saving coordinates...")
        file_dir = "../data/coordinate_14augment"
        save_stl_point(points_all, lbl_name_all, err_name_all, out_directory=file_dir)
        save_file(os.path.join(file_dir, "score.csv"), lbl_all, data_format="dict_list", field_name=header)
    print("stl_to_image.py: done")
