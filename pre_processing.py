# This file is to convert stl file and csv file into cross_section image
# After running this, use imgtotfrecord.py to turn images into tfrecord

# Import Libraries
import time
import csv
import os
from open_save_file import get_label, get_file_name, save_plot
from stlSlicer import getSlicer, slicecoor
import numpy as np

v = '1.2.0'
# Initial version: Based on main.ipynb
# 1.1: Implemented ignore data that has problem
# 1.2: Now save image with their own name
print("pre_processing.py version: " + str(v))

degree = [0, 45, 90, 135]
numdeg = len(degree)


# Only can be used for 'normalize_coor' function
def get_coor_distance(stl_points, mode_remove):
    distance = []
    if mode_remove:
        for i in range((np.shape(stl_points)[0]) - 2):
            distance.append(np.linalg.norm(stl_points[i, :] - stl_points[i + 1, :]) +
                            np.linalg.norm(stl_points[i + 1, :] - stl_points[i + 2, :]))
    else:
        for i in range((np.shape(stl_points)[0]) - 1):
            distance.append(np.linalg.norm(stl_points[i, :] - stl_points[i + 1, :]))
    return distance


# stl_points are expected to be numpy arrays of coordinate of a single image
def fix_amount_of_point(stl_points, coor_amount):
    if type(stl_points).__module__ != np.__name__:
        raise ValueError("Input is not numpy array, currently %s" % type(stl_points))
    stl_points = stl_points.astype(float)
    # In case stl_point is too much, need to remove
    if np.shape(stl_points)[0] > coor_amount:
        # Remove index of the closest distance until satisfy
        for i in range(np.shape(stl_points)[0] - coor_amount):
            # Calculate distance between two points which are two space away (Actually focus on i+1, behind and forward)
            distance = get_coor_distance(stl_points, True)
            distance_index = np.argsort(distance)
            remove_index = distance_index[0]  # Choose min distance as the index to remove
            stl_points = np.delete(stl_points, remove_index + 1, axis=0)  # Remove from the points
    else:
        for i in range(coor_amount - np.shape(stl_points)[0]):
            # Find distance between each point
            distance = get_coor_distance(stl_points, False)
            distance_index = np.argsort(distance)
            add_index = distance_index[-1]  # Choose max distance as the index to add coordinate
            new_point = (stl_points[add_index, :] + stl_points[add_index + 1, :]) / 2
            stl_points = np.insert(stl_points, add_index + 1, new_point, axis=0)
    return stl_points


def get_cross_section(data_type, stat_type, folder_name='../global_data/stl_data', file_name="PreparationScan.stl",
                      csv_dir = '../global_data/Ground Truth Score_new.csv'):
    """
    Get coordinates of stl file and label
    :param data_type:   String, Type of label e.g. [Taper/Occ]
    :param stat_type:   String, Label measurement e.g [Average/Median]
    :param folder_name: String, folder directory of stl file
    :param csv_dir:     String, file directory of label (csv file)
    :param file_name:   String, filename can be None
    :return:
    stl_points_all          List of all point (np array)
    label_all               List of label
    label_name_all          List of label name (id)
    error_file_names_all    List of label name that has error
    degree                  List of degree on each example
    augment_config          List of degree of augmentations
    """
    # Get data and transformed to cross-section image
    name_dir, image_name = get_file_name(folder_name=folder_name, file_name=file_name)
    label, label_name = get_label(data_type, stat_type, double_data=False, one_hotted=False, normalized=False, file_dir=csv_dir)
    # Number of data should be the same as number of label
    if image_name != label_name:
        print(image_name)
        print(label_name)
        diff = list(set(image_name).symmetric_difference(set(label_name)))
        raise Exception("ERROR, image and label not similar: %d images, %d labels. Possible missing files: %s"
                        % (len(image_name), (len(label_name)), diff))

    augment_config = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    # To verify number of coordinates
    min_point = 1000
    max_point = 0

    stl_points_all = []
    label_all = []
    label_name_all = []
    error_file_names_all = []
    for i in range(len(name_dir)):
        # Prepare two set of list, one for data, another for augmented data
        label_temp = label
        label_name_temp = label_name
        points_all = getSlicer(name_dir[i], 0, degree, augment=augment_config, axis=1)
        stl_points = []
        error_file_names = []  # Names of file that cannot get cross-section image

        for point in points_all:

            if point is None:  # If the output has error, remove label of that file
                error_file_names.append(image_name[i])
                index = label_name_temp.index(image_name[i])
                label_name_temp.pop(index)
                label_temp.pop(index)
                break
            else:
                stl_points.append(point)
                if len(point[0]) > max_point:
                    max_point = len(point[0])
                if len(point[0]) < min_point:
                    min_point = len(point[0])

        stl_points_all.append(stl_points)
        label_all.append(label_temp)
        label_name_all.append(label_name_temp)
        error_file_names_all.append(error_file_names)

    # The output is list(examples) of list(degrees) of numpy array (N*2 coordinates)
    for label_name in label_name_all:
        print("Finished with %d examples" % (len(label_name)))

    print("Max amount of coordinates: %s, min  coordinates: %s" % (max_point, min_point))
    return stl_points_all, label_all, label_name_all, error_file_names_all, degree, augment_config


def save_image(stl_points, stl_points_augmented, label_name, error_file_names, image_dir="./data/cross_section"):
    # Save data as png image
    png_name = "PreparationScan"
    save_plot(stl_points, image_dir, png_name, label_name, 0, degree)
    print("Finished saving first set of data")
    # Save again for augmented data
    save_plot(stl_points_augmented, image_dir, png_name, label_name, 1, degree)
    print("Finished saving second set of data")

    # Save names which has defect on it, use when convert to tfrecord
    with open(image_dir + '/error_file.txt', 'w') as filehandle:
        for listitem in error_file_names:
            filehandle.write('%s\n' % listitem)


def stl_point_to_movement(stl_points):  # stl_points is list of all file (all examples)
    new_stl_points = []
    for stl_point_sample in stl_points:  # stl_point_sample is one example of stl_points
        new_points_sample = []
        for stl_point_image in stl_point_sample:  # stl_point_image is one degree of cross-section
            difference = stl_point_image[1:, :] - stl_point_image[0:-1, :]  # Find difference between each position
            new_points_sample.append(difference)
        new_stl_points.append(new_points_sample)
    return new_stl_points


# Save coordinate as .npy file
def save_coordinate(coor_list, out_directory, file_header_name, image_name, augment_number, degree):
    # Check if size coordinate, image name has same length
    if len(coor_list) != len(image_name):
        raise ValueError("save_plot: number of image(%s) is not equal to number of image_name(%s)"
                         % (len(coor_list), len(image_name)))
    out_directory = os.path.abspath(out_directory)
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    if len(degree) != len(coor_list[0]):
        print("# of Degree expected: %d" % len(degree))
        print("# of Degree found: %d" % len(coor_list[0]))
        raise Exception('Number of degree specified is not equals to coordinate ')

    for corr_index in range(len(coor_list)):
        for deg_index in range(len(degree)):
            coor = coor_list[corr_index][deg_index]
            print(np.shape(coor))
            # Name with some additional data
            fullname = "%s_%s_%s_%d.npy" % (file_header_name, image_name[corr_index], augment_number, degree[deg_index])
            output_name = os.path.join(out_directory, fullname)
            np.savetxt(output_name, coor, delimiter=',')
            # np.save(output_name, coor)
            # np.savetxt(output_name.replace(".npy",".txt"), coor)
    print("Finished saving coordinates: %d files with %d rotations at dir: %s" % (
        len(coor_list), len(degree), out_directory))


def save_stl_point(stl_points, stl_points_augmented, label_name, error_file_names, file_dir="./data/coordinates"):
    # This convert coordinates into vector between each coordinate
    stl_points = stl_point_to_movement(stl_points)
    stl_points_augmented = stl_point_to_movement(stl_points_augmented)
    # Save data as png image
    coor_name = "PreparationScan"
    save_coordinate(stl_points, file_dir, coor_name, label_name, 0, degree)
    print("Finished saving first set of data")
    # Save again for augmented data
    save_coordinate(stl_points_augmented, file_dir, coor_name, label_name, 1, degree)
    print("Finished saving second set of data")

    # Save names which has defect on it, use when convert to tfrecord
    with open(file_dir + '/error_file.txt', 'w') as filehandle:
        for listitem in error_file_names:
            filehandle.write('%s\n' % listitem)


if __name__ == '__main__':
    # Output 'points' as list[list[numpy]] (example_data, degrees, points)
    save_img = True
    save_coor = False
    is_fix_amount = True
    fix_amount = 100  # After get the movement, it will be reduced to 300

    # data_type, stat_type will not be used unless you want to look at lbl value
    points_all, lbl_all, lbl_name_all, err_name_all, deg, aug = get_cross_section(data_type="BL", stat_type="median")
    if is_fix_amount:
        for i in len(lbl_all):
            points = points_all[i]
            print("Adjusting number of coordinates...")
            for p_index in range(len(points)):
                for d_index in range(len(degree)):
                    points[p_index][d_index] = fix_amount_of_point(points[p_index][d_index], fix_amount)
            points_all[i] = points

    if save_img:
        print("Start saving images...")
        save_image(points_all, lbl_name_all, err_name_all, image_dir="./data/cross_section_100")

    if save_coor:
        fix_amount = fix_amount + 1  # Compensate for the missing data
        print("Start saving coordinates...")
        save_stl_point(points_all, lbl_name_all, err_name_all, file_dir="./data/coordinate_100")
    print("pre_processing.py: done")
