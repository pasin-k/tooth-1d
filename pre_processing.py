# This file is to convert stl file and csv file into cross_section image
# After running this, use imgtotfrecord.py to turn images into tfrecord

# Import Libraries
import time
import csv
import os
from open_save_file import get_label, get_file_name, save_plot
from stlSlicer import getSlicer, slicecoor
import numpy as np
from open_save_file import save_file

v = '1.2.0'
# Initial version: Based on main.ipynb
# 1.1: Implemented ignore data that has problem
# 1.2: Now save image with their own name
print("pre_processing.py version: " + str(v))

augment_config = [0, -1, -2, -3, 1, 2, 3]
degree = [0, 45, 90, 135]
numdeg = len(degree)


def get_coor_distance(stl_points, mode_remove):
    """
    Only can be used for 'normalize_coor' function
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
def fix_amount_of_point(stl_points, coor_amount):
    if type(stl_points).__module__ != np.__name__:
        raise ValueError("Input is not numpy array, currently %s" % type(stl_points))
    stl_points = stl_points.astype(float)
    # In case stl_point is too much, undersampling
    if np.shape(stl_points)[0] > coor_amount:
        # Remove index of the closest distance until satisfy
        for i in range(np.shape(stl_points)[0] - coor_amount):
            # Calculate distance between two points which are two space away (Actually focus on i+1, behind and forward)
            distance = get_coor_distance(stl_points, True)
            distance_index = np.argsort(distance)
            remove_index = distance_index[0]  # Choose min distance as the index to remove
            stl_points = np.delete(stl_points, remove_index + 1, axis=0)  # Remove from the points
    else:  # Oversampling
        for i in range(coor_amount - np.shape(stl_points)[0]):
            # Find distance between each point
            distance = get_coor_distance(stl_points, False)
            distance_index = np.argsort(distance)
            add_index = distance_index[-1]  # Choose max distance as the index to add coordinate
            new_point = (stl_points[add_index, :] + stl_points[add_index + 1, :]) / 2
            stl_points = np.insert(stl_points, add_index + 1, new_point, axis=0)
    # print(np.shape(stl_points))
    return stl_points


def get_cross_section(data_type, stat_type, augment_config=[0], folder_name='../global_data/stl_data',
                      file_name="PreparationScan.stl",
                      csv_dir='../global_data/Ground Truth Score_new.csv'):
    """
    Get coordinates of stl file and label
    :param data_type:       String, Type of label e.g. [Taper/Occ]
    :param stat_type:       String, Label measurement e.g [Average/Median]
    :param augment_config:  List of all augmentation angles
    :param folder_name:     String, folder directory of stl file
    :param csv_dir:         String, file directory of label (csv file)
    :param file_name:       String, filename can be None
    :return:
    stl_points_all          List of all point (np array)
    label_all               List of label
    label_name_all          List of label name (id)
    error_file_names_all    List of label name that has error
    """
    # Get data and transformed to cross-section image
    data_type = ["Occ_B", "Occ_F", "Occ_L", "Occ_Sum", "BL", "MD", "Taper_Sum", "Integrity", "Width", "Surface", "Sharpness"]
    stat_type = ["median"]

    name_dir, image_name = get_file_name(folder_name=folder_name, file_name=file_name)

    label = dict()
    label_header = ["name"]

    for d in data_type:
        for s in stat_type:
            l, label_name = get_label(d, s, double_data=False, one_hotted=False, normalized=False, file_dir=csv_dir)
            label[d + "_" + s] = l
            label_header.append(d + "_" + s)

    # Number of data should be the same as number of label
    if image_name != label_name:
        print(image_name)
        print(label_name)
        diff = list(set(image_name).symmetric_difference(set(label_name)))
        raise Exception("ERROR, image and label not similar: %d images, %d labels. Possible missing files: %s"
                        % (len(image_name), (len(label_name)), diff))

    # To verify number of coordinates
    min_point = 1000
    max_point = 0

    stl_points_all = []
    label_all = {k: [] for k in dict.fromkeys(label.keys())}
    label_name_all = []
    error_file_names_all = []
    for i in range(len(name_dir)):
        # Prepare two set of list, one for data, another for augmented data
        label_name_temp = []
        points_all = getSlicer(name_dir[i], 0, degree, augment=augment_config, axis=1)
        stl_points = []
        error_file_names = []  # Names of file that cannot get cross-section image

        for index, point in enumerate(points_all):
            augment_val = augment_config[index]
            if augment_val < 0:
                augment_val = "n" + str(abs(augment_val))
            else:
                augment_val = str(abs(augment_val))
            if point is None:  # If the output has error, remove label of that file
                error_file_names.append(image_name[i] + "_" + augment_val)
                # index = label_name_temp.index(image_name[i])
                # label_name_temp.pop(index)
                # label_temp.pop(index)
            else:
                stl_points.append(point)
                label_name_temp.append(image_name[i] + "_" + augment_val)
                if len(point[0]) > max_point:
                    max_point = len(point[0])
                if len(point[0]) < min_point:
                    min_point = len(point[0])

        # Add all label (augmented included)
        for key, value in label.items():
            label_all[key] += [value[i] for _ in range(len(stl_points))]
        # label_all.append([label[i]] * len(stl_points))
        # stl_points_all.append(stl_points)
        stl_points_all += stl_points  # Add these points to the big one
        label_name_all += label_name_temp  # Also add label name to the big one
        error_file_names_all += error_file_names  # Same as error file
    label_all["name"] = label_name_all

    # The output is list(examples) of list(degrees) of numpy array (N*2 coordinates)
    for label_name in label_name_all:
        print("Finished with %d examples" % (len(label_name)))

    print("Max amount of coordinates: %s, min  coordinates: %s" % (max_point, min_point))
    return stl_points_all, label_all, label_name_all, error_file_names_all, label_header


def save_image(stl_points, label_name, error_file_names, image_dir="./data/cross_section"):
    # Save data as png image
    png_name = "PreparationScan"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    open(image_dir + '/error_file.txt', 'w').close()
    open(image_dir + '/config.txt', 'w').close()
    for j in range(len(label_name)):
        save_plot(stl_points[j], image_dir, png_name, label_name[j], degree)
        # Save names which has defect on it, use when convert to tfrecord
        if j % 50 == 0:
            print("Saved %s out of %s" % (j, len(label_name)))
    with open(image_dir + '/error_file.txt', 'a') as filehandle:
        for listitem in error_file_names:
            filehandle.write('%s\n' % listitem)
    print("Finished saving data")
    with open(image_dir + '/config.txt', 'a') as filehandle:
        filehandle.write('%s\n' % len(degree))
        filehandle.write('%s\n' % len(augment_config))


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
def save_coordinate(coor_list, out_directory, file_header_name, image_name, degree):
    # Check if size coordinate, image name has same length
    if len(coor_list) != len(degree):
        raise ValueError("Number of degree is not equal to %s, found %s", (len(degree), len(coor_list)))
    # if len(coor_list) != len(image_name):
    #     raise ValueError("save_plot: number of image(%s) is not equal to number of image_name(%s)"
    #                      % (len(coor_list), len(image_name)))
    out_directory = os.path.abspath(out_directory)
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    # if len(degree) != len(coor_list[0]):
    #     print("# of Degree expected: %d" % len(degree))
    #     print("# of Degree found: %d" % len(coor_list[0]))
    #     raise Exception('Number of degree specified is not equals to coordinate ')

    # for corr_index in range(len(coor_list)):
    for deg_index in range(len(degree)):
        coor = coor_list[deg_index]
        # coor = coor_list[corr_index][deg_index]
        # Name with some additional data
        fullname = "%s_%s_%d.npy" % (file_header_name, image_name, degree[deg_index])
        # fullname = "%s_%s_%s_%d.npy" % (file_header_name, image_name, augment_number, degree[deg_index])
        output_name = os.path.join(out_directory, fullname)
        # np.savetxt(output_name, coor, delimiter=',')
        np.save(output_name, coor)
        # np.savetxt(output_name.replace(".npy",".txt"), coor)
    # print("Finished saving coordinates: %d files with %d rotations at dir: %s" % (
    #     len(coor_list), len(degree), out_directory))


def save_stl_point(stl_points, label_name, error_file_names, file_dir="./data/coordinates"):
    # Save data as png image
    coor_name = "PreparationScan"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    open(file_dir + '/error_file.txt', 'w').close()
    open(file_dir + '/config.txt', 'w').close()
    # This convert coordinates into vector between each coordinate
    stl_points = stl_point_to_movement(stl_points)
    for j in range(len(label_name)):
        save_coordinate(stl_points[j], file_dir, coor_name, label_name[j], degree)

    # Save names which has defect on it, use when convert to tfrecord
    with open(file_dir + '/error_file.txt', 'a') as filehandle:
        for listitem in error_file_names:
            filehandle.write('%s\n' % listitem)
    print("Finished saving data")
    with open(file_dir + '/config.txt', 'a') as filehandle:
        filehandle.write('%s\n' % len(degree))
        filehandle.write('%s\n' % len(augment_config))


if __name__ == '__main__':
    # Output 'points' as list[list[numpy]] (example_data, degrees, points)
    save_img = True
    save_coor = True
    is_fix_amount = True
    fix_amount = 300  # After get the movement, it will be reduced to 300

    # data_type, stat_type will not be used unless you want to look at lbl value
    points_all, lbl_all, lbl_name_all, err_name_all, header = get_cross_section(data_type="BL",
                                                                                stat_type="median",
                                                                                augment_config=augment_config, )
    # folder_name="/home/pasin/Documents/Link_to_Tooth/Tooth/Model/global_data/stl_data_debug",
    # csv_dir="/home/pasin/Documents/Link_to_Tooth/Tooth/Model/global_data/Ground Truth Score_debug.csv")
    # folder_name='../global_data/stl_data_debug',
    # csv_dir='../global_data/Ground Truth Score_debug.csv')
    if is_fix_amount:
        fix_amount = fix_amount + 1  # Compensate for the missing data
        print("Adjusting number of coordinates... Takes a long time")
        for i in range(len(points_all)):
            for d_index in range(len(degree)):
                points_all[i][d_index] = fix_amount_of_point(points_all[i][d_index], fix_amount)
            print("Done %s out of %s" % (i + 1, len(points_all)))

    if save_img:
        print("Start saving images...")
        image_dir = "./data/cross_section_newer"
        save_image(points_all, lbl_name_all, err_name_all, image_dir=image_dir)
        save_file(os.path.join(image_dir, "score.csv"), lbl_all, data_format="dict_list", field_name=header)

    if save_coor:
        print("Start saving coordinates...")
        file_dir = "./data/coordinate_newer"
        save_stl_point(points_all, lbl_name_all, err_name_all, file_dir=file_dir)
        save_file(os.path.join(file_dir, "score.csv"), lbl_all, data_format="dict_list", field_name=header)
    print("pre_processing.py: done")
