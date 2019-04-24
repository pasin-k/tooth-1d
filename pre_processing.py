# This file is to convert stl file and csv file into cross_section image
# After running this, use imgtotfrecord.py to turn images into tfrecord

# Import Libraries
import time
import csv
import os
from ImportData2D import get_label, get_file_name, save_plot
from stlSlicer import getSlicer, slicecoor, rotatestl
import numpy as np

v = '1.2.0'
# Initial version: Based on main.ipynb
# 1.1: Implemented ignore data that has problem
# 1.2: Now save image with their own name
print("pre_processing.py version: " + str(v))

degree = list([0, 45, 90, 135])
numdeg = len(degree)


# Get stl file and label, convert to stl_file
def get_cross_section(data_type, stat_type):
    # Get data and transformed to cross-section image
    name_dir, image_name = get_file_name(folder_name='../global_data/stl_data', file_name="PreparationScan.stl")
    label, label_name = get_label(data_type, stat_type, double_data=True, one_hotted=False, normalized=False)
    # Number of data should be the same as number of label
    if image_name != label_name:
        print(image_name)
        print(label_name)
        diff = list(set(image_name).symmetric_difference(set(label_name)))
        raise Exception("ERROR, image and label not similar: %d images, %d labels. Possible missing files: %s"
                        % (len(image_name), (len(label_name)), diff))

    augment_config = list([False, True])  # Original data once, Augmented once
    # Prepare two set of list, one for data, another for augmented data
    stl_points = list()
    stl_points_augmented = list()
    error_file_names = list()  # Names of file that cannot get cross-section image
    min_point = 1000
    max_point = 0
    for i in range(len(name_dir)):
        for augment in augment_config:
            points = getSlicer(name_dir[i], 0, degree, augment, axis=1)
            if points is None:  # If the output has error, remove label of that file
                error_file_names.append(image_name[i])
                index = label_name.index(image_name[i])
                label_name.pop(index)
                label.pop(index * 2)
                label.pop(index * 2)  # Do it again if we double the data
                break
            else:
                if len(points[0]) > max_point:
                    max_point = len(points[0])
                if len(points[0]) < min_point:
                    min_point = len(points[0])
                if augment:
                    stl_points_augmented.append(points)
                else:
                    stl_points.append(points)

    # The output is list(examples) of list(degrees) of numpy array (N*2 coordinates)
    print("Finished with %d examples, %d augmented examples" % (len(stl_points), len(stl_points_augmented)))
    print("Number of score received (bugged file removed): %d (Originally: %d)" % (len(label), len(label) / 2))

    print("Max point: %s, min  point: %s" % (max_point, min_point))
    # augment_num = int(len(label)/len(label_name))
    # label_name_aug = [val for val in label_name for _ in range(augment_num)]
    return stl_points, stl_points_augmented, label, label_name, error_file_names, degree


def save_image(stl_points, stl_points_augmented, label_name, error_file_names):
    # Directory to save image and errorfile
    image_dir = "./data/cross_section_test"
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
    if len(coor_list) != len(image_name):
        raise ValueError("save_plot: number of image(%s) is not equal to number of image_name(%s)"
                         % (len(coor_list), len(image_name)))
    out_directory = os.path.abspath(out_directory)
    if len(degree) != len(coor_list[0]):
        print("# of Degree expected: %d" % len(degree))
        print("# of Degree found: %d" % len(coor_list[0]))
        raise Exception('Number of degree specified is not equals to coordinate ')

    for i in range(len(coor_list)):
        for d in range(len(degree)):
            coor = coor_list[i][d]
            # Name with some additional data
            fullname = "%s_%s_%s_%d.npy" % (file_header_name, image_name[i], augment_number, degree[d])
            output_name = os.path.join(out_directory, fullname)
            np.save(output_name, coor)
            np.savetxt("%s_%s_%d.txt" % (file_header_name, image_name[i], degree[d]), coor)
    print("Finished saving coordinates: %d files with %d rotations at dir: %s" % (len(coor_list), len(degree), out_directory))


def save_stl_point(stl_points, stl_points_augmented, label_name, error_file_names):
    # Directory to save coordinates
    file_dir = "./data/coordinates"
    # This convert coordinates into vector between each coordinate
    stl_points = stl_point_to_movement(stl_points)
    stl_points_augmented = stl_point_to_movement(stl_points_augmented)
    # Save data as png image
    coor_name = "PreparationScan"
    save_coordinate(stl_points, file_dir, coor_name, label_name, 0, degree)
    print("Finished saving first set of data")
    # Save again for augmented data
    coor_name = "PreparationScan" + "_1"
    save_coordinate(stl_points_augmented, file_dir, coor_name, label_name, 0, degree)
    print("Finished saving second set of data")

    # Save names which has defect on it, use when convert to tfrecord
    with open(file_dir + '/error_file.txt', 'w') as filehandle:
        for listitem in error_file_names:
            filehandle.write('%s\n' % listitem)


if __name__ == '__main__':
    # Output 'points' as list[list[numpy]] (example_data, degrees, points)
    save_img = True
    save_coor = False

    # data_type, stat_type will not be used unless you want to look at lbl value
    points, points_aug, lbl, lbl_name, err_name, deg = get_cross_section(data_type="BL", stat_type="median")

    if save_img:
        save_image(points, points_aug, lbl_name, err_name)

    if save_coor:
        save_stl_point(points, points_aug, lbl_name, err_name)
    print("pre_processing.py: done")
