# This file is to convert stl file and csv file into cross_section image
# After running this, use imgtotfrecord.py to turn images into tfrecord

# Import Libraries
import time
from ImportData2D import get_label, get_file_name, save_plot
from stlSlicer import getSlicer, slicecoor, rotatestl
import numpy as np

v = '1.2.0'
# Initial version: Based on main.ipynb
# 1.1: Implemented ignore data that has problem
# 1.2: Now save image with their own name
print("pre_processing.py version: " + str(v))

degree = list([0, 45, 90, 135])

def main():
    # Get data and transformed to cross-section image
    name_dir, image_name = get_file_name(folder_name='../global_data/', file_name="PreparationScan.stl")
    label, label_name = get_label("Taper_Sum", "median", double_data=True, one_hotted=False, normalized=False)
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
    print("Number of score received: %d (Originally: %d" % (len(label), len(label) / 2))

    print("Max point: %s, min  point: %s" % (max_point, min_point))
    # augment_num = int(len(label)/len(label_name))
    # label_name_aug = [val for val in label_name for _ in range(augment_num)]
    return stl_points, stl_points_augmented, label_name, error_file_names, degree


def save_image(stl_points, stl_points_augmented, label_name, error_file_names):
    # Save data as png image
    png_name = "PreparationScan" + "_0"
    save_plot(stl_points, "./data/cross_section", png_name, label_name, degree)
    print("Finished saving first set of data")
    # Save again for augmented data
    png_name = "PreparationScan" + "_1"
    save_plot(stl_points_augmented, "./data/cross_section", png_name, label_name, degree)
    print("Finished saving second set of data")

    # Save names which has defect on it, use when convert to tfrecord
    with open('./data/cross_section/error_file.txt', 'w') as filehandle:
        for listitem in error_file_names:
            filehandle.write('%s\n' % listitem)


def save_movement(stl_points, label_name):
    new_stl_points = []
    for i in len(stl_points)[0]:
        new_points = []
        for d in range(len(degree)):
            points = stl_points[i][d]
        new_stl_points.append(new_points)


def pca(train_images,test_images, pca_components):
    pass


if __name__ == '__main__':
    points, points_aug, lbl_name, err_name, deg = main()  # Output as list[list[numpy]] (example_data, degrees, points)
    print(type(points))
    print(len(points))
    print(type(points[0]))
    print(len(points[0]))
    print(type(points[0][0]))
    print(len(points[0][0]))
    # save_image(points, points_aug, lbl_name, err_name)
    print("pre_processing.py: done")
