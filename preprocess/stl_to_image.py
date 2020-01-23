"""
This file is to convert stl file and csv file into cross_section image
After running this, use image_to_tfrecord.py to turn images into tfrecord
"""

# Import Libraries
import os
from utils.open_save_file import save_plot, save_coordinate, save_file, get_cross_section_label
import numpy as np
import json
from multiprocessing import cpu_count
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

ProgressBar().register()


def get_coor_distance(stl_points, mode_remove):
    """
    Only can be used for 'normalize_coor' function, calculate distance between points
    :param stl_points:  List of points
    :param mode_remove: True: Undersampling, False: Oversampling
    :return:            List of distance
    """
    distance = []
    if mode_remove:  # Find distance between points that point and the next two point
        for i in range(1, (np.shape(stl_points)[0]) - 1):
            distance.append(np.linalg.norm(stl_points[i - 1, :] - stl_points[i, :]) +
                            np.linalg.norm(stl_points[i, :] - stl_points[i + 1, :]))
    else:
        for i in range((np.shape(stl_points)[0]) - 1):  # Find distance between that point and next point
            distance.append(np.linalg.norm(stl_points[i, :] - stl_points[i + 1, :]))
    return distance


# To use with pandas.apply
def point_sampling_wrapper(point_list):
    for d_index in range(len(degree)):
        point_list[d_index] = point_sampling(point_list[d_index], fix_amount)
    return point_list


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
        while np.shape(stl_points)[0] > coor_amount:
            # Calculate distance between two points which are two space away (Actually focus on i+1, behind and forward)
            distance = get_coor_distance(stl_points, True)
            distance_index = np.argsort(distance)
            remove_index = distance_index[0]  # Choose min distance as the index to remove
            stl_points = np.delete(stl_points, remove_index + 1, axis=0)  # Remove from the points
    else:  # Oversampling
        # Add point between two points with most distance
        while np.shape(stl_points)[0] < coor_amount:
            # Find distance between each point
            distance = get_coor_distance(stl_points, False)
            distance_index = np.argsort(distance)
            add_index = distance_index[-1]  # Choose max distance as the index to add coordinate
            new_point = (stl_points[add_index, :] + stl_points[add_index + 1, :]) / 2
            stl_points = np.insert(stl_points, add_index + 1, new_point, axis=0)
    return stl_points


def save_image(im_data, out_directory="./data/cross_section"):
    """
    Save all cross-section into an png image
    :param im_data: pd.DataFrame with ['points'] and ['name'] as columns
    :param out_directory: Directory to save images
    :return: File saved: Images, error_file.json, config.json
    """
    # Save data as png image
    png_name = "PreparationScan"
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    # for j in range(len(label_name)):
    #     save_plot(stl_points[j], out_directory, "%s_%s" % (png_name, label_name[j]), degree, marker='x')
    #     if j % 50 == 0:
    #         print("Saved %s out of %s" % (j, len(label_name)))
    ddf = dd.from_pandas(im_data, npartitions=cpu_count() * 2)
    ddf.apply(save_plot, meta=im_data, args=(out_directory, degree), axis=1).compute(scheduler='processes')
    print("Finished saving data")


def stl_point_to_movement(stl_points):  # stl_points is list of points (for each degree)
    """
    Get difference between each coordinates instead
    :param stl_points: list of points
    :return:
    """
    new_points_sample = []
    for stl_point_image in stl_points:  # stl_point_image is one degree of cross-section
        difference = stl_point_image[1:, :] - stl_point_image[0:-1, :]  # Find difference between each position
        new_points_sample.append(difference)
    return new_points_sample


def save_stl_point(im_data, out_directory="./data/coordinates", use_diff=True):
    """
    Save all cross-section into an .npy
    :param im_data: pd.DataFrame with ['points'] and ['name'] as columns
    :param out_directory: Directory to save images
    :param use_diff: Boolean, If true, will use the vector between every two coordinate instead
    :return: File saved: npy, error_file.json, config.json
        """
    # Save data as .npy file
    coor_name = "PreparationScan"
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    # This convert coordinates into vector between each coordinate
    if use_diff:
        ddf = dd.from_pandas(im_data['points'], npartitions=cpu_count() * 2)
        im_data['points'] = ddf.apply(stl_point_to_movement, meta=im_data['points']).compute(
            scheduler='processes')
        # image_data['points'] = stl_point_to_movement(image_data['points'])
    # for j in range(len(label_name)):
    #     save_coordinate(stl_points[j], out_directory, "%s_%s" % (coor_name, label_name[j]), degree)
    ddf = dd.from_pandas(im_data, npartitions=cpu_count() * 2)
    ddf.apply(save_coordinate, meta=im_data, args=(out_directory, coor_name, degree), axis=1).compute(
        scheduler='processes')


# augment_config = [0, 0.5, 1]
a_range = 5
step = 1
augment_config = [i for i in np.arange(-a_range, a_range + 0.1, step)] + [i for i in
                                                                          np.arange(180 - a_range, 180.1 + a_range,
                                                                                    step)]
# augment_config = [i for i in np.arange(-5, 5.1, 1)] + [i for i in np.arange(-175, 185.1, 1)]
print("Augment Config:", len(augment_config), augment_config)
degree = [0, 45, 90, 135]

# Fetch stl file and save as either image or .npy file of coordinates
if __name__ == '__main__':
    # Output 'points' as list[list[numpy]] (example_data, degrees, points)
    save_coor = False
    save_img = True
    is_fix_amount = True
    fix_amount = 300  # Sampling coordinates to specified amount
    use_real_point = True  # Use actual point, else will use difference between each point instead

    # data_type, stat_type will not be used unless you want to look at lbl value
    image_data, error_name, header = get_cross_section_label(degree=degree,
                                                             augment_config=augment_config,
                                                             # folder_name='../../global_data/stl_data_debug',
                                                             # csv_dir='../../global_data/Ground Truth Score_debug.csv',
                                                             )
    # points_all = image_data.pop('points')
    use_diff = not use_real_point
    if is_fix_amount:
        if use_diff:
            fix_amount = fix_amount + 1  # Compensate for the missing data when finding diffrence
        print("Adjusting number of coordinates... Takes a long time")
        ddf = dd.from_pandas(image_data['points'], npartitions=cpu_count() * 2)
        image_data['points'] = ddf.apply(point_sampling_wrapper, meta=image_data['points']).compute(
            scheduler='processes')
        # points_all = points_all.swifter.apply(point_sampling_wrapper)

    if save_coor:
        file_dir = "../data/coor_debug_nofix"
        print("Start saving coordinates at", os.path.abspath(file_dir))
        # Save image (as coordiantes)
        save_stl_point(image_data, out_directory=file_dir, use_diff=use_diff)

        # Save names with error, for future use
        with open(file_dir + '/error_file.json', 'w') as filehandle:
            json.dump({'error_name': error_name}, filehandle)
        with open(file_dir + '/config.json', 'w') as filehandle:
            json.dump({'degree': degree, 'augment_config': augment_config}, filehandle)
        # Save score as csv file
        image_data.drop('points', axis=1).to_csv(os.path.join(file_dir, "score.csv"), index=False)
        print("Finished saving coordinates")

    if save_img:
        image_dir = "../data/image_14aug_new"
        print("Start saving images at", os.path.abspath(image_dir))

        # Save image
        save_image(image_data, out_directory=image_dir)
        # Save names with error, for future use
        with open(image_dir + '/error_file.json', 'w') as filehandle:
            json.dump({'error_name': error_name}, filehandle)
        with open(image_dir + '/config.json', 'w') as filehandle:
            json.dump({'degree': degree, 'augment_config': augment_config}, filehandle)
        # Save score as csv file
        image_data.drop('points', axis=1).to_csv(os.path.join(image_dir, "score.csv"), index=False)

    print("stl_to_image.py: done")
