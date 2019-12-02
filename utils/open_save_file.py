import os
import json
import csv
import random
import numpy as np
import matplotlib as mpl
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

mpl.use('TkAgg')
import matplotlib.pyplot as plt


def get_file_name(folder_name='../global_data/', file_name=None, exception_file=None):
    """
    Search for all filename within folder that end "file name", can except a specific filename
    :param folder_name:     Folder direcory to search
    :param file_name:       Retrieve only filename specified, can be None
    :param exception_file:  List, file name to be excluded
    :return: file_dir       List of full directory of each file
             folder_dir     List of number of folder (Use for specific format to identify data label)
    """
    print("get_file_name: Import from %s, searching for file name with %s" % (os.path.abspath(folder_name), file_name))
    if exception_file is None:
        exception_file = []

    file_dir = list()
    folder_dir = list()
    for root, dirs, files in os.walk(folder_name):
        for filename in files:
            if file_name is None:  # Add everything if no file_name specified
                if filename in exception_file:
                    pass
                else:
                    file_dir.append(os.path.abspath(os.path.join(root, filename)))
            else:
                if filename.split('/')[-1] == file_name:
                    file_dir.append(os.path.abspath(os.path.join(root, filename)))
                    folder_dir.append(root.split('/')[-1].split('-')[0])  # File name always has format: "[id]-2"
    file_dir.sort()
    folder_dir.sort()
    print("get_file_name: Uploaded %d file names" % len(file_dir))
    return file_dir, folder_dir


def get_label(dataname, stattype, double_data=False, one_hotted=False, normalized=False,
              file_dir='../global_data/Ground Truth Score_new.csv'):
    """
    Get label of Ground Truth Score.csv file
    :param dataname:    String, Type of label e.g. [Taper/Occ]
    :param stattype:    String, Label measurement e.g [average/median]
    :param double_data: Boolean, Double amount of data of label, for augmentation **Not using anymore**
    :param one_hotted:  Boolean, Return output as one-hot data
    :param normalized:  Boolean, Normalize output to 0-1 (Not applied for one hot)
    :param file_dir:    Directory of csv file
    :return: labels     List of score of requested dat
             label_name List of score name, used to identify order of data
    """
    label_name_key = ["Occ_B", "Occ_F", "Occ_L", "Occ_Sum", "BL", "MD", "Taper_Sum", "Integrity", "Width", "Surface",
                      "Sharpness"]
    label_name = dict()
    for i, key in enumerate(label_name_key):
        label_name[key] = 3 * i
    label_max_score = {"Occ_B": 5, "Occ_F": 5, "Occ_L": 5, "Occ_Sum": 15,
                       "BL": 5, "MD": 5, "Taper_Sum": 10, "Integrity": 5, "Width": 5, "Surface": 5, "Sharpness": 5}
    stat_type = {"average": 1, "median": 2}

    try:
        data_column = label_name[dataname]
    except KeyError:
        raise Exception(
            "Wrong dataname, Type as %s, Valid name: %s" % (dataname, label_name_key))
    try:
        if one_hotted & (stattype == 1):
            stattype = 2
            print("Note: One-hot mode only supported median")
        label_column = data_column
        avg_column = data_column + 1
        data_column = data_column + stat_type[stattype]  # Shift the interested column by one or two, depends on type
    except KeyError:
        raise Exception("Wrong stattype, Type as %s, Valid name: (\"average\",\"median\")" % stattype)

    max_score = label_max_score[dataname]
    labels_name = []
    labels_data = []
    avg_data = []
    print("get_label: Import from %s" % os.path.join(file_dir))
    with open(file_dir) as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        header = True
        for row in read_csv:
            if header:
                header = False
            else:
                try:
                    if row[data_column] != '':
                        label = row[label_column]
                        val = row[data_column]
                        avg_val = row[avg_column]
                        if one_hotted or (not normalized):  # Don't normalize if output is one hot encoding
                            normalized_value = int(val)  # Turn string to int
                        else:
                            normalized_value = float(val) / max_score  # Turn string to float
                        labels_name.append(label)
                        labels_data.append(normalized_value)
                        avg_data.append(float(avg_val))
                except IndexError:
                    print("Data incomplete, no data of %s, or missing label in csv file" % file_dir)

    # If consider median data on anything except Taper_Sum/Occ_sum and does not normalized
    if stattype is "median" and (not normalized) and dataname is not "Occ_Sum" and dataname is not "Taper_Sum":
        labels_data = readjust_median_label(labels_data, avg_data)

    # Sort data by name
    labels_name, labels_data = zip(*sorted(zip(labels_name, labels_data)))
    labels_name = list(labels_name)  # Turn tuples into list
    labels_data = list(labels_data)

    # Duplicate value if required
    if double_data:
        labels_data = [val for val in labels_data for _ in (0, 1)]

    # Turn to one hotted if required
    if one_hotted:
        one_hot_labels = list()
        for label in labels_data:
            label = int(label)
            one_hot_label = np.zeros(max_score + 1)
            one_hot_label[label] = 1
            one_hot_labels.append(one_hot_label)
        print("get_label: Upload one-hotted label completed (as a list): %d examples" % (len(one_hot_labels)))
        return one_hot_labels, labels_name
    else:
        print("get_label: Upload non one-hotted label completed (as a list): %d examples" % (len(labels_data)))
        return labels_data, labels_name


def get_cross_section_label(degree, augment_config=None, folder_name='../../global_data/stl_data',
                            file_name="PreparationScan.stl",
                            csv_dir='../../global_data/Ground Truth Score_new.csv'):
    """
    Get coordinates of stl file and label from csv file
    :param degree:          List of rotation angles
    :param augment_config:  List of all augmentation angles
    :param folder_name:     String, folder directory of stl file
    :param csv_dir:         String, file directory of label (csv file)
    :param file_name:       String, filename can be None
    :return:
    stl_points_all          List of all point (ndarray)
    label_all               Dict of label(Check 'data_type'), name, error_name
    """
    from utils.stl_slicer import get_cross_section

    if augment_config is None:
        augment_config = [0]

    # Get data and transformed to cross-section image.
    data_type = ["Occ_B", "Occ_F", "Occ_L", "Occ_Sum", "BL", "MD", "Taper_Sum", "Integrity", "Width", "Surface",
                 "Sharpness"]  # CSV header
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
    label_all["name"] = []
    label_all["error_name"] = []
    for i in range(len(name_dir)):
        # Prepare two set of list, one for data, another for augmented data
        label_name_temp = []
        points_all = get_cross_section(name_dir[i], 0, degree, augment=augment_config, axis=1)
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
        stl_points_all += stl_points  # Add these points to the big one
        label_all["name"] += label_name_temp  # Also add label name to the big one
        label_all["error_name"] += error_file_names  # Same as error file

    # The output is list(examples) of list(degrees) of numpy array (N*2 coordinates)
    for label_name in label_all["name"]:
        print("Finished with %d examples" % (len(label_name)))

    print("Max amount of coordinates: %s, min  coordinates: %s" % (max_point, min_point))
    return stl_points_all, label_all, label_header


# The two functions below is used for new type of data, currently on prototype
def get_label_new_data(dataname, file_dir='../global_data/new_score(okuyama).csv',
                       one_hotted=False, normalized=False, ):
    """
    Get label of Ground Truth Score.csv file for new set of score
    :param dataname:    String, Type of label e.g. [Taper/Occ]
    :param file_dir:    Directory of csv file
    :param one_hotted:  Boolean, Return output as one-hot data
    :param normalized:  Boolean, Normalize output to 0-1 (Does not work if one_hotted is True)
    :return: labels     List of score of requested dat
             label_name List of score name, used to identify order of data
    """
    label_name_key = {"Taper": 1, "Width": 2, "Sharpness": 4}
    max_score = 5

    try:
        data_column = label_name_key[dataname]
    except KeyError:
        raise Exception(
            "Wrong dataname, Type as %s, Valid name: %s" % (dataname, label_name_key))

    labels_name = []
    labels_data = []
    print("get_label: Import from %s" % os.path.join(file_dir))
    with open(file_dir) as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        header = True  # Ignore first row
        for row in read_csv:
            if header:
                header = False
            else:
                try:
                    if row[data_column] != '':
                        label = row[0]
                        val = row[data_column]
                        if one_hotted or (not normalized):  # Don't normalize if output is one hot encoding
                            normalized_value = int(val)  # Turn string to int
                        else:
                            normalized_value = float(val) / max_score  # Turn string to float
                        labels_name.append(label)
                        labels_data.append(normalized_value)
                except IndexError:
                    print("Data incomplete, no data of %s, or missing label in csv file" % file_dir)

    # Sort data by name
    labels_name, labels_data = zip(*sorted(zip(labels_name, labels_data)))
    labels_name = list(labels_name)  # Turn tuples into list
    labels_data = list(labels_data)

    # Turn to one hotted if required
    if one_hotted:
        one_hot_labels = list()
        for label in labels_data:
            label = (label - 1) / 2
            one_hot_labels.append(np.array([int(i == label) for i in range(3)]))
        print("get_label: Upload one-hotted label completed (as a list): %d examples" % (len(one_hot_labels)))
        return one_hot_labels, labels_name
    else:
        print("get_label: Upload non one-hotted label completed (as a list): %d examples" % (len(labels_data)))
        return labels_data, labels_name


def get_cross_section_label_new_data(degree, augment_config=None, folder_name='../../global_data/stl_data',
                                     file_name="PreparationScan.stl",
                                     csv_dir='../../global_data/new_score(okuyama).csv'):
    """
    Get coordinates of stl file and label from csv file
    :param degree:          List of rotation angles
    :param augment_config:  List of all augmentation angles
    :param folder_name:     String, folder directory of stl file
    :param csv_dir:         String, file directory of label (csv file)
    :param file_name:       String, filename can be None
    :return:
    stl_points_all          List of all point (ndarray)
    label_all               Dict of label(Check 'data_type'), name, error_name
    """
    from utils.stl_slicer import get_cross_section

    if augment_config is None:
        augment_config = [0]

    # Get data and transformed to cross-section image.
    data_type = ["Taper", "Width", "Sharpness"]  # CSV header
    # stat_type = ["median"]

    name_dir, image_name = get_file_name(folder_name=folder_name, file_name=file_name)

    # Put label and header of each score category into dictionary
    label = dict()
    label_header = ["name"]
    for d in data_type:
        l, label_name = get_label_new_data(d, one_hotted=False, normalized=False, file_dir=csv_dir)
        label[d] = l
        label_header.append(d)
        for deg, name in zip(degree, label_name[0:len(degree)]):
            assert int(name.split('_')[-1]) in degree, "Degree input is not the same as label in {}, found {}".format(
                degree, name)
    label["name"] = label_name

    # To verify number of coordinates
    min_point = 1000
    max_point = 0

    # List of all cross-section ndarray (Augmented)
    stl_points_all = []

    # Create a dictionary of empty list
    label_all = {k: [] for k in dict.fromkeys(label.keys())}
    label_all["name"] = label["name"]
    label_all["error_name"] = []

    for l_index, name in enumerate(label["name"]):
        name_id = name.split('_')[0]
        deg = int(name.split('_')[1])

        try:
            name_index = image_name.index(name_id)
            points_all = get_cross_section(name_dir[name_index], 0, deg, augment=augment_config, axis=1)

            # points_all is list of all possible augmentation but some might be None due to stl file corruption
            for p_index, point in enumerate(points_all):
                augment_val = augment_config[p_index]

                # Convert augment angle to string, change negative to 'n' instead
                if augment_val >= 0:
                    augment_val = str(augment_val)
                else:
                    augment_val = "n" + str(abs(augment_val))

                # Check if cross-section doesn't have error
                if point is None:
                    label_all["error_name"].append("{}_{}".format(name, augment_val))
                else:
                    stl_points_all.append(point)
                    label_all["name"].append("{}_{}".format(name, augment_val))
                    for key in label.keys():
                        print(len(label[key]))
                        print(l_index)
                        label_all[key].append(label[key][l_index])
                    # Check number of points in cross-section
                    if len(point[0]) > max_point:
                        max_point = len(point[0])
                    elif len(point[0]) < min_point:
                        min_point = len(point[0])
        except ValueError:  # Cannot find file with this id
            pass

    # The output is list(examples) of list(degrees) of numpy array (N*2 coordinates)
    print("Finished with {} examples".format(len(label_all["name"])))

    print("Max amount of coordinates: %s, min  coordinates: %s" % (max_point, min_point))
    return stl_points_all, label_all, label_header


def predict_get_cross_section(degree, augment_config=None, folder_name='../../global_data/stl_data',
                              file_name="PreparationScan.stl"):
    """
    Get coordinates of stl file from csv file, only use for prediction
    :param degree:          List of rotation angles
    :param augment_config:  List of all augmentation angles, if still want to do
    :param folder_name:     String, folder directory of stl file
    :param file_name:       String, filename can be None
    :return:
    stl_points_all          List of all point (ndarray)
    error_file_names_all    List of label name that has error
    """
    from utils.stl_slicer import get_cross_section

    if augment_config is None:
        augment_config = [0]

    name_dir, image_name = get_file_name(folder_name=folder_name, file_name=file_name)

    # To verify number of coordinates
    min_point = 1000
    max_point = 0

    file_name_all = []
    stl_points_all = []
    error_file_names_all = []
    for i in range(len(name_dir)):
        # Prepare two set of list, one for data, another for augmented data
        points_all = get_cross_section(name_dir[i], 0, degree, augment=augment_config, axis=1)
        stl_points = []
        error_file_names = []  # Names of file that cannot get cross-section image
        file_name = []
        for index, point in enumerate(points_all):
            augment_val = augment_config[index]
            if augment_val < 0:
                augment_val = "n" + str(abs(augment_val))
            else:
                augment_val = str(abs(augment_val))
            if point is None:  # If the output has error, remove label of that file
                error_file_names.append(image_name[i] + "_" + augment_val)
            else:
                file_name.append(image_name[i] + "_" + augment_val)
                stl_points.append(point)
                if len(point[0]) > max_point:
                    max_point = len(point[0])
                if len(point[0]) < min_point:
                    min_point = len(point[0])

        stl_points_all += stl_points  # Add these points to the big one
        file_name_all += file_name
        error_file_names_all += error_file_names  # Same as error file
    print("Max amount of coordinates: %s, min  coordinates: %s" % (max_point, min_point))
    return stl_points_all, file_name_all, error_file_names_all


def readjust_median_label(label, avg_data):
    """
    Since some score can only be in a certain range (E.g. 1,3 or 5), if any median score that is outside of this range
    appear, move it to the nearby value instead based on average. (Round the other direction from average)
    :param label: List of actual score
    :param avg_data: Average value of the whole data
    :return:
    """
    possible_value = [1, 3, 5]
    if len(label) != len(avg_data):
        raise ValueError("Size of label and average data is not equal")
    for i, label_value in enumerate(label):
        if not (label_value in possible_value):
            # Check if value is over/under boundary, if so, choose the min/max value
            if label_value < possible_value[0]:
                label[i] = possible_value[0]
            elif label_value > possible_value[-1]:
                label[i] = possible_value[-1]
            else:
                if label_value > avg_data[i]:  # If median is more than average, round up
                    label[i] = min(filter(lambda x: x > label_value, possible_value))
                else:  # If median is less or equal to average, around down
                    label[i] = max(filter(lambda x: x < label_value, possible_value))
    return label


def save_plot(coor_list, out_directory, image_name, degree, file_type="png", show_axis=False):
    """
    Save list of coordinates as a PNG image
    :param coor_list:           List of ndarrays <- get from stlSlicer should have slices = len(degree)
    :param out_directory:       String, Directory to save output
    :param image_name:          List of name of the image
    :param degree:              List of angles used in, add the angle in file name as well
    :param file_type:           [Optional], such as png,jpeg,...
    :param show_axis:           [Optional], if true, will show axis
    :return:                    File saved at out_directory
    """
    if len(coor_list) != len(degree):
        raise ValueError("Number of degree is not equal to %s, found %s", (len(degree), len(coor_list)))
    out_directory = os.path.abspath(out_directory)
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    for d in range(len(degree)):
        coor = coor_list[d]
        fullname = "%s_%d.%s" % (image_name, degree[d], file_type)
        output_name = os.path.join(out_directory, fullname)

        dpi = 100
        img_size = 800
        fig = plt.figure(figsize=(img_size / dpi, img_size / dpi), dpi=dpi)
        ax = fig.gca()
        ax.set_autoscale_on(False)
        min_x, max_x, min_y, max_y = -6.5, 6.5, -6.5, 6.5
        if min(coor[:, 0]) < min_x or max(coor[:, 0]) > max_x:
            ax.plot(coor[:, 0], coor[:, 1], linewidth=1.0)
            ax.axis([min_x - 1, max_x + 1, min_y, max_y])
            fig.savefig(os.path.join(out_directory, "bugged"), bbox_inches='tight')
            print("Bugged at %s" % output_name)
            raise ValueError("X-coordinate is beyond limit axis (%s,%s)" % (min_x, max_x))

        if min(coor[:, 1]) < min_y or max(coor[:, 1]) > max_y:
            ax.plot(coor[:, 0], coor[:, 1], linewidth=1.0)
            ax.axis([min_x, max_x, min_y - 1, max_y + 1])
            fig.savefig(os.path.join(out_directory, "bugged"), bbox_inches='tight')
            print("Bugged at %s" % output_name)
            raise ValueError("Y-coordinate is beyond limit axis (%s,%s)" % (min_y, max_y))
        ax.plot(coor[:, 0], coor[:, 1], 'k', linewidth=1.0)
        ax.axis([min_x, max_x, min_y, max_y])

        if not show_axis:
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        fig.savefig(output_name, bbox_inches='tight')
        plt.close('all')
    # print("Finished plotting for %d images with %d rotations at %s" % (len(coor_list), len(degree), out_directory))


# Save coordinate as .npy file
def save_coordinate(coor_list, out_directory, image_name, degree):
    """
    Save list of coordinates as a .npy file
    :param coor_list: List of coordinates, should have length equals to len(degree)
    :param out_directory: Output directory
    :param image_name: Name of image
    :param degree: List of degree of rotation
    :return: File saved at out_directory
    """
    # Check if size coordinate, image name has same length
    if len(coor_list) != len(degree):
        raise ValueError("Number of degree is not equal to %s, found %s", (len(degree), len(coor_list)))
    out_directory = os.path.abspath(out_directory)
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    # for corr_index in range(len(coor_list)):
    for deg_index in range(len(degree)):
        coor = coor_list[deg_index]
        fullname = "%s_%d.npy" % (image_name, degree[deg_index])
        output_name = os.path.join(out_directory, fullname)
        np.save(output_name, coor)


def split_train_test(packed_image, tfrecord_name, configs, class_weight):
    train_amount = int(configs['train_eval_ratio'] * len(packed_image))  # Calculate amount of training data

    train_data = packed_image[0:train_amount]
    eval_data = packed_image[train_amount:]

    # Save names of files of train address
    file_name = "../data/tfrecord/%s/%s_0.json" % (tfrecord_name, tfrecord_name)
    with open(file_name, 'w') as filehandle:
        # json.dump({"class_weight": class_weight, "train": train_data, "eval": eval_data}, filehandle, indent=4,
        json.dump({"class_weight": class_weight}, filehandle, indent=4,
                  sort_keys=True,
                  separators=(',', ': '), ensure_ascii=False)
    return train_data, eval_data


def split_kfold(grouped_address, k_num, seed=0):
    """
    Split data into multiple set using KFold algorithm
    :param grouped_address:     List, all data ready to be shuffled [[X1,y1],[X2,y2],...]
    :param k_num:               Int, number of k-fold
    :return:                    List of Train, Eval data
    """
    # kfold = KFold(k_num, shuffle=True,random_state=0)
    kfold = StratifiedKFold(k_num, shuffle=False, random_state=seed)
    data, label = [list(e) for e in zip(*grouped_address)]
    train_address = []
    eval_address = []
    # for train_indices, test_indices in kfold.split(grouped_address):
    new_label = [i["Sharpness_median"] for i in label]
    for train_indices, test_indices in kfold.split(data, new_label):
        train_address_fold = []
        test_address_fold = []
        for train_indice in train_indices:
            train_address_fold.append([data[train_indice], label[train_indice]])
        for test_indice in test_indices:
            test_address_fold.append([data[test_indice], label[test_indice]])
        train_address.append(train_address_fold)
        eval_address.append(test_address_fold)
    return train_address, eval_address


def get_input_and_label(tfrecord_name, dataset_folder, configs, seed, get_data=False, k_fold=None):
    """
    This function is specifically used in image_to_tfrecord, fetching
    :param tfrecord_name:   String, Directory of output file
    :param dataset_folder:  String, Folder directory of input data [Only data in this folder]
    :param configs:         Dictionary, containing {numdeg, train_eval_ratio, data_type}
    :param seed:            Integer, to determine randomness
    :param get_data:        Boolean, if true will return raw data instead of file name
    :param k_fold:          Integer, parameter for KFold. If None, will have no K-fold
    :return:                Train, Eval: Tuple of list[image address, label]. Also save some txt file
                            loss_weight: numpy array use for loss weight
    """
    numdeg = configs['numdeg']

    # Get image address and labels
    image_address, _ = get_file_name(folder_name=dataset_folder, file_name=None,
                                     exception_file=["config.txt", "error_file.txt", "score.csv"])

    labels, _ = read_score(os.path.join(dataset_folder, "score.csv"),
                           data_type=configs['data_type'])

    if len(image_address) / len(labels) != numdeg or len(image_address) == 0:
        print(image_address)
        raise Exception(
            '# of images and labels is not compatible: %d images, %d labels. '
            'Expected # of images to be %s times of label' % (
                len(image_address), len(labels), numdeg))
    # Create list of file names of 0 degree
    image_name = []
    for i in range(len(labels)):
        image_name.append(image_address[i * numdeg].split('.')[0])  # Only 0 degree

    # Load data if get_data is True, image address will now be list of ndarray instead
    if get_data:
        image_address_temp = []
        for addr in image_address:
            image_address_temp.append(np.load(addr))
        image_address = image_address_temp

    # Group up images and label together, then pack all augmentation together, then shuffle
    packed_image = []
    for i in range(len(labels)):
        packed_image.append([image_address[i * numdeg:(i + 1) * numdeg], labels[i]])
    if k_fold is None:
        # Pack data
        temp_image = []  # New temporary address packs augmented data together
        # temp_image_name = []
        temp_name = {}
        # Loop over each file, packed image from same stl file together
        for g_add, ex_g_add in zip(packed_image, image_name):
            if os.path.basename(ex_g_add).split('_')[1] in temp_name:
                temp_image[temp_name[os.path.basename(ex_g_add).split('_')[1]]].append(g_add)
                # temp_image_name[temp_name[os.path.basename(ex_g_add).split('_')[1]]].append(ex_g_add)
            else:
                temp_name[os.path.basename(ex_g_add).split('_')[1]] = len(temp_image)
                temp_image.append([g_add])
                # temp_image_name.append([ex_g_add])

        # Zip, shuffle, unzip
        random.Random(seed).shuffle(temp_image)
        print(len(list(temp_name.keys())))

        # Unpack data
        packed_image = [item for sublist in temp_image for item in sublist]
        # image_name = [item.split('/')[-1] for sublist in temp_image_name for item in sublist]
    else:
        # Pack data
        temp_image = []  # New temporary address packs augmented data together
        temp_name = {}
        for g_add, ex_g_add in zip(packed_image, image_name):
            if os.path.basename(ex_g_add).split('_')[1] in temp_name:
                temp_image[temp_name[os.path.basename(ex_g_add).split('_')[1]]].append(g_add)
            else:
                temp_name[os.path.basename(ex_g_add).split('_')[1]] = len(temp_image)
                temp_image.append([g_add])
        random.Random(seed).shuffle(packed_image)
        # Unpack data
        packed_image = [item for sublist in temp_image for item in sublist]
    # Calculate loss weight
    _, label = [list(e) for e in zip(*packed_image)]

    class_weight = {}
    for c in configs['data_type']:
        if not c == "name":
            score = [i[c] for i in label]
            c_weight = compute_class_weight('balanced', np.unique(score), score)  # Assume score always 1,3,5
            if np.shape(c_weight)[0] < 3:  # Sometime class 1 is missing, use weight 1 instead
                possible_score = [1, 3, 5]
                for index, sc in enumerate(possible_score):
                    if not np.any(np.unique(score) == sc):  # Check if data doesn't exist, insert 1
                        try:
                            c_weight = np.insert(c_weight, index, 1)
                        except IndexError:
                            c_weight = np.concatenate(c_weight, 1)
            class_weight[c] = c_weight.tolist()

    if k_fold is not None:  # If k-cross validation, output will be list of each k-fold
        train_image_temp, eval_image_temp = split_kfold(packed_image, k_fold, seed)
        train_image = []
        eval_image = []
        for i in range(k_fold):
            single_train_image = train_image_temp[i]
            single_eval_image = eval_image_temp[i]
            if not get_data:  # Put in special format for writing tfrecord (pipeline)
                single_train_image = tuple(
                    [list(e) for e in zip(*single_train_image)])  # Convert to tuple of list[image address, label]

                single_eval_image = tuple(
                    [list(e) for e in zip(*single_eval_image)])  # Convert to tuple of list[image address, label]
                print(
                    "Train files: %d, Evaluate Files: %d" % (len(single_train_image[0]), len(single_eval_image[0])))
            else:
                print("Train files: %d, Evaluate Files: %d" % (len(single_train_image), len(single_eval_image)))
            train_image.append(single_train_image)
            eval_image.append(single_eval_image)

            # Save label distribution for weight balancing
            file_name = "../data/tfrecord/%s/%s_%s.json" % (tfrecord_name, tfrecord_name, i)
            with open(file_name, 'w') as filehandle:
                json.dump({"class_weight": class_weight}, filehandle, indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)

    else:  # Normal operation, but will return as a list of one big data as well
        train_image, eval_image = split_train_test(packed_image, tfrecord_name, configs, class_weight)
        if not get_data:  # Put in special format for writing tfrecord (pipeline)
            train_image = tuple(
                [list(e) for e in zip(*train_image)])  # Convert to tuple of list[image address, label]

            eval_image = tuple(
                [list(e) for e in zip(*eval_image)])  # Convert to tuple of list[image address, label]
            print("Train files: %d, Evaluate Files: %d" % (len(train_image[0]), len(eval_image[0])))
        else:
            print("Train files: %d, Evaluate Files: %d" % (len(train_image), len(eval_image)))
        train_image = [train_image]
        eval_image = [eval_image]

    return train_image, eval_image


def read_file(csv_dir, header=False):
    """
    Read csv file
    :param csv_dir: String, directory of file
    :param header:  Boolean, true will read first row as header
    :return: data:          List of data on each row
             header_name:   List of header name
    """
    header_name = []
    data = []
    with open(csv_dir) as csvFile:
        read_csv = csv.reader(csvFile, delimiter=',')
        for row in read_csv:
            if header:
                header_name.append(row)
                header = False
            else:
                data.append(row)
    if not header_name:
        return data
    else:
        return data, header_name


def read_score(csv_dir, data_type):
    """
    Extension to read_file, specifically used to read csv file made from stl_to_image.py
    :param csv_dir:
    :param data_type: List of data type to fetch
    :return:
    """
    # Prevent case that input is not list
    if not type(data_type) is list:
        data_type = [data_type]

    data = []
    data_name = []
    with open(csv_dir) as csvFile:
        read_csv = csv.reader(csvFile, delimiter=',')
        is_header = True
        for row in read_csv:
            if is_header:
                header_name = row
                data_index = [header_name.index(i) for i in data_type]  # If ValueError, data_type is not in csv header
                # data_index = header_name.index(data_type)  # Find index of data
                is_header = False
            else:
                data_name.append(row[0])  # Assume name is first column
                data_dict = {}
                for index in data_index:
                    if header_name[index] == "name":
                        data_dict[header_name[index]] = row[index]
                    else:
                        data_dict[header_name[index]] = int(row[index])
                data.append(data_dict)
    return data, data_name


# one_row = true means that all data will be written in one row
def save_file(csv_dir, all_data, field_name=None, write_mode='w', data_format=None, create_folder=True):
    """
    Save file to .csv
    :param csv_dir:         String, Directory + file name of csv (End with .csv)
    :param all_data:        Data to save
    :param field_name:      List of field name if needed
    :param write_mode:      String, 'w' or 'a'
    :param data_format:     String, depending on data format: {"dict_list", "double_list"}
    :param create_folder:   Boolean, will create folder if not exist
    :return:
    """
    # Create a folder if not exist
    if create_folder:
        direct = os.path.dirname(csv_dir)
        if not os.path.exists(direct):
            os.makedirs(direct)
    if data_format == "dict_list":  # Data format: {'a':[a1, a2], 'b':[b1,b2]} -> Size of all list must be the same
        if field_name is None:
            raise ValueError("Need field_name ")
        with open(csv_dir, write_mode) as csvFile:
            writer = csv.DictWriter(csvFile, fieldnames=field_name)
            writer.writeheader()
            for i in range(len(all_data[field_name[0]])):
                temp_data = dict()
                for key in field_name:
                    temp_data[key] = all_data[key][i]
                writer.writerow(temp_data)
    elif data_format == "header_only":
        if field_name is None:
            raise ValueError("Need filed name")
        with open(csv_dir, write_mode) as csvFile:
            writer = csv.DictWriter(csvFile, fieldnames=field_name)
            writer.writeheader()
    elif data_format == "double_list":  # Data format: [[a1,a2],[b1,b2,b3]]
        with open(csv_dir, write_mode) as csvFile:
            writer = csv.writer(csvFile)
            for data in all_data:
                writer.writerow(data)  # May need to add [data] in some case
    elif data_format == "one_row":  # Data format: [a1,a2,a3]
        with open(csv_dir, write_mode) as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(all_data)
    else:
        with open(csv_dir, write_mode) as csvFile:
            writer = csv.writer(csvFile)
            for data in all_data:
                writer.writerow([data])  # May need to add [data] in some case


def check_exist(dictionary, **kwargs):
    """
    Check if key exist in dictionary or not, if not will replace by the value given
    :param dictionary: Dictionary parameter
    :param kwargs: key, value pair. If value is None, will raise Error when cannot find key
    :return: Same dictionary
    """
    output_dict = dictionary
    for key, value in kwargs.items():
        try:
            output_dict[key] = dictionary[key]
            # output = params[dict_name]
        except (KeyError, TypeError) as error:
            if value is None:
                raise KeyError("Parameter '%s' not defined" % key)
            else:
                output_dict[key] = value
                print("Parameters: %s not found, use default value = %s" % (key, value))
    return output_dict


# Below are unused stl-to-voxel functions
'''
def create_name(num_file, exceptions=[-1]):
    name = []
    num_files = num_file
    exception = exceptions  # In case some data is missing, add in to the list
    for i in range(1, num_files + 1):
        if i != exception[0]:
            string = "vox" + str(i)
            name.append(string)
        else:
            exception.remove(exception[0])
            if len(exception) == 0:
                exception.append(-1)
    return name


def change_size(voxel, size=(256, 256, 256)):
    dimension = voxel['dim']
    data = voxel['global_data']
    new_dim = []
    new_data = []
    for i in range(0, len(dimension)):
        dim = dimension[i]
        misX = int((256 - dim[0]) / 2)
        misY = int((256 - dim[1]) / 2)
        misZ = int((256 - dim[2]) / 2)
        new_dt = np.zeros(size)
        new_dt[misX:misX + dim[0], misY:misY + dim[1], misZ:misZ + dim[2]] = data[i]
        new_data.append(new_dt)
        new_dim.append((256, 256, 256))
    voxel = {'dim': new_dim, 'data': new_data}
    return voxel


def voxeltonumpy(num_file, exception=[-1], normalize_size=True):
    dimension = []
    voxel_data = []
    name = create_name(num_file, exception)
    print("Total number of files are: %d" % (len(name)))
    max_X = 0
    max_Y = 0
    max_Z = 0
    for i in range(0, len(name)):
        mod = importlib.import_module(name[i])
        w = mod.widthGrid
        h = mod.heightGrid
        d = mod.depthGrid
        V = mod.lookup
        dimen = (w + 1, h + 1, d + 1)  # Add 1 more pixel because value start at zero
        dimension.append(dimen)
        voxel = np.zeros(dimen)
        print("{%s} Size of model is: %s" % (name[i], str(np.shape(voxel))))
        for i in range(0, len(V)):
            x = V[i]['x']
            y = V[i]['y']
            z = V[i]['z']
            voxel[x, y, z] = 1
        voxel_data.append(voxel)
        del mod
    voxel = {'dim': dimension, 'data': voxel_data}
    if (normalize_size):
        voxel = change_size(voxel)
        print("All data set size to " + str(voxel['dim'][0]))
    return voxel


def mattovoxel(targetfolder, dataname='gridOUTPUT', foldername="mat s"):
    # "data" folder has to be outside of this file
    # Directory is /data/(targetfolder)/(foldername)/ <- This directory contain all data files

    rootfolder = '../global_data/'
    folder_dir = os.path.join(rootfolder, targetfolder, foldername)
    print("Import .mat file from folder: " + os.path.abspath(folder_dir))
    namelist = []
    for root, dirs, files in os.walk(folder_dir):
        for filename in files:
            namelist.append(filename)

    ## Check if data file exists
    num_example = len(namelist)
    if (num_example == 0):
        print("Folder directory does not exist or no file in directory")
        return None

    data = scipy.io.loadmat(os.path.join(folder_dir, namelist[0]))
    (row, col, dep) = np.shape(data[dataname])

    voxel = np.zeros((num_example, row, col, dep), dtype='float16')  # Voxel is np.array
    # voxel = []  #Voxel is a list

    # Import data into numpy array
    for i in range(0, len(namelist)):
        data = scipy.io.loadmat(os.path.join(folder_dir, namelist[i]))
        # print(namelist[i]+" uploaded")
        voxel[i, :, :, :] = data[dataname]  # In case of voxel is np.array
        # voxel.append(data['gridOUTPUT'])   #In case of voxel is list
    print("Import completed: data size of %s, %s examples" % (str((row, col, dep)), num_example))
    return voxel


def get2DImage(directory, name, singleval=False, realVal=False, threshold=253):
    # directory is full directory of folder
    # name is either 'axis1' or 'axis2'
    # singleval will only get one example (for debug)
    # realVal will skip thresholding
    namelist = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            word = filename.split("_", 1)
            word = word[-1].split(".", 1)
            if (word[0] == name):
                namelist.append(filename)
    num_im = len(namelist)
    tempim = imageio.imread(os.path.join(directory, namelist[0]))
    (w, h, d) = np.shape(tempim)
    if (singleval):
        grayim = 0.2989 * tempim[:, :, 0] + 0.5870 * tempim[:, :, 1] + 0.1140 * tempim[:, :, 2]
        if (not realVal):
            grayim = (grayim >= threshold)
        grayim = grayim.astype(int)
        print("Get 2D image from %s done with size: (%d,%d)" % (name, w, h))
        return grayim
    else:
        grayim = np.zeros((num_im, w, h))
        for i in range(0, num_im):
            image = imageio.imread(os.path.join(directory, namelist[i]))
            grayim[i, :, :] = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
        if (not realVal):
            grayim = (grayim >= threshold)
        grayim = grayim.astype(int)
        print("Get 2D images from %s done with size: (%d,%d,%d)" % (name, w, h, num_im))
        return grayim
'''
