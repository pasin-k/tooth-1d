# import sys
import os
import json
# import glob
import csv
import random
import numpy as np
import matplotlib as mpl
import collections
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from random import shuffle


def get_file_name(folder_name='../global_data/', file_name=None, exception_file=None):
    """
    Get sorted List of filenames: search for all file within folder with specific file name, ignore specific filename
    :param folder_name:     Folder direcory to search
    :param file_name:       Retrieve only filename specified, can be None
    :param exception_file:  List, file name to be excluded
    :return: file_dir       List of full directory of each file
             folder_dir     List of number of folder (Use for specific format to identify data label)
    """
    print("get_file_name: Import from %s, searching for file name with %s" % (os.path.abspath(folder_name), file_name))
    if exception_file is None:
        exception_file = []

    # if file_name is None:
    #     file_name = 'PreparationScan.stl'
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
                    folder_dir.append(root.split('/')[-1].split('-')[0])
    file_dir.sort()
    folder_dir.sort()
    print("get_file_name: Uploaded %d file names" % len(file_dir))
    return file_dir, folder_dir


# Since some score can only be in a certain range (E.g. 1,3 or 5), if any median score that is outside of this range
# appear, move it to the nearby value instead based on average. (Round the other direction from average)
def readjust_median_label(label, avg_data):
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


def get_label(dataname, stattype, double_data=False, one_hotted=False, normalized=False,
              file_dir='../global_data/Ground Truth Score_new.csv'):
    """
    Get label of Ground Truth Score.csv file
    :param dataname:    String, Type of label e.g. [Taper/Occ]
    :param stattype:    String, Label measurement e.g [Average/Median]
    :param double_data: Boolean, Double amount of data of label, for augmentation **Not using anymore
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
    # label_name = {"Occ_B": 0, "Occ_F": 3, "Occ_L": 6, "Occ_Sum": 9, "BL": 12, "MD": 15, "Taper_Sum": 18}
    label_max_score = {"Occ_B": 5, "Occ_F": 5, "Occ_L": 5, "Occ_Sum": 15,
                       "BL": 5, "MD": 5, "Taper_Sum": 10, "Integrity": 5, "Width": 5, "Surface": 5, "Sharpness": 5}
    stat_type = {"average": 1, "median": 2}

    try:
        data_column = label_name[dataname]
    except:
        raise Exception(
            "Wrong dataname, Type as %s, Valid name: %s" % (dataname, label_name_key))
    try:
        if one_hotted & (stattype == 1):
            stattype = 2
            print("Note: One-hot mode only supported median")
        label_column = data_column
        avg_column = data_column + 1
        data_column = data_column + stat_type[stattype]  # Shift the interested column by one or two, depends on type
    except:
        raise Exception("Wrong stattype, Type as %s, Valid name: (\"average\",\"median\")" % stattype)

    max_score = label_max_score[dataname]
    labels_name = []
    labels_data = []
    avg_data = []
    print("get_label: Import from %s" % os.path.join(file_dir))
    with open(file_dir) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        header = True
        for row in readCSV:
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
        # if filetype != 'list':  # If not list, output will be as numpy instead
        #     size = len(labels)
        #     labelnum = np.zeros((size,))
        #     for i in range(0, size):
        #         labelnum[i] = labels[i]
        #     labels = labelnum
        #     print("Upload label completed (as a numpy): %d examples" % (size))
        # else:
        print("get_label: Upload non one-hotted label completed (as a list): %d examples" % (len(labels_data)))
        return labels_data, labels_name


def save_plot(coor_list, out_directory, file_header_name, image_name, degree, file_type="png", show_axis=False):
    """
    Plot the list of coordinates and save it as PNG image
    :param coor_list:           List of numpy coordinates <- get from stlSlicer should have 4 cross section
    :param out_directory:       String, Directory to save output
    :param file_header_name:    String, header of name of the file, follow by image_num
    :param image_name:          List of name of the image
    :param degree:              List of angles used in, add the angle in file name as well
    :param file_type:           [Optional], such as png,jpeg,...
    :param show_axis:           [Optional], if true, will show axis
    :return:                    Save as output outside
    """
    if len(coor_list) != len(degree):
        raise ValueError("Number of degree is not equal to %s, found %s", (len(degree), len(coor_list)))
    out_directory = os.path.abspath(out_directory)
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    fig = plt.figure()

    for d in range(len(degree)):
        coor = coor_list[d]
        # plt.plot(coor[:, 0], coor[:, 1], color='black', linewidth=1)
        # plt.axis('off')
        # Name with some additional data
        fullname = "%s_%s_%d.%s" % (file_header_name, image_name, degree[d], file_type)
        # fullname = "%s_%s_%s_%d.%s" % (file_header_name, image_name, augment_number, degree[d], file_type)
        output_name = os.path.join(out_directory, fullname)

        dpi = 100
        img_size = 800
        fig = plt.figure(figsize=(img_size / dpi, img_size / dpi), dpi=dpi)
        ax = fig.gca()
        ax.set_autoscale_on(False)
        min_x, max_x, min_y, max_y = -5, 5, -6.4, 6.4
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


def split_train_test(grouped_address, example_grouped_address, tfrecord_name, configs, class_weight):
    # Split into train and test set
    data_index = list(range(len(example_grouped_address)))
    train_index = []
    eval_index = []

    train_amount = int(configs['train_eval_ratio'] * len(grouped_address))  # Calculate amount of training data
    # Open file and read the content in a list
    file_name = "./data/tfrecord/%s/%s_0.json" % (
        tfrecord_name, tfrecord_name)
    if os.path.isfile(file_name):  # Check if file exist
        with open(file_name) as filehandle:
            data_loaded = json.load(filehandle)
            for d in data_loaded['train']:
                try:
                    # print((example_grouped_address))
                    index = example_grouped_address.index(d)
                    train_index.append(index)
                    data_index.remove(index)
                except (ValueError, IndexError) as e:
                    print("Cannot find file (Train): %s" % d)
            for d in data_loaded['eval']:
                try:
                    index = example_grouped_address.index(d)
                    eval_index.append(index)
                    data_index.remove(index)
                except (ValueError, IndexError) as e:
                    print("Cannot find file (Eval): %s" % d)

        print("Use %s train examples, %s eval examples from previous tfrecords as training" % (
            len(train_index), len(eval_index)))

    # Split training and test (Split 80:20)
    train_amount = train_amount - len(train_index)
    if train_amount < 0:
        train_amount = 0
        print("imgtotfrecord: amount of training is not correct, might want to check")

    train_index.extend(data_index[0:train_amount])
    eval_index.extend(data_index[train_amount:])

    train_data = [grouped_address[i] for i in train_index]
    eval_data = [grouped_address[i] for i in eval_index]
    train_address = [example_grouped_address[i] for i in train_index]
    eval_address = [example_grouped_address[i] for i in eval_index]

    # Save names of files of train address
    file_name = "./data/tfrecord/%s/%s_0.json" % (tfrecord_name, tfrecord_name)
    with open(file_name, 'w') as filehandle:
        json.dump({"class_weight": class_weight, "train": train_address, "eval": eval_address}, filehandle, indent=4,
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


def get_input_and_label(tfrecord_name, dataset_folder, configs, seed, get_data=False,
                        double_data=False, k_cross=False, k_num=5):
    """
    This function is used in imgtotfrecord
    :param tfrecord_name:   String, Name of tfrecord output file
    :param dataset_folder:  String, Folder directory of input data [Assume only data is in this folder]
    :param configs:         Dictionary, containing numdeg, train_eval_ratio, data_type
    :param seed:            Integer, to determine shuffling
    :param get_data:        Boolean, if true will return raw data instead of file name
    :param double_data:     Boolean, if true will do flip data augmentation
    :param k_cross:         Boolean, if true will use K-fold cross validation, else
    :param k_num:           Integer, parameter for KFold
    :return:                Train, Eval: Tuple of list[image address, label]. Also save some txt file
                            loss_weight: numpy array use for loss weight
    """
    numdeg = configs['numdeg']

    # Get image address, or image data
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

    # Create list of file names used in split_train_test (To remember which one is train/eval)
    if not k_cross:
        example_grouped_address = []
        for i in range(len(labels)):
            example_grouped_address.append(image_address[i * numdeg].split('.')[0])  # Only 0 degree

    # Convert name to data (only if request output to be value, not file name)
    if get_data:
        image_address_temp = []
        for addr in image_address:
            image_address_temp.append(np.load(addr))
        image_address = image_address_temp

    # Group up 4 images and label together first, shuffle
    grouped_address = []
    for i in range(len(labels)):
        grouped_address.append([image_address[i * numdeg:(i + 1) * numdeg], labels[i]])  # All degrees
    if not k_cross:
        # Packing data
        temp_address = []  # New temporary address packs augmented data together
        temp_example_address = []
        temp_name = {}
        for g_add, ex_g_add in zip(grouped_address, example_grouped_address):
            if os.path.basename(ex_g_add).split('_')[1] in temp_name:
                temp_address[temp_name[os.path.basename(ex_g_add).split('_')[1]]].append(g_add)
                temp_example_address[temp_name[os.path.basename(ex_g_add).split('_')[1]]].append(ex_g_add)
            else:
                temp_name[os.path.basename(ex_g_add).split('_')[1]] = len(temp_address)
                temp_address.append([g_add])
                temp_example_address.append([ex_g_add])

        # Zip, shuffle, unzip
        z = list(zip(temp_address, temp_example_address))
        # z = list(zip(grouped_address, example_grouped_address))
        random.Random(seed).shuffle(z)
        temp_address[:], temp_example_address[:] = zip(*z)
        # grouped_address[:], example_grouped_address[:] = zip(*z)

        # Unpack data
        grouped_address = [item for sublist in temp_address for item in sublist]
        example_grouped_address = [item.split('/')[-1] for sublist in temp_example_address for item in sublist]
    else:
        # Pack data
        temp_address = []  # New temporary address packs augmented data together
        temp_name = {}
        for g_add, ex_g_add in zip(grouped_address, example_grouped_address):
            if os.path.basename(ex_g_add).split('_')[1] in temp_name:
                temp_address[temp_name[os.path.basename(ex_g_add).split('_')[1]]].append(g_add)
            else:
                temp_name[os.path.basename(ex_g_add).split('_')[1]] = len(temp_address)
                temp_address.append([g_add])
        random.Random(seed).shuffle(grouped_address)
        # Unpack data
        grouped_address = [item for sublist in temp_address for item in sublist]

    # Calculate loss weight
    _, label = [list(e) for e in zip(*grouped_address)]

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

    if k_cross:  # If k_cross mode, output will be list
        train_address_temp, eval_address_temp = split_kfold(grouped_address, k_num, seed)
        train_address = []
        eval_address = []
        for i in range(k_num):
            single_train_address = train_address_temp[i]
            single_eval_address = eval_address_temp[i]
            if not get_data:  # Put in special format for writing tfrecord (pipeline)
                single_train_address = tuple(
                    [list(e) for e in zip(*single_train_address)])  # Convert to tuple of list[image address, label]

                single_eval_address = tuple(
                    [list(e) for e in zip(*single_eval_address)])  # Convert to tuple of list[image address, label]
                print(
                    "Train files: %d, Evaluate Files: %d" % (len(single_train_address[0]), len(single_eval_address[0])))
            else:
                print("Train files: %d, Evaluate Files: %d" % (len(single_train_address), len(single_eval_address)))
            train_address.append(single_train_address)
            eval_address.append(single_eval_address)

            # Save label distribution for weight balancing
            file_name = "./data/tfrecord/%s/%s_%s.json" % (tfrecord_name, tfrecord_name, i)
            with open(file_name, 'w') as filehandle:
                json.dump({"class_weight": class_weight}, filehandle, indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)

    else:
        train_address, eval_address = split_train_test(grouped_address, example_grouped_address,
                                                       tfrecord_name, configs, class_weight)
        if not get_data:  # Put in special format for writing tfrecord (pipeline)
            train_address = tuple(
                [list(e) for e in zip(*train_address)])  # Convert to tuple of list[image address, label]

            eval_address = tuple(
                [list(e) for e in zip(*eval_address)])  # Convert to tuple of list[image address, label]
            print("Train files: %d, Evaluate Files: %d" % (len(train_address[0]), len(eval_address[0])))
        else:
            print("Train files: %d, Evaluate Files: %d" % (len(train_address), len(eval_address)))

    return train_address, eval_address


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
        readCSV = csv.reader(csvFile, delimiter=',')
        for row in readCSV:
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
    Extension to read_file, specifically used to read csv file made from pre_processing.py
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
        readCSV = csv.reader(csvFile, delimiter=',')
        is_header = True
        for row in readCSV:
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
