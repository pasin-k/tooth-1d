import numpy as np
import os
import csv


# Get sorted List of filenames: search for all file within folder with specific file name
# If not required specific file name, type None
# folder_dir is used for checking missing files
# We specifically ignore 'error_file.txt'
def get_file_name(folder_name='../global_data/', file_name='PreparationScan.stl'):
    file_dir = list()
    folder_dir = list()
    for root, dirs, files in os.walk(folder_name):
        for filename in files:
            if file_name is None:  # Add everything if no file_name specified
                if filename == 'error_file.txt':
                    pass
                else:
                    file_dir.append(os.path.abspath(os.path.join(root, filename)))
            else:
                if filename.split('/')[-1] == file_name:
                    file_dir.append(os.path.abspath(os.path.join(root, filename)))
                    folder_dir.append(root.split('/')[-1].split('-')[0])
    file_dir.sort()
    folder_dir.sort()
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


# double: Duplicate score twice for augmentation
# one_hotted: False-> Output will be continuous, True-> Output will be vector with 1 on the correct score
# normalized: True will give result as maximum of 1, False will give raw value
# output labels_name is only for checking missing files
def get_label(dataname, datatype, double_data=True, one_hotted=False, normalized=False,
              file_dir='../global_data/Ground Truth Score_new.csv'):
    label_name = {"Occ_B": 0, "Occ_F": 3, "Occ_L": 6, "Occ_Sum": 9,
                  "BL": 12, "MD": 15, "Taper_Sum": 18}
    label_max_score = {"Occ_B": 5, "Occ_F": 5, "Occ_L": 5, "Occ_Sum": 15,
                       "BL": 5, "MD": 5, "Taper_Sum": 10}
    stat_type = {"average": 1, "median": 2}

    try:
        data_column = label_name[dataname]
    except:
        raise Exception(
            "Wrong dataname, Type as %s, Valid name: (\"Occ_B\",\"Occ_F\",\"Occ_L\","
            "\"Occ_sum\",\"BL\",\"MD\",\"Taper_Sum\")" % dataname)
    try:
        if one_hotted & (datatype == 1):
            datatype = 2
            print("Note: One-hot mode only supported median")
        label_column = data_column
        avg_column = data_column + 1
        data_column = data_column + stat_type[datatype]  # Shift the interested column by one or two, depends on type
    except:
        raise Exception("Wrong datatype, Type as %s, Valid name: (\"average\",\"median\")" % datatype)

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

    # If consider median data on anything except Taper_Sum/Occ_sum and does not normalized
    if (datatype is "median" and (not normalized) and dataname is not "Occ_Sum" and dataname is not "Taper_Sum"):
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
        return labels_data, labels_name


def get_coordinates(dataset_folder, csv_dir):
    '''
    :param dataset_folder:  Directory of data
    :param csv_dir:         Directory of label (Has default value as ../global_data/Ground Truth Score_new.csv
    :return:                image_data: list of example of data. Inside contain 4 lists (cross-section) of numpy array.
                            labels: list of labels
    '''

    numdeg = 4
    image_address, _ = get_file_name(folder_name=dataset_folder, file_name=None)
    image_data_temp = [np.loadtxt(addr, delimiter=',') for addr in image_address]
    assert len(image_data_temp) % 4 == 0, "Data is not compatible, possible missing data"
    image_data = [image_data_temp[4 * i:4 * i + 4] for i in range(int(len(image_data_temp) / 4))]
    # Get label and label name[Not used right now]
    if csv_dir is None:
        labels, label_name = get_label(dataname='BL', datatype="median", double_data=True,
                                       one_hotted=False, normalized=False)
    else:
        labels, label_name = get_label(dataname='BL', datatype="median", double_data=True,
                                       one_hotted=False, normalized=False, file_dir=csv_dir)

    # Some data has defect so we need to remove label of those defects
    error_file_names = []
    with open(dataset_folder + '/error_file.txt', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            current_name = line[:-1]
            # add item to the list
            error_file_names.append(current_name)
    for name in error_file_names:
        try:
            index = label_name.index(name)
            label_name.pop(index)
            labels.pop(index * 2)
            labels.pop(index * 2)  # Do it again if we double the data
        except ValueError:
            pass
    assert len(image_data) == len(labels), "Size of data and label is not compatible: %s, %s" % (
    len(image_data), len(labels))
    return image_data, labels


X, Y = get_coordinates('./data/coordinate_300_point', None)

print(len(X))
print(type(X[0][0]))
print(Y)
