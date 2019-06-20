# import sys
import os
# import glob
import csv
# %matplotlib inline
# import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import collections
from sklearn.model_selection import KFold

mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from random import shuffle

v = '2.3.1'  # Add getFilename function, get absolute path now
# 1.8 Works with pycharm
# 1.9 Edit Name of dataname and datatype
# 2.1 Add One_hot encoding
# 2.2 Add normalize option
# 2.3 Add new version of get_file_name
# Add double label for augmentation
# Note in case of directly using from jupyter, change data/ to ../data
print("ImportData2D.py version: " + str(v))


# Get sorted List of filenames: search for all file within folder with specific file name
# If not required specific file name, type None
# folder_dir is used for checking missing files
# We specifically ignore 'error_file.txt'
def get_file_name(folder_name='../global_data/', file_name='PreparationScan.stl'):
    print("get_file_name: Import from %s, searching for file name with %s" % (os.path.abspath(folder_name), file_name))
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


# Plot the list of coordinates and save it as PNG image
# Input     coor_list        -> List of {List of numpy coordinates <- get from stlSlicer}
#           out_directory    -> String, Directory to save output
#           file_header_name -> String, header of name of the file, follow by image_num
#           image_num        -> List of name of the image
#           augment_number   -> Since augmentation has same image_number, we use this to differentiate
#           degree           -> List of angles used in, add the angle in file name as well
#           file_type        -> [Optional], such as png,jpeg,...
# Output    none             -> Only save as output outside
def save_plot(coor_list, out_directory, file_header_name, image_name, augment_number, degree, file_type="png"):
    if len(coor_list) != len(image_name):
        raise ValueError("save_plot: number of image(%s) is not equal to number of image_name(%s)"
                         % (len(coor_list), len(image_name)))
    out_directory = os.path.abspath(out_directory)
    if len(degree) != len(coor_list[0]):
        print("# of Degree expected: %d" % len(degree))
        print("# of Degree found: %d" % len(coor_list[0]))
        raise Exception('Number of degree specified is not equals to coordinate ')

    fig = plt.figure(figsize=(6, 4))

    for i in range(len(coor_list)):
        for d in range(len(degree)):
            coor = coor_list[i][d]
            # plt.plot(coor[:, 0], coor[:, 1], color='black', linewidth=1)
            # plt.axis('off')
            # Name with some additional data
            fullname = "%s_%s_%s_%d.%s" % (file_header_name, image_name[i], augment_number, degree[d], file_type)
            output_name = os.path.join(out_directory, fullname)

            # plt.savefig(output_name, bbox_inches='tight')
            # plt.clf()

            ax = fig.add_subplot(111)
            ax.plot(coor[:, 0], coor[:, 1], color='black', linewidth=1)
            ax.axis('off')
            fig.savefig(output_name, dpi=60)
            fig.clf()
    print("Finished plotting for %d images with %d rotations at %s" % (len(coor_list), len(degree), out_directory))


def split_train_test(grouped_address, example_grouped_address, tfrecord_name, configs):
    # Split into train and test set
    train_address = []
    eval_address = []

    train_amount = int(configs['train_eval_ratio'] * len(grouped_address))  # Calculate amount of training data

    # Open file and read the content in a list
    file_name = "./data/tfrecord/%s/%s_%s_%s.txt" % (
        tfrecord_name, tfrecord_name, configs['label_data'], configs['label_type'])
    if os.path.isfile(file_name):  # Check if file exist
        with open(file_name) as f:
            filehandle = f.read().splitlines()
        is_training = 0  # 0 = checking for distribution, 1 = checking for train, 2 = checking for eval
        if not (filehandle[0] == 'distribution'):
            print(filehandle[0])
            raise KeyError("File does not have correct format, need 'train' and 'eval' keyword within file")
        for line in filehandle[1:]:
            if is_training == 0:
                if line == 'train':
                    is_training = 1
            elif is_training == 1:
                if line == 'eval':
                    is_training = 2
                else:
                    # check if it exist in grouped exist, if found, put in train_address
                    for i, name in enumerate(example_grouped_address):
                        if line in name:
                            train_address.append(grouped_address[i])
                            grouped_address.remove(grouped_address[i])
                            example_grouped_address.remove(example_grouped_address[i])
                            break
            else:
                for i, name in enumerate(example_grouped_address):
                    if line in name:
                        eval_address.append(grouped_address[i])
                        grouped_address.remove(grouped_address[i])
                        example_grouped_address.remove(example_grouped_address[i])
                        break

        print("Use %s train examples, %s eval examples from previous tfrecords as training" % (
            len(train_address), len(eval_address)))

    # Split training and test (Split 80:20)
    train_amount = train_amount - len(train_address)
    if train_amount < 0:
        train_amount = 0
        print("imgtotfrecord: amount of training is not correct, might want to check")
    train_address.extend(grouped_address[0:train_amount])
    eval_address.extend(grouped_address[train_amount:])
    return train_address, eval_address


def split_kfold(grouped_address, k_num):
    kfold = KFold(k_num, shuffle=True)
    train_address = []
    eval_address = []
    for train_indices, test_indices in kfold.split(grouped_address):
        train_address_fold = []
        test_address_fold = []
        for train_indice in train_indices:
            train_address_fold.append(grouped_address[train_indice])
        for test_indice in test_indices:
            test_address_fold.append(grouped_address[test_indice])
        train_address.append(train_address_fold)
        eval_address.append(test_address_fold)
    return train_address, eval_address


def get_input_and_label(tfrecord_name, dataset_folder, csv_dir, configs, get_data=False,
                        double_data=True, k_cross=False, k_num=5):
    """
    This function is used in imgtotfrecord
    :param tfrecord_name:   String, Name of tfrecord output file
    :param dataset_folder:  String, Folder directory of input data [Assume only data is in this folder]
    :param csv_dir:         String, directory of label file (.csv). Can be None and use default directory
    :param configs:         Dictionary, containing numdeg, train_eval_ratio, labels_data, labels_type
    :param get_data:        Boolean, if true will return raw data instead of file name
    :param double_data:     Boolean, if true will do flip data augmentation
    :param k_cross:         Boolean, if true will use K-fold cross validation, else
    :param k_num:           Integer, parameter for KFold
    :return:
    """
    numdeg = configs['numdeg']

    # Get image address, or image data
    image_address, _ = get_file_name(folder_name=dataset_folder, file_name=None)
    # Convert name to data (only if request output to be value, not file name)
    if get_data:
        image_address_temp = []
        for addr in image_address:
            image_address_temp.append(np.load(addr))
        image_address = image_address_temp

    # Get label and label name[Not used right now]
    if csv_dir is None:
        labels, label_name = get_label(configs['label_data'], configs['label_type'], double_data=double_data,
                                       one_hotted=False, normalized=False)
    else:
        labels, label_name = get_label(configs['label_data'], configs['label_type'], double_data=double_data,
                                       one_hotted=False, normalized=False, file_dir=csv_dir)

    label_count = collections.Counter(labels)  # Use for frequency count

    # Check list of name that has error (from preprocessing.py), remove it from label
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
            if double_data:
                labels.pop(index * 2)  # Do it again if we double the data
        except ValueError:
            pass

    if len(image_address) / len(labels) != numdeg:
        print(image_address)
        raise Exception(
            '# of images and labels is not compatible: %d images, %d labels. Expected # of images to be 4 times of label' % (
                len(image_address), len(labels)))

    # Group up 4 images and label together first, shuffle
    grouped_address = []
    example_grouped_address = []  # A 0 degree filename used when adding more example
    for i in range(len(labels)):
        grouped_address.append([image_address[i * numdeg:(i + 1) * numdeg], labels[i]])  # All degrees
        example_grouped_address.append(image_address[i * numdeg].split('.')[0])  # Only 0 degree

    # Zip, shuffle, unzip
    z = list(zip(grouped_address, example_grouped_address))
    shuffle(z)
    grouped_address[:], example_grouped_address[:] = zip(*z)

    if k_cross:
        pass
    else:
        train_address, eval_address = split_train_test(grouped_address, example_grouped_address, tfrecord_name, configs)

    grouped_train_address = tuple(
        [list(e) for e in zip(*train_address)])  # Convert to tuple of list[image address, label]

    grouped_eval_address = tuple(
        [list(e) for e in zip(*eval_address)])  # Convert to tuple of list[image address, label]

    # print(grouped_eval_address)
    print("Train files: %d, Evaluate Files: %d" % (len(grouped_train_address[0]), len(grouped_eval_address[0])))

    # Save names of files of train address
    file_name = "./data/tfrecord/%s/%s_%s_%s.txt" % (
        tfrecord_name, tfrecord_name, configs['label_data'], configs['label_type'])
    with open(file_name, 'w') as filehandle:
        # Header with 'eval'
        filehandle.write('distribution\n')
        for score, freq in label_count.items():
            filehandle.write('%s_%s\n' % (score, freq))

        # Header with 'train'
        filehandle.write('train\n')
        for listitem in train_address:
            # For each filename, select 0 degree angle and remove _0.png
            if listitem[0][0].count('.') != 1:
                raise KeyError("train_address has string '.' more than once in name, please check")
            new_list_item = listitem[0][0].split('.')[0]
            filehandle.write('%s\n' % new_list_item)

        # Header with 'eval'
        filehandle.write('eval\n')
        for listitem in eval_address:
            # For each filename, select 0 degree angle and remove _0.png
            if listitem[0][0].count('.') != 1:
                raise KeyError("eval_address has string '.' more than once in name, please check")
            new_list_item = listitem[0][0].split('.')[0]
            filehandle.write('%s\n' % new_list_item)
    '''
    # Save names of files of eval address
    file_name = './data/' + tfrecord_name + '_eval_address.txt'
    with open(file_name, 'w') as filehandle:
        for listitem in eval_address:
            new_list_item = listitem[0][0].replace('_0.png', '')
            filehandle.write('%s\n' % new_list_item)
    '''
    return grouped_train_address, grouped_eval_address


def read_file(csv_dir, header=False):
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


# one_row = true means that all data will be written in one row
def save_file(csv_dir, all_data, field_name=None, write_mode='w', one_row=False, create_folder=False):
    # Create a folder if not exist
    if create_folder:
        direct = os.path.dirname(csv_dir)
        if not os.path.exists(direct):
            os.makedirs(direct)
    with open(csv_dir, write_mode) as csvFile:
        # Check if there is fieldname, if so, create first
        if field_name is None:
            writer = csv.writer(csvFile)
        else:
            writer = csv.DictWriter(csvFile, fieldnames=field_name)
            writer.writeheader()
        # If one_row is true -> use when data is list of one row, want to write in single row
        # else it will be split in to multiple row
        if not one_row:
            for data in all_data:
                writer.writerow([data])
        else:
            writer.writerow(all_data)


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
