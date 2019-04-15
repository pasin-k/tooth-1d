# import sys
import importlib
import scipy.io
import os
# import glob
import csv
import random
import imageio
# %matplotlib inline
# import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

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


# double: Duplicate score twice for augmentation
# one_hotted: False-> Output will be continuous, True-> Output will be vector with 1 on the correct score
# normalized: True will give result as maximum of 1, False will give raw value
# output labels_name is only for checking missing files
def get_label(dataname, datatype, double_data=True, one_hotted=False, normalized=True,
              file_dir='../global_data/Ground Truth Score_new.csv'):
    label_name = {"Occ_B": 0, "Occ_F": 3, "Occ_L": 6, "Occ_Sum": 9, "BL": 12, "MD": 15, "Taper_Sum": 18}
    label_max_score = {"Occ_B": 5, "Occ_F": 5, "Occ_L": 5, "Occ_Sum": 15, "BL": 5, "MD": 5, "Taper_Sum": 10}
    type = {"average": 1, "median": 2}

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
        data_column = data_column + type[datatype]  # Shift the interested column by one or two, depends on type
    except:
        raise Exception("Wrong datatype, Type as %s, Valid name: (\"average\",\"median\")" % datatype)

    max_score = label_max_score[dataname]
    labels_name = []
    labels_data = []
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
                    if one_hotted or (not normalized):  # Don't normalize if output is one hot encoding
                        normalized_value = int(val)  # Turn string to int
                    else:
                        normalized_value = float(val) / max_score  # Turn string to float
                    labels_name.append(label)
                    labels_data.append(normalized_value)

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
# Input     CoorList        -> List of {List of numpy coordinates <- get from stlSlicer}
#           outDirectory    -> String, Directory to save output
#           fileName        -> String, name of file to save
#           image_num       -> List of name of the image
#           Degree          -> List of angles used in
#           fileType        -> [Optional], such as png,jpeg,...
# Output    none            -> Only save as output outside
def save_plot(coor_list, out_directory, file_name, image_name, degree, file_type="png"):
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
            plt.plot(coor[:, 0], coor[:, 1], color='black', linewidth=1)
            plt.axis('off')
            # Name with some additional data
            fullname = "%s_%s_%d.%s" % (file_name, image_name[i], degree[d], file_type)
            output_name = os.path.join(out_directory, fullname)

            plt.savefig(output_name, bbox_inches='tight')
            plt.clf()
    print("Finished plotting for %d images with %d rotations at %s" % (len(coor_list), len(degree), out_directory))


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

