import tensorflow as tf
import numpy as np
import random
import os
import datetime
import json
from utils.open_save_file import get_input_and_label, read_file, save_file

numdeg = 4  # Number of images on each example

# Default value
configs = {'train_eval_ratio': 0.8,
           'data_type': 'BL_median',
           }


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# Run images from stl_to_image.py into tfrecords
def serialize_image(example):
    global numdeg
    # Record name and image data
    feature = {}
    for i in range(numdeg):
        feature['img' + str(i)] = _bytes_feature(example['img' + str(i)])
    # Record all available label, mostly int except for 'name'
    for d in configs['data_type']:
        if d == "name":
            feature[d] = _bytes_feature(bytes(example[d], encoding='utf8'))
        else:
            feature[d] = _int64_feature(example[d])

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example.SerializeToString()


def read_image(file_name, label):
    global numdeg
    file_values = label
    for i in range(numdeg):
        file_values['img' + str(i)] = tf.read_file(file_name[i])
    return file_values


def image_to_tfrecord(tfrecord_name, dataset_folder, csv_dir=None, k_fold=None):
    """

    :param tfrecord_name:   String, Name of .tfrecord output file
    :param dataset_folder:  Folder of the input data  (Not include label)
    :param csv_dir:         Folder of label data (If not specified, will use the default directory)
    :param k_fold:          Boolean, Option to save tfrecord for k_fold usage
    :param k_num:           int, If k_fold is true, select amount of k-fold
    :return: save 3 files: train.tfrecord, eval.tfrecord, config (as .json)
    """
    # Create new directory if not created, get all info and zip to tfrecord
    if tfrecord_name.split('.')[-1] == "tfrecords":  # Remove extension if exist
        tfrecord_name = os.path.splitext(os.path.basename(tfrecord_name))[0]
    tfrecord_dir = os.path.join("../data/tfrecord", tfrecord_name)
    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)

    # Read config file to get amount of degree and augmentation
    config_data = read_file(os.path.join(dataset_folder, "config.txt"))
    configs['numdeg'], configs['num_augment'] = config_data[0], config_data[1]

    # Get file name from dataset_folder
    grouped_train_address, grouped_eval_address = get_input_and_label(tfrecord_name, dataset_folder, csv_dir, configs,
                                                                      get_data=False, k_fold=k_fold)

    # if not k_fold:
    #     k_num = 1
    #     grouped_train_address = [grouped_train_address]
    #     grouped_eval_address = [grouped_eval_address]
    if k_fold is None:
        k_fold = 1
    for i in range(k_fold):
        train_address = grouped_train_address[i]
        eval_address = grouped_eval_address[i]

        # Start writing train dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(train_address)
        train_dataset = train_dataset.map(read_image)  # Read file address, and get info as string

        it = train_dataset.make_one_shot_iterator()

        elem = it.get_next()

        tfrecord_train_name = os.path.join(tfrecord_dir, "%s_%s_train.tfrecords" % (tfrecord_name, i))
        tfrecord_eval_name = os.path.join(tfrecord_dir, "%s_%s_eval.tfrecords" % (tfrecord_name, i))

        with tf.Session() as sess:
            writer = tf.python_io.TFRecordWriter(tfrecord_train_name)
            while True:
                try:
                    elem_result = serialize_image(sess.run(elem))

                    writer.write(elem_result)
                except tf.errors.OutOfRangeError:
                    break
            writer.close()

        eval_dataset = tf.data.Dataset.from_tensor_slices(eval_address)
        eval_dataset = eval_dataset.map(read_image)

        it = eval_dataset.make_one_shot_iterator()

        elem = it.get_next()

        with tf.Session() as sess:
            writer = tf.python_io.TFRecordWriter(tfrecord_eval_name)
            while True:
                try:
                    elem_result = serialize_image(sess.run(elem))
                    # print(elem_result)
                    writer.write(elem_result)
                except tf.errors.OutOfRangeError:
                    break
            writer.close()
        print("TFrecords created: %s, %s" % (tfrecord_train_name, tfrecord_eval_name))


# Run images from stl_to_image.py into tfrecords, not using right now
def serialize_coordinate(example):
    feature = {}
    for i in range(numdeg):
        feature['img' + str(i)] = _float_feature(example['img' + str(i)])
    # Record all available label
    for d in configs['data_type']:
        if d == "name":
            feature[d] = _bytes_feature(bytes(example[d], encoding='utf8'))
        else:
            feature[d] = _int64_feature(example[d])

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example.SerializeToString()


def read_coordinate(file_name, label):
    file_values = label
    # file_values = {'label': label}
    for i in range(numdeg):
        file_values['img' + str(i)] = (file_name[i] * 10000).tostring()
    return file_values


def write_tfrecord(all_data, file_dir, degree, coordinate_length):
    """
    Create tfrecord file by adding each datapoint sequentially
    :param all_data: List of all data to save as tfrecords
    :param file_dir: Directory and file name to save (End with .tfrecords)
    :param degree: List of degree used
    :param coordinate_length: Number of points on each file (Similar to image size but in 1D)
    :return:
    """
    with tf.python_io.TFRecordWriter(file_dir) as writer:
        for data in all_data:
            default_key = list(data.keys())[0]  # Use for label since score should be the same
            # Add general info
            feature = {'degree': _int64_feature(degree), 'length': _int64_feature(coordinate_length)}
            # Add labels
            for d in configs['data_type']:
                if d == "name":  # Save name as bytes
                    feature[d] = _bytes_feature(bytes(data[default_key][1][d], encoding='utf8'))
                else:  # Save other score as int
                    feature[d] = _int64_feature(data[default_key][1][d])
            # Add data
            for dname in data.keys():  # Flatten each dataset (in case of more than one)
                for n in range(degree):  # Flatten each degree
                    for j in range(2):  # Flatten x,y axis
                        val = data[dname][0][n][:, j].reshape(-1)
                        if np.shape(val)[0] != coordinate_length:
                            print("Error", data[dname][1]["name"])
                            print(np.shape(val)[0])
                        # Save image as float list
                        feature['%s_%s_%s' % (dname, n, j)] = tf.train.Feature(float_list=tf.train.FloatList(value=val))
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Write TFrecord file
            writer.write(example.SerializeToString())


def coordinate_to_tfrecord(tfrecord_name, dataset_folders, k_fold=None):
    """
    tfrecord_name   : Name of .tfrecord output file
    dataset_folder  : Folder of the input data  (Not include label)
    k_fold          : Integer, amount of K-fold cross validation. Can be None to disable
    save 4 files: train.tfrecord, eval.tfrecord, .txt (Save from another file)
    """

    # Create new directory if not created, get all info and zip to tfrecord
    if tfrecord_name.split('.')[-1] == "tfrecords":  # Remove extension if exist
        tfrecord_name = tfrecord_name[0:-10]
    tfrecord_dir = os.path.join("../data/tfrecord", tfrecord_name)
    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)

    if isinstance(dataset_folders, str):
        dataset_folders = {'img': dataset_folders}

    if k_fold is None:
        dataset_list = [dict()]
    else:
        dataset_list = [dict() for i in range(k_fold)]
    seed = random.randint(0, 1000000)  # So that the result is always the same

    # Convert data into list (kfolds) of dictionary (left/right/...) of list of (train,eval) data
    for dataset_name, dataset_folder in dataset_folders.items():
        # Get amount of degree and augmentation
        config_data = read_file(os.path.join(dataset_folder, "config.txt"))
        configs['numdeg'] = int(config_data[0][0])  # Get data from config.txt
        configs['num_augment'] = int(config_data[1][0])

        # Get data from dataset_folder
        grouped_train_data, grouped_eval_data = get_input_and_label(tfrecord_name, dataset_folder,
                                                                    configs, seed, get_data=True, k_fold=k_fold)
        print(grouped_train_data[0][0])
        # if k_fold is None:
        #     k_fold = 1
        #     grouped_train_data = [grouped_train_data]
        #     grouped_eval_data = [grouped_eval_data]
        for i, (td, ed) in enumerate(zip(grouped_train_data, grouped_eval_data)):
            dataset_list[i][dataset_name] = [td, ed]

    if k_fold is None:
        k_fold = 1

    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    for i in range(k_fold):
        tfrecord_train_name = os.path.join(tfrecord_dir, "%s_%s_train.tfrecords" % (tfrecord_name, i))
        tfrecord_eval_name = os.path.join(tfrecord_dir, "%s_%s_eval.tfrecords" % (tfrecord_name, i))
        dataset_dict = dataset_list[i]

        coordinate_length = len(list(dataset_dict.values())[0][0][0][0][0])
        degree = len(list(dataset_dict.values())[0][0][0][0])
        dataset_name = dataset_dict.keys()

        # Rearrange order of data
        train_data = []
        eval_data = []
        data_amount = len(list(dataset_dict.values())[0][0])  # Sample amount of train data
        for j in range(data_amount):
            td = {}
            for n in dataset_name:
                td[n] = dataset_dict[n][0][j]
            train_data.append(td)
        data_amount = len(list(dataset_dict.values())[0][1])  # Sample amount of eval data
        for j in range(data_amount):
            ed = {}
            for n in dataset_name:
                ed[n] = dataset_dict[n][1][j]
            eval_data.append(ed)

        write_tfrecord(train_data, tfrecord_train_name, degree, coordinate_length)
        write_tfrecord(eval_data, tfrecord_eval_name, degree, coordinate_length)

        # Update info in json file
        with open("../data/tfrecord/%s/%s_%s.json" % (tfrecord_name, tfrecord_name, i)) as filehandle:
            data_loaded = json.load(filehandle)
        data_loaded["data_degree"] = degree
        data_loaded["data_length"] = coordinate_length
        data_loaded["dataset_name"] = list(dataset_folders.keys())
        data_loaded["timestamp"] = time_stamp
        with open("../data/tfrecord/%s/%s_%s.json" % (tfrecord_name, tfrecord_name, i), 'w') as filehandle:
            json.dump(data_loaded, filehandle, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
        print("TFrecords created: %s, %s" % (tfrecord_train_name, tfrecord_eval_name))


if __name__ == '__main__':
    data_mode = "coordinate"  # image or coordinate
    # Select type of label to use
    label_data = ["name", "Occ_B_median", "Occ_F_median", "Occ_L_median", "BL_median", "MD_median",
                  "Integrity_median", "Width_median", "Surface_median", "Sharpness_median"]
    configs['train_eval_ratio'] = 0.8
    k_fold = None

    configs['data_type'] = label_data
    print("Use label from %s with (%s) train:eval ratio" % (
        configs['data_type'], configs['train_eval_ratio']))

    if data_mode == "image":
        image_to_tfrecord(tfrecord_name="preparation_img_test", dataset_folder="../data/cross_section",
                          k_fold=k_fold)
    elif data_mode == "coordinate":
        coordinate_to_tfrecord(tfrecord_name="debug_coor_no_augment",
                               dataset_folders="../data/coordinate_no_augment_debug", k_fold=k_fold)
        # coordinate_to_tfrecord(tfrecord_name="coor_right",
        #                        dataset_folders="../data/segment_14/right_point", k_fold=k_fold)
        # coordinate_to_tfrecord(tfrecord_name="coor_left_right", dataset_folders={'right': "../data/segment_14/right_point",
        #                                                                     'left': "../data/segment_14/left_point"},
        #                        k_fold=k_fold)
    else:
        raise ValueError("Wrong data_mode")
    print("Complete")
