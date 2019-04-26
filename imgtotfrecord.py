import tensorflow as tf
import numpy as np
import os
import warnings
from open_save_file import get_label, get_file_name, get_input_and_label

numdeg = 4  # Number of images on each example

configs = {'numdeg': numdeg,
           'train_eval_ratio': 0.8,
           'label_data': "Taper_Sum",
           'label_type': "median"
           }


# Run images from pre_processing.py into tfrecords
def serialize_image(example):
    feature = {'label': _int64_feature(example['label'])}
    for i in range(numdeg):
        feature['img' + str(i)] = _bytes_feature(example['img' + str(i)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example.SerializeToString()


def read_image(file_name, label):
    file_values = {'label': label}
    for i in range(numdeg):
        file_values['img' + str(i)] = tf.read_file(file_name[i])
    return file_values


# Run images from pre_processing.py into tfrecords
def serialize_coordinate(example):
    feature = {'label': _int64_feature(example['label'])}
    for i in range(numdeg):
        feature['img' + str(i)] = _float_feature(example['img' + str(i)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example.SerializeToString()


def read_coordinate(file_name, label):
    file_values = {'label': label}
    for i in range(numdeg):
        print(file_name[i])
        data = np.load(file_name[i])
        # data = tf.read_file(file_name[i])
        print(type(data))
        file_values['img' + str(i)] = data
    return file_values


'''
# Read and load image
def load_image(addr):
    img = cv2.imread(addr, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('IMAGE WINDOW',img)
    # cv2.waitKey()
    # print(np.shape(img))
    return img
'''


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def image_to_tfrecord(tfrecord_name, dataset_folder, csv_dir=None):
    grouped_train_address, grouped_eval_address, eval_score = get_input_and_label(tfrecord_name, dataset_folder, csv_dir, configs)
    # Start writing train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(grouped_train_address)
    train_dataset = train_dataset.map(read_image)  # Read file address, and get info as string

    it = train_dataset.make_one_shot_iterator()

    elem = it.get_next()

    # Start getting all info and zip to tfrecord
    tfrecord_train_name = os.path.join("./data/tfrecord", "%s_%s_%s_train.tfrecords" % (
        tfrecord_name, configs['label_data'], configs['label_type']))
    tfrecord_eval_name = os.path.join("./data/tfrecord", "%s_%s_%s_eval.tfrecords" % (
        tfrecord_name, configs['label_data'], configs['label_type']))
    eval_score_name = os.path.join("./data/tfrecord", "%s_%s_%s_score.npy" % (
        tfrecord_name, configs['label_data'], configs['label_type']))
    np.save(eval_score_name, np.asarray(eval_score))

    with tf.Session() as sess:
        writer = tf.python_io.TFRecordWriter(tfrecord_train_name)
        while True:
            try:
                elem_result = serialize_image(sess.run(elem))

                writer.write(elem_result)
            except tf.errors.OutOfRangeError:
                break
        writer.close()

    eval_dataset = tf.data.Dataset.from_tensor_slices(grouped_eval_address)
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


# tfrecord_name : Name of .tfrecord file to be created
# dataset_folder : Folder of the data (Not include label)
# csv_dir : Folder of label data (If not specified, will use the default directory)
def coordinate_to_tfrecord(tfrecord_name, dataset_folder, csv_dir=None):
    grouped_train_address, grouped_eval_address = get_input_and_label(tfrecord_name, dataset_folder, csv_dir, configs)
    # Start writing train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(grouped_train_address)
    train_dataset = train_dataset.map(read_coordinate)  # Read file address, and get info as string

    it = train_dataset.make_one_shot_iterator()

    elem = it.get_next()

    # Start getting all info and zip to tfrecord
    tfrecord_train_name = os.path.join("./data/tfrecord", "%s_%s_%s_coor_train.tfrecords" % (
        tfrecord_name, configs['label_data'], configs['label_type']))
    tfrecord_eval_name = os.path.join("./data/tfrecord", "%s_%s_%s_coor_eval.tfrecords" % (
        tfrecord_name, configs['label_data'], configs['label_type']))

    with tf.Session() as sess:
        writer = tf.python_io.TFRecordWriter(tfrecord_train_name)
        while True:
            try:
                elem_result = serialize_coordinate(sess.run(elem))

                writer.write(elem_result)
            except tf.errors.OutOfRangeError:
                break
        writer.close()

    eval_dataset = tf.data.Dataset.from_tensor_slices(grouped_eval_address)
    eval_dataset = eval_dataset.map(read_coordinate)

    it = eval_dataset.make_one_shot_iterator()

    elem = it.get_next()

    with tf.Session() as sess:
        writer = tf.python_io.TFRecordWriter(tfrecord_eval_name)
        while True:
            try:
                elem_result = serialize_coordinate(sess.run(elem))
                # print(elem_result)
                writer.write(elem_result)
            except tf.errors.OutOfRangeError:
                break
        writer.close()
    print("TFrecords created: %s, %s" % (tfrecord_train_name, tfrecord_eval_name))


if __name__ == '__main__':
    get_image = True
    # Select type of label to use
    label_data = ["Occ_Sum", "Taper_Sum", "Occ_L", "Occ_F", "Occ_B", "BL", "MD", "Taper_Sum"]
    label_type = ["average", "median"]
    configs['numdeg'] = 4
    configs['train_eval_ratio'] = 0.8
    # configs['label_data'] = "Taper_Sum"
    configs['label_type'] = "median"

    label_datas = ["Taper_Sum", "BL", "MD", "Occ_Sum"]  # Too lazy to do all of these one at a time
    for i in label_datas:
        configs['label_data'] = i
        print("Use label from %s category '%s' with (%s) train:eval ratio" % (
            configs['label_data'], configs['label_type'], configs['train_eval_ratio']))
        # File name will be [tfrecord_name]_train_Taper_sum_median
        tfrecord_file_name = "preparation_361"
        # tfrecord_name = "original_preparation_data"
        # csv_name = "../global_data/Ground Truth Score_50.csv"
        # Directory of image

        if get_image:
            dataset_folder_dir = "./data/cross_section"
            image_to_tfrecord(tfrecord_file_name, dataset_folder_dir)
        else:
            dataset_folder_dir = "./data/coordinates"
            coordinate_to_tfrecord(tfrecord_file_name, dataset_folder_dir)
    print("Complete")
