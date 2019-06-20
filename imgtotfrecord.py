import tensorflow as tf
import numpy as np
import os
from open_save_file import get_input_and_label

numdeg = 4  # Number of images on each example

configs = {'numdeg': numdeg,
           'train_eval_ratio': 0.8,
           'label_data': "Taper_Sum",
           'label_type': "median"
           }


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


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


'''
# Read and load image
def load_image(addr):
    img = cv2.imread(addr, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('IMAGE WINDOW',img)
    # cv2.waitKey()
    # print(np.shape(img))
    return img
'''


def image_to_tfrecord(tfrecord_name, dataset_folder, csv_dir=None):
    # Create new directory if not created, get all info and zip to tfrecord
    tfrecord_dir = os.path.join("./data/tfrecord", tfrecord_name)
    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)

    #Get file name from dataset_folder
    grouped_train_address, grouped_eval_address = get_input_and_label(tfrecord_name, dataset_folder, csv_dir, configs)
    # Start writing train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(grouped_train_address)
    train_dataset = train_dataset.map(read_image)  # Read file address, and get info as string

    it = train_dataset.make_one_shot_iterator()

    elem = it.get_next()



    tfrecord_train_name = os.path.join(tfrecord_dir, "%s_%s_%s_train.tfrecords" % (
        tfrecord_name, configs['label_data'], configs['label_type']))
    tfrecord_eval_name = os.path.join(tfrecord_dir, "%s_%s_%s_eval.tfrecords" % (
        tfrecord_name, configs['label_data'], configs['label_type']))
    # eval_score_name = os.path.join("./data/tfrecord", "%s_%s_%s_score.npy" % (
    #     tfrecord_name, configs['label_data'], configs['label_type']))
    # np.save(eval_score_name, np.asarray(eval_score))

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
        file_values['img' + str(i)] = (file_name[i]*10000).tostring()
    return file_values


def coordinate_to_tfrecord(tfrecord_name, dataset_folder, csv_dir=None):
    """
    tfrecord_name   : Name of .tfrecord output file
    dataset_folder  : Folder of the input data  (Not include label)
    csv_dir         : Folder of label data (If not specified, will use the default directory)
    """
    # Create new directory if not created, get all info and zip to tfrecord
    tfrecord_dir = os.path.join("./data/tfrecord", tfrecord_name)
    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)

    # Get data from dataset_folder
    grouped_train_address, grouped_eval_address = get_input_and_label(tfrecord_name, dataset_folder,
                                                                      csv_dir, configs, get_data=True)

    tfrecord_train_name = os.path.join(tfrecord_dir, "%s_%s_%s_train.tfrecords" % (
        tfrecord_name, configs['label_data'], configs['label_type']))
    tfrecord_eval_name = os.path.join(tfrecord_dir, "%s_%s_%s_eval.tfrecords" % (
        tfrecord_name, configs['label_data'], configs['label_type']))

    coordinate_length = len(grouped_train_address[0][0][0])
    degree = len(grouped_train_address[0][0])

    with tf.python_io.TFRecordWriter(tfrecord_train_name) as writer:
        for train_data in grouped_train_address:
            feature = {'label': _int64_feature(train_data[1]),
                       'degree': _int64_feature(degree),
                       'length':  _int64_feature(coordinate_length)}
            for i in range(numdeg):
                for j in range(2):
                    val = train_data[0][i][:,j].reshape(-1)
                    feature['img_%s_%s' % (i,j)] =tf.train.Feature(float_list=tf.train.FloatList(value=val))
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Write TFrecord file
            writer.write(example.SerializeToString())

    with tf.python_io.TFRecordWriter(tfrecord_eval_name) as writer:
        for train_data in grouped_eval_address:
            feature = {'label': _int64_feature(train_data[1]),
                       'degree': _int64_feature(degree),
                       'length':  _int64_feature(coordinate_length)}
            for i in range(numdeg):
                for j in range(2):
                    val = train_data[0][i][:, j].reshape(-1)
                    feature['img_%s_%s' % (i, j)] = tf.train.Feature(float_list=tf.train.FloatList(value=val))
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # print(example)

            # Write TFrecord file
            writer.write(example.SerializeToString())
    print("TFrecords created: %s, %s" % (tfrecord_train_name, tfrecord_eval_name))


if __name__ == '__main__':
    get_image = False
    # Select type of label to use
    label_data = ["Occ_Sum", "Taper_Sum", "Occ_L", "Occ_F", "Occ_B", "BL", "MD", "Taper_Sum"]
    label_type = ["average", "median"]
    configs['numdeg'] = 4
    configs['train_eval_ratio'] = 0.8
    # configs['label_data'] = "Taper_Sum"
    configs['label_type'] = "median"

    label_datas = ["Taper_Sum", "BL", "MD", "Occ_Sum", "Occ_L", "Occ_F", "Occ_B"]  # Too lazy to do all of these one at a time

    for label_data_index in label_datas:
        configs['label_data'] = label_data_index
        print("Use label from %s category '%s' with (%s) train:eval ratio" % (
            configs['label_data'], configs['label_type'], configs['train_eval_ratio']))
        # File name will be [tfrecord_name]_train_Taper_sum_median

        # tfrecord_name = "original_preparation_data"
        # csv_name = "../global_data/Ground Truth Score_50.csv"
        # Directory of image

        if get_image:
            image_to_tfrecord(tfrecord_name="preparation_361", dataset_folder="./data/cross_section")
        else:
            coordinate_to_tfrecord(tfrecord_name="preparation_coor_300", dataset_folder="./data/coordinate_300_point")
    print("Complete")
