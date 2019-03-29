import tensorflow as tf
import numpy as np
import os
import sys
import time
import ast
import warnings
import cv2
from random import shuffle
from ImportData2D import get_label, get_file_name

numdeg = 4  # Number of images on each example


# Version 1.1.0
# Run images from pre_processing.py into tfrecords


# def serialize(img_addr,img_label,tffilename):
#     # open the TFRecords file
#     writer = tf.python_io.TFRecordWriter(tffilename)
#     for i in range(len(img_addr)):
#         # print how many images are saved every 100 images
#         if not i % 100:
#             print('Saving {} data: {}/{}'.format(tffilename, i, len(img_addr)))
#             sys.stdout.flush()
#         # Create a feature
#         feature = {}
#         feature['label'] = _float_feature(img_label[i])
#
#         for j in range(len(img_addr[0])):
#             # Load the image
#             img = load_image(img_addr[i][j])
#             feature['img'+str(j)] = _bytes_feature(tf.compat.as_bytes(img.tostring()))
#         # Create an example protocol buffer
#         example = tf.train.Example(features=tf.train.Features(feature=feature))
#
#         # Serialize to string and write on the file
#         writer.write(example.SerializeToString())
#
#     writer.close()
#     sys.stdout.flush()

def serialize(example):
    feature = {'label': _int64_feature(example['label'])}
    for i in range(numdeg):
        feature['img' + str(i)] = _bytes_feature(example['img' + str(i)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

    return tf_example.SerializeToString()


def read_file(file_name, label):
    file_values = {'label': label}
    for i in range(numdeg):
        file_values['img' + str(i)] = tf.io.read_file(file_name[i])
    return file_values


# Read and load image
def load_image(addr):
    img = cv2.imread(addr, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('IMAGE WINDOW',img)
    # cv2.waitKey()
    # print(np.shape(img))
    return img


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def run(tfrecord_name, dataset_folder):
    # Select ype of label to use
    label_data = ["Occ_Sum", "Taper_Sum"]  # Occ_sum: Max = 15, Taper_Sum: Max = 10
    label_type = ["average", "median"]
    label_data_num = 1  # Use Taper
    label_type_num = 1  # Use median
    train_eval_ratio = 0.8  # Ratio of training data
    print("Use label from %s of %s category with {%s} train:eval ratio" % (
        label_type[label_type_num], label_data[label_data_num], train_eval_ratio))

    # Start getting all info and zip to tfrecord
    train_name = "%s_%s_%s_train.tfrecords" % (tfrecord_name, label_data[label_data_num], label_type[label_type_num])
    eval_name = "%s_%s_%s_eval.tfrecords" % (tfrecord_name, label_data[label_data_num], label_type[label_type_num])
    train_name = os.path.join("./data", train_name)
    eval_name = os.path.join("./data", eval_name)

    image_address, _ = get_file_name(folder_name=dataset_folder, file_name=None)
    labels, label_name = get_label(label_data[label_data_num], label_type[label_type_num], double_data=True,
                                   one_hotted=False, normalized=False)

    # Check list of name that has error, remove it from label
    error_file_names = []
    with open('./data/cross_section/error_file.txt', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            current_name = line[:-1]
            # add item to the list
            error_file_names.append(current_name)
    for name in error_file_names:
        index = label_name.index(name)
        label_name.pop(index)
        labels.pop(index * 2)
        labels.pop(index * 2)  # Do it again if we double the data

    if len(image_address) / len(labels) != numdeg:
        print(image_address)
        raise Exception(
            '# of images and labels is not compatible: %d images, %d labels' % (len(image_address), len(labels)))

    # Group up 4 images and label together first, shuffle
    grouped_address = list()
    example_grouped_address = list()  # Use for checking the file name, in case of adding more example
    for i in range(len(labels)):
        grouped_address.append([image_address[i * numdeg:(i + 1) * numdeg], labels[i]])
        example_grouped_address.append(image_address[i*numdeg])
    z = list(zip(grouped_address, example_grouped_address))
    shuffle(z)
    grouped_address[:], example_grouped_address[:] = zip(*z)
    train_amount = int(train_eval_ratio * len(grouped_address))

    train_address = list()
    eval_address = list()

    # open file and read the content in a list
    file_name = './data/' + tfrecord_name + '_train_address.txt'
    if os.path.isfile(file_name):
        with open(file_name, 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                current_name = line[:-1]
                # check if it exist in grouped exist, if found, put in train_address
                for i, name in enumerate(example_grouped_address):
                    if current_name in name:
                        train_address.append(grouped_address[i])
                        grouped_address.remove(grouped_address[i])
                        example_grouped_address.remove(example_grouped_address[i])
                        break
        print("Use %s examples from previous tfrecords as training" % len(train_address))

    # open file and read the content in a list
    file_name = './data/' + tfrecord_name + '_eval_address.txt'
    if os.path.isfile(file_name):
        with open(file_name, 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                current_name = line[:-1]
                # check if it exist in grouped exist, if found, put in eval_address
                for i, name in enumerate(example_grouped_address):
                    if current_name in name:
                        eval_address.append(grouped_address[i])
                        grouped_address.remove(grouped_address[i])
                        example_grouped_address.remove(example_grouped_address[i])
                        break
        print("Use %s examples from previous tfrecords as evaluation" % len(eval_address))

    print(example_grouped_address)
    # Split training and test (Split 80:20)
    train_amount = train_amount - len(train_address)
    if train_amount < 0:
        train_amount = 0
        warnings.warn("imgtotfrecord: amount of training is not correct, might want to check")
    train_address.extend(grouped_address[0:train_amount])
    grouped_train_address = tuple(
        [list(e) for e in zip(*train_address)])  # Convert to tuple of list[image address, label]
    eval_address.extend(grouped_address[train_amount:])
    grouped_eval_address = tuple(
        [list(e) for e in zip(*eval_address)])  # Convert to tuple of list[image address, label]

    # # Split training and test (Split 80:20)
    # train_address = []
    # for i in range(int(train_eval_ratio * len(labels))):
    #     train_address.append([image_address[i * numdeg:(i + 1) * numdeg], labels[i]])
    # grouped_train_address = tuple(
    #     [list(e) for e in zip(*train_address)])  # Convert to tuple of list[image address, label]
    # eval_address = []
    # for i in range(int(train_eval_ratio * len(labels)), len(labels)):
    #     eval_address.append([image_address[i * numdeg:(i + 1) * numdeg], labels[i]])
    # grouped_eval_address = tuple(
    #     [list(e) for e in zip(*eval_address)])  # Convert to tuple of list[image address, label]
    print(grouped_eval_address)
    print("Train files: %d, Evaluate Files: %d" % (len(grouped_train_address[0]), len(grouped_eval_address[0])))

    # Save names of files of train address
    file_name = './data/' + tfrecord_name + '_train_address.txt'
    with open(file_name, 'w') as filehandle:
        for listitem in train_address:
            new_list_item = listitem[0][0].replace('_0.png', '')
            filehandle.write('%s\n' % new_list_item)

    # Save names of files of eval address
    file_name = './data/' + tfrecord_name + '_eval_address.txt'
    with open(file_name, 'w') as filehandle:
        for listitem in eval_address:
            new_list_item = listitem[0][0].replace('_0.png', '')
            filehandle.write('%s\n' % new_list_item)

    # Start writing train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(grouped_train_address)
    train_dataset = train_dataset.map(read_file)

    it = tf.compat.v1.data.make_one_shot_iterator(train_dataset)

    elem = it.get_next()

    with tf.compat.v1.Session() as sess:
        writer = tf.io.TFRecordWriter(train_name)
        while True:
            try:
                elem_result = serialize(sess.run(elem))

                writer.write(elem_result)
            except tf.errors.OutOfRangeError:
                break
        writer.close()

    eval_address = tf.data.Dataset.from_tensor_slices(grouped_eval_address)
    eval_address = eval_address.map(read_file)

    it = tf.compat.v1.data.make_one_shot_iterator(eval_address)

    elem = it.get_next()

    with tf.compat.v1.Session() as sess:
        writer = tf.io.TFRecordWriter(eval_name)
        while True:
            try:
                elem_result = serialize(sess.run(elem))

                writer.write(elem_result)
            except tf.errors.OutOfRangeError:
                break
        writer.close()
    print("TFrecords created: %s, %s" % (train_name, eval_name))


if __name__ == '__main__':
    # File name will be [tfrecord_name]_train_Taper_sum_median
    tfrecord_name = "preparation_181_data"
    # Directory of image
    dataset_folder = "./data/cross_section"
    run(tfrecord_name, dataset_folder)
    print("Complete")
