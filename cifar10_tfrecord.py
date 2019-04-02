import tensorflow as tf
import numpy as np
import os
import sys
import time
import ast
import warnings
import cv2
from random import shuffle


def serialize(example):
    feature = {'label': _int64_feature(example['label']),
               'img': _bytes_feature(example['img'].tobytes())}

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

    return tf_example.SerializeToString()


def read_file(image, label):
    file_values = {'label': label, 'img': image}
    return file_values


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def run(tfrecord_name):
    # Start getting all info and zip to tfrecord
    tfrecord_train_name = "%s_train.tfrecords" % (tfrecord_name)
    tfrecord_eval_name = "%s_eval.tfrecords" % (tfrecord_name)
    tfrecord_train_name = os.path.join("./data", tfrecord_train_name)
    tfrecord_eval_name = os.path.join("./data", tfrecord_eval_name)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.map(read_file)
    it = train_dataset.make_one_shot_iterator()
    elem = it.get_next()

    with tf.Session() as sess:
        writer = tf.python_io.TFRecordWriter(tfrecord_train_name)
        while True:
            try:
                elem_result = serialize(sess.run(elem))
                # print(elem_result)
                writer.write(elem_result)
            except tf.errors.OutOfRangeError:
                break
        writer.close()

    eval_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    eval_dataset = eval_dataset.map(read_file)

    it = eval_dataset.make_one_shot_iterator()
    elem = it.get_next()

    with tf.Session() as sess:
        writer = tf.python_io.TFRecordWriter(tfrecord_eval_name)
        while True:
            try:
                elem_result = serialize(sess.run(elem))
                # print(elem_result)
                writer.write(elem_result)
            except tf.errors.OutOfRangeError:
                break
        writer.close()
    print("TFrecords created: %s, %s" % (tfrecord_train_name, tfrecord_eval_name))


if __name__ == '__main__':
    tfrecord_name = 'cifar10'
    run(tfrecord_name)
    print("Complete")
