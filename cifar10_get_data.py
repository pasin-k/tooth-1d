import tensorflow as tf
# tf.enable_eager_execution()
import numpy as np

# Fixed parameter
DEPTH = 3  # Number of rotations
HEIGHT = 32  # Used for cropping
WIDTH = 32  # Used for cropping


# Import tfrecord to dataset
def deserialize(example):
    feature = {'label': tf.FixedLenFeature([], tf.int64), 'img': tf.FixedLenFeature([], tf.string)}
    return tf.parse_single_example(example, feature)


def decode(data_dict):
    image = tf.decode_raw(data_dict['img'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])
    image = tf.cast(tf.reshape(image, [HEIGHT, WIDTH, DEPTH]), tf.float32)
    label = tf.cast(data_dict['label'], tf.int64)
    # output = (image_stacked, label)
    # return {'images': image_stacked, 'label': label}  # Output is [Channel, Height, Width]
    return image, label


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def train_input_fn(data_path, batch_size):
    dataset = tf.data.TFRecordDataset(data_path)
    print(dataset)
    dataset = dataset.map(deserialize, num_parallel_calls=7)
    print(dataset)
    dataset = dataset.map(decode, num_parallel_calls=7)
    print(dataset)
    dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size, drop_remainder=False)  # Maybe batch after repeat?
    dataset = dataset.repeat(None)
    dataset = dataset.prefetch(buffer_size=None)
    return dataset


def eval_input_fn(data_path, batch_size):
    eval_dataset = tf.data.TFRecordDataset(data_path)
    eval_dataset = eval_dataset.map(deserialize)
    eval_dataset = eval_dataset.map(decode)
    eval_dataset = eval_dataset.batch(batch_size, drop_remainder=False)  # No need to shuffle this time
    return eval_dataset


def get_data_from_path(data_path):
    dataset = tf.data.TFRecordDataset(data_path)
    # print(dataset)
    dataset = dataset.map(deserialize)
    dataset = dataset.map(decode)

    dataset = dataset.batch(1000, drop_remainder=False)
    whole_dataset_tensors = tf.data.experimental.get_single_element(dataset)
    with tf.Session() as sess:
        whole_dataset_arrays = sess.run(whole_dataset_tensors)
    images = whole_dataset_arrays[0]
    label = whole_dataset_arrays[1]
    return images, label
