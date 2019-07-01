import tensorflow as tf
import os
import numpy as np

# Read TFRecord file, return as tf.dataset, specifically used for

numdegree = 4
# Import tfrecord to dataset
def deserialize(example):
    feature = {'label': tf.FixedLenFeature([], tf.int64),
               'degree': tf.FixedLenFeature([], tf.int64),
               'length': tf.FixedLenFeature([], tf.int64)}

    for i in range(4):
        for j in range(2):
            feature['img_%s_%s' % (i, j)] = tf.VarLenFeature(tf.float32)

    return tf.parse_single_example(example, feature)


def decode_one_axis(data_dict):
    # Create initial image, then stacking it
    image_decoded = []
    degree = tf.cast(data_dict['degree'], tf.int32)
    length = tf.cast(data_dict['length'], tf.int32)
    # Stacking the rest
    for i in range(numdegree):
        for j in range(2):
            img = data_dict['img_%s_%s' % (i, j)]
            img = tf.sparse.to_dense(img)
            img = tf.reshape(img, [data_dict['length']])
            # file_cropped = tf.squeeze(tf.image.resize_image_with_crop_or_pad(file_decoded, image_height, image_width))
            image_decoded.append(img)

    if numdegree != 4:
        raise ValueError("Edit this function as well, this compatible with numdeg=4")
    image_stacked = tf.concat([image_decoded[0], image_decoded[1], image_decoded[2], image_decoded[3],
                               image_decoded[4], image_decoded[5], image_decoded[6], image_decoded[7]], axis=0)
    # image_stacked.set_shape([tf.multiply(tf.convert_to_tensor(numdegree),length)])
    image_stacked = tf.cast(image_stacked, tf.float32)
    label = tf.cast(data_dict['label'], tf.float32)
    # output = (image_stacked, label)
    # return {'images': image_stacked, 'label': label}  # Output is [Channel, Height, Width]
    return image_stacked, label


def decode_multiple_axis(data_dict):
    # Create initial image, then stacking it
    image_decoded = []
    degree = tf.cast(data_dict['degree'], tf.int32)
    # length = tf.cast(data_dict['length'], tf.int32)
    length = data_dict['length']
    # Stacking the rest
    for i in range(numdegree):
        for j in range(2):
            img = data_dict['img_%s_%s' % (i, j)]
            img = tf.sparse.to_dense(img)
            # img = tf.reshape(img, [299])
            img = tf.reshape(img, [length])
            # img = tf.reshape(img, [data_dict['length']])
            # file_cropped = tf.squeeze(tf.image.resize_image_with_crop_or_pad(file_decoded, image_height, image_width))
            image_decoded.append(img)

    if numdegree != 4:
        raise ValueError("Edit this function as well, this compatible with numdeg=4")
    image_stacked = tf.stack([image_decoded[0], image_decoded[1], image_decoded[2], image_decoded[3],
                              image_decoded[4], image_decoded[5], image_decoded[6], image_decoded[7]], axis=1)
    image_stacked = tf.cast(image_stacked, tf.float32)
    label = tf.cast(data_dict['label'], tf.float32)
    # output = (image_stacked, label)
    # return {'images': image_stacked, 'label': label}  # Output is [Channel, Height, Width]
    return image_stacked, label


def train_input_fn(data_path, batch_size, data_type):
    if not os.path.exists(data_path):
        raise ValueError("Train input file does not exist")
    # data_type=0 -> data is vectorize in to one vector else, stack in different dimension
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(deserialize, num_parallel_calls=7)
    if data_type == 0:
        dataset = dataset.map(decode_one_axis, num_parallel_calls=7)
    else:
        dataset = dataset.map(decode_multiple_axis, num_parallel_calls=7)
    dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size, drop_remainder=False)  # Maybe batch after repeat?
    dataset = dataset.repeat(None)
    dataset = dataset.prefetch(buffer_size=None)
    return dataset


def eval_input_fn(data_path, batch_size, data_type):
    if not os.path.exists(data_path):
        raise ValueError("Eval input file does not exist")
    eval_dataset = tf.data.TFRecordDataset(data_path)
    eval_dataset = eval_dataset.map(deserialize)
    if data_type == 0:
        eval_dataset = eval_dataset.map(decode_one_axis)
    else:
        eval_dataset = eval_dataset.map(decode_multiple_axis)
    eval_dataset = eval_dataset.batch(batch_size, drop_remainder=False)  # No need to shuffle this time
    return eval_dataset


def get_data_from_path(data_path, data_type = 0):
    if not os.path.exists(data_path):
        raise ValueError("Input file does not exist")
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(deserialize)
    if data_type == 0:
        dataset = dataset.map(decode_one_axis)
    else:
        dataset = dataset.map(decode_multiple_axis)

    iterator = dataset.make_one_shot_iterator()
    next_image_data = iterator.get_next()
    images = []
    label = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        try:
            # Keep extracting data till TFRecord is exhausted
            while True:
                data = sess.run(next_image_data)
                # print(data)
                images.append(data[0])
                label.append(data[1])
        except tf.errors.OutOfRangeError:
            pass

    return images, label
