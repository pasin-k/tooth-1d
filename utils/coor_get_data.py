import tensorflow as tf
import os
import numpy as np

# Read TFRecord file, return as tf.dataset, specifically used for

numdegree = None
data_length = None
label_type_global = None
single_slice = False
name_type = None


# Import tfrecord to dataset
def deserialize(example):
    global name_type
    feature = {label_type_global: tf.FixedLenFeature([], tf.int64),
               'degree': tf.FixedLenFeature([], tf.int64),
               'length': tf.FixedLenFeature([], tf.int64),
               "name": tf.FixedLenFeature([], tf.string)}  # Becareful if tfrecord doesn't have name

    for n in name_type:
        for i in range(4):
            for j in range(2):
                feature['%s_%s_%s' % (n, i, j)] = tf.VarLenFeature(tf.float32)

    return tf.parse_single_example(example, feature)


def decode(data_dict):
    global label_type_global, numdegree, name_type, single_slice
    # Create initial image, then stacking it for 1dCNN model
    image_decoded = []
    dataset_amount = len(name_type)
    # Stacking the rest
    for n in name_type:
        for i in range(numdegree):
            for j in range(2):
                img = data_dict['%s_%s_%s' % (n, i, j)]
                img = tf.sparse.to_dense(img)
                img = tf.reshape(img, [data_length])
                # file_cropped = tf.squeeze(tf.image.resize_image_with_crop_or_pad(file_decoded, image_height, image_width))
                image_decoded.append(img)

    if numdegree != 4:
        raise ValueError("Edit this function as well, this compatible with numdeg=4")
    if single_slice:  # Fetch only the first cross-section
        image_stacked = tf.stack(image_decoded[0:numdegree * dataset_amount * 2:numdegree], axis=1)
        if dataset_amount > 4 or dataset_amount < 1:
            raise ValueError("Datset_amount not compatible: Found %s, but accept 1-4" % dataset_amount)
    else:  # Fetch all cross-section
        image_stacked = tf.stack(image_decoded[0:numdegree * dataset_amount * 2], axis=1)
        if dataset_amount > 4 or dataset_amount < 1:
            raise ValueError("Datset_amount not compatible: Found %s, but accept 1-4" % dataset_amount)

    image_stacked = tf.cast(image_stacked, tf.float32)
    label = tf.cast(data_dict[label_type_global], tf.float32)
    name = tf.cast(data_dict['name'], tf.string)
    feature = {'image': image_stacked, 'name': name}
    return feature, label


def train_input_fn(data_path, batch_size, configs):
    """
    Use to fetch training dataset
    :param data_path: Training tfrecords path
    :param batch_size: Batch size
    :param configs: Dictionary, should contain data_degree, data_length, label_type, dataset_name
    :return: tf.Dataset use for training
    """
    global numdegree, data_length, label_type_global, single_slice, name_type
    numdegree, data_length, label_type_global, name_type = configs['data_degree'], configs['data_length'], \
                                                           configs['label_type'], \
                                                           configs['dataset_name']
    print("Fetching label type for training: %s" % label_type_global)

    if not os.path.exists(data_path):
        raise ValueError("Train input file does not exist")
    # data_type=0 -> data is vectorize in to one vector else, stack in different dimension
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(deserialize, num_parallel_calls=7)
    dataset = dataset.map(decode, num_parallel_calls=7)
    dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size, drop_remainder=False)  # Maybe batch after repeat?
    dataset = dataset.repeat(None)
    dataset = dataset.prefetch(buffer_size=None)
    return dataset


def eval_input_fn(data_path, batch_size, configs):
    """
    Use to fetch validation dataset
    :param data_path: Evaluate tfrecords path
    :param batch_size: Batch size
    :param configs: Dictionary, should contain data_degree, data_length, label_type, dataset_name
    :return: tf.Dataset use for validation
    """
    global numdegree, data_length, label_type_global, single_slice, name_type
    print("Fetching label type for eval: %s" % label_type_global)
    numdegree, data_length, label_type_global, name_type = configs['data_degree'], configs['data_length'], \
                                                           configs['label_type'], configs['dataset_name']
    if not os.path.exists(data_path):
        raise ValueError("Eval input file does not exist")
    eval_dataset = tf.data.TFRecordDataset(data_path)
    eval_dataset = eval_dataset.map(deserialize)
    eval_dataset = eval_dataset.map(decode)
    eval_dataset = eval_dataset.batch(batch_size, drop_remainder=False)  # No need to shuffle this time
    return eval_dataset


def get_data_from_path(data_path, label_type):
    """
    Use to read data from tfrecords file. Mostly use for debugging
    :param data_path:
    :param label_type:
    :return:
    """

    global numdegree, data_length, label_type_global, name_type

    label_data = ["name", "Occ_B_median", "Occ_F_median", "Occ_L_median", "BL_median", "MD_median", "Integrity_median",
                  "Width_median", "Surface_median", "Sharpness_median"]
    numdegree = 4
    data_length = 0
    name_type = "img"
    label_type_global = label_type
    if not os.path.exists(data_path):
        raise ValueError("Input file does not exist")
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(deserialize)
    dataset = dataset.map(decode)

    iterator = dataset.make_one_shot_iterator()
    next_image_data = iterator.get_next()
    features = []
    label = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            # Keep extracting data till TFRecord is exhausted
            while True:
                data = sess.run(next_image_data)
                features.append(data[0])
                label.append(data[1])
        except tf.errors.OutOfRangeError:
            pass

    return features, label
