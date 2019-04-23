import tensorflow as tf
import numpy as np

# Fixed parameter
numdegree = 4  # Number of rotations
image_height = 240  # Used for cropping
image_width = 360  # Used for cropping


# Import tfrecord to dataset
def deserialize(example):
    feature = {'label': tf.FixedLenFeature([], tf.int64)}
    for i in range(numdegree):
        feature['img' + str(i)] = tf.FixedLenFeature([], tf.string)
    return tf.parse_single_example(example, feature)


def decode(data_dict):
    if numdegree != 4:
        raise Exception('Number of degree specified is not compatible, edit code')
    # Create initial image, then stacking it
    image_decoded = []

    # Stacking the rest
    for i in range(0, numdegree):
        img = data_dict['img' + str(i)]
        file_decoded = tf.image.decode_png(img, channels=1)
        file_cropped = tf.squeeze(tf.image.resize_image_with_crop_or_pad(file_decoded, image_height, image_width))
        image_decoded.append(file_cropped)

    image_stacked = tf.stack([image_decoded[0], image_decoded[1], image_decoded[2], image_decoded[3]], axis=2)
    image_stacked = tf.cast(image_stacked, tf.float32)
    label = tf.cast(data_dict['label'], tf.float32)
    # output = (image_stacked, label)
    # return {'images': image_stacked, 'label': label}  # Output is [Channel, Height, Width]
    return image_stacked, label


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def train_input_fn(data_path, batch_size):
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(deserialize, num_parallel_calls=7)
    dataset = dataset.map(decode, num_parallel_calls=7)
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

    iterator = dataset.make_one_shot_iterator()
    next_image_data = iterator.get_next()
    images = [1]
    label = [2]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        try:
            # Keep extracting data till TFRecord is exhausted
            while True:
                data = sess.run(next_image_data)
                images.append(data[0])
                label.append(data[1])
        except tf.errors.OutOfRangeError:
            pass

    # dataset = dataset.batch(1000, drop_remainder=False)
    # whole_dataset_tensors = tf.data.experimental.get_single_element(dataset)
    # with tf.Session() as sess:
    #     whole_dataset_arrays = sess.run(whole_dataset_tensors)
    #     print(whole_dataset_arrays)
    # images = whole_dataset_arrays[0]
    # label = whole_dataset_arrays[1]
    return images, label

