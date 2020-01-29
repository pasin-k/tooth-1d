import tensorflow as tf
import os
import numpy as np

# Fixed parameter
numdegree = 4  # Number of rotations
image_height = 299  # Used for cropping
image_width = 299  # Used for cropping


# Import tfrecord to dataset
def deserialize(example):
    global numdegree, label_type_global
    feature = {label_type_global: tf.FixedLenFeature([], tf.int64),
               'degree': tf.FixedLenFeature([], tf.int64),
               # 'length': tf.FixedLenFeature([], tf.int64),
               "name": tf.FixedLenFeature([], tf.string)}  # Becareful if tfrecord doesn't have name
    for i in range(numdegree):
        feature['img_' + str(i)] = tf.FixedLenFeature([], tf.string)
    return tf.parse_single_example(example, feature)


def decode(data_dict):
    global numdegree, image_height, image_width
    if numdegree != 4:
        raise Exception('Number of degree specified is not compatible, edit code')
    # Create initial image, then stacking it
    image_decoded = []
    numdegree = 1 # Special case
    # Stacking the rest
    for i in range(0, numdegree):
        img = data_dict['img_' + str(i)]
        file_decoded = tf.image.decode_png(img, channels=1)
        file_cropped = tf.squeeze(tf.image.resize_image_with_crop_or_pad(file_decoded, image_height, image_width))
        image_decoded.append(file_cropped)

    # image_stacked = tf.stack([image_decoded[0], image_decoded[1], image_decoded[2], image_decoded[3]], axis=2)
    image_stacked = [image_decoded[0]/255]
    image_stacked = tf.cast(image_stacked, tf.float32)
    label = tf.cast(data_dict[label_type_global], tf.float32)
    name = tf.cast(data_dict['name'], tf.string)
    feature = {'image': image_stacked, 'name': name}
    # output = (image_stacked, label)
    # return {'images': image_stacked, 'label': label}  # Output is [Channel, Height, Width]
    return feature, label


# def _int64_feature(value):
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#
#
# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
#
# def _float_feature(value):
#     return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def train_input_fn(data_path, batch_size, configs):
    global numdegree, label_type_global, name_type
    numdegree, label_type_global, name_type = configs['data_degree'], \
                                              configs['label_type'], \
                                              configs['dataset_name']
    print("Fetching label type for training: %s" % label_type_global)
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(deserialize, num_parallel_calls=7)
    dataset = dataset.map(decode, num_parallel_calls=7)
    dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size, drop_remainder=False)  # Maybe batch after repeat?
    dataset = dataset.repeat(None)
    dataset = dataset.prefetch(buffer_size=None)
    return dataset


def eval_input_fn(data_path, batch_size, configs):
    global numdegree, label_type_global, name_type
    numdegree, label_type_global, name_type = configs['data_degree'], \
                                              configs['label_type'], \
                                              configs['dataset_name']
    eval_dataset = tf.data.TFRecordDataset(data_path)
    eval_dataset = eval_dataset.map(deserialize)
    eval_dataset = eval_dataset.map(decode)
    eval_dataset = eval_dataset.batch(batch_size, drop_remainder=False)  # No need to shuffle this time
    return eval_dataset





def get_data_from_path(data_path, label_type):
    global numdegree, label_type_global, name_type
    dataset = tf.data.TFRecordDataset(data_path)
    label_data = ["name", "Occ_B_median", "Occ_F_median", "Occ_L_median", "BL_median", "MD_median", "Integrity_median",
                  "Width_median", "Surface_median", "Sharpness_median"]
    numdegree = 4
    # data_length = 300
    name_type = ["img"]
    label_type_global = label_type
    if not os.path.exists(data_path):
        raise ValueError("Input file does not exist")
    # print(dataset)
    dataset = dataset.map(deserialize)
    dataset = dataset.map(decode)


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
                images.append(data[0])
                label.append(data[1])
        except tf.errors.OutOfRangeError:
            pass
    return images, label


def read_raw_tfrecord(tfrecord_path):  # For debugging purpose, reading all content inside
    i = 0
    for example in tf.python_io.tf_record_iterator(tfrecord_path):
        i += 1
        print(tf.train.Example.FromString(example))
    print(i)


if __name__ == '__main__':
    data_path = "/home/pasin/Documents/Google_Drive/Aa_TIT_LAB_Comp/Library/Tooth/Model/my2DCNN/data/tfrecord/image_14aug" \
                "/image_14aug_0_eval.tfrecords"
    label_type = "Width_median"

    f, l = get_data_from_path(data_path, label_type)
    # print("Feature", f[0])
    print("Label", l)
    print("1", l.count(1.0))
    print("3", l.count(3.0))
    print("5", l.count(5.0))
    # read_raw_tfrecord(data_path)
