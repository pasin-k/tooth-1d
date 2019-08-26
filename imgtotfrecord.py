import tensorflow as tf
import numpy as np
import os
from open_save_file import get_input_and_label, read_file, save_file

numdeg = 4  # Number of images on each example

# Global Variable
configs = {'train_eval_ratio': 0.8,
           'data_type': 'BL_median',
           }

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# Run images from pre_processing.py into tfrecords
def serialize_image(example):
    # Record name and image data
    feature = {}
    for i in range(numdeg):
        feature['img' + str(i)] = _bytes_feature(example['img' + str(i)])
    # Record all available label
    for d in configs['data_type']:
        if d == "name":
            feature[d] = _bytes_feature(bytes(example[d], encoding='utf8'))
        else:
            feature[d] = _int64_feature(example[d])

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example.SerializeToString()


def read_image(file_name, label):
    file_values = label
    # file_values = {'label': label}
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


def image_to_tfrecord(tfrecord_name, dataset_folder, csv_dir=None, k_fold=False, k_num=5):
    """
    tfrecord_name   : Name of .tfrecord output file
    dataset_folder  : Folder of the input data  (Not include label)
    csv_dir         : Folder of label data (If not specified, will use the default directory)
    save 4 files: train.tfrecord, eval.tfrecord, .txt (Save from another file)
    """
    # Create new directory if not created, get all info and zip to tfrecord
    tfrecord_dir = os.path.join("./data/tfrecord", tfrecord_name)
    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)

    # Get amount of degree and augmentation
    config_data = read_file(os.path.join(dataset_folder, "config.txt"))
    configs['numdeg'] = config_data[0]
    configs['num_augment'] = config_data[1]

    # Get file name from dataset_folder
    grouped_train_address, grouped_eval_address = get_input_and_label(tfrecord_name, dataset_folder, csv_dir, configs,
                                                                      get_data=False, k_cross=k_fold, k_num=k_num)

    if not k_fold:
        k_num = 1
        grouped_train_address = [grouped_train_address]
        grouped_eval_address = [grouped_eval_address]
    for i in range(k_num):
        train_address = grouped_train_address[i]
        eval_address = grouped_eval_address[i]

        # Start writing train dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(train_address)
        train_dataset = train_dataset.map(read_image)  # Read file address, and get info as string

        it = train_dataset.make_one_shot_iterator()

        elem = it.get_next()

        # tfrecord_train_name = os.path.join(tfrecord_dir, "%s_%s_%s_%s_train.tfrecords" % (
        #     tfrecord_name, configs['label_data'], configs['label_type'], i))
        # tfrecord_eval_name = os.path.join(tfrecord_dir, "%s_%s_%s_%s_eval.tfrecords" % (
        #     tfrecord_name, configs['label_data'], configs['label_type'], i))
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


# Run images from pre_processing.py into tfrecords, not using right now
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


def coordinate_to_tfrecord(tfrecord_name, dataset_folder, csv_dir=None, k_fold=False, k_num=5):
    """
    tfrecord_name   : Name of .tfrecord output file
    dataset_folder  : Folder of the input data  (Not include label)
    csv_dir         : Folder of label data (If not specified, will use the default directory)
    save 4 files: train.tfrecord, eval.tfrecord, .txt (Save from another file)
    """

    # Create new directory if not created, get all info and zip to tfrecord
    tfrecord_dir = os.path.join("./data/tfrecord", tfrecord_name)
    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)

    # Get amount of degree and augmentation
    config_data = read_file(os.path.join(dataset_folder, "config.txt"))
    configs['numdeg'] = int(config_data[0][0])      #  Get data from config.txt
    configs['num_augment'] = int(config_data[1][0])

    # Get data from dataset_folder
    grouped_train_data, grouped_eval_data = get_input_and_label(tfrecord_name, dataset_folder,
                                                                      csv_dir, configs, get_data=True,
                                                                      k_cross=k_fold, k_num=k_num)
    # Use this to debug if number of point in varying
    # print(len(grouped_eval_data))
    # for d in grouped_eval_data:
    #     print(np.shape(d[0][0]))
    # raise ValueError("debug")

    if not k_fold:
        k_num = 1
        grouped_train_data = [grouped_train_data]
        grouped_eval_data = [grouped_eval_data]
    for i in range(k_num):
        train_data = grouped_train_data[i]
        eval_data = grouped_eval_data[i]
        # tfrecord_train_name = os.path.join(tfrecord_dir, "%s_%s_%s_%s_train.tfrecords" % (
        #     tfrecord_name, configs['label_data'], configs['label_type'], i))
        # tfrecord_eval_name = os.path.join(tfrecord_dir, "%s_%s_%s_%s_eval.tfrecords" % (
        #     tfrecord_name, configs['label_data'], configs['label_type'], i))
        tfrecord_train_name = os.path.join(tfrecord_dir, "%s_%s_train.tfrecords" % (tfrecord_name, i))
        tfrecord_eval_name = os.path.join(tfrecord_dir, "%s_%s_eval.tfrecords" % (tfrecord_name, i))

        coordinate_length = len(train_data[0][0][0])
        degree = len(train_data[0][0])

        with tf.python_io.TFRecordWriter(tfrecord_train_name) as writer:
            for data in train_data:
                # print(data)
                # Add general info
                feature = {'degree': _int64_feature(degree),
                           'length': _int64_feature(coordinate_length)}
                # Add labels
                for d in configs['data_type']:
                    if d == "name":
                        feature[d] = _bytes_feature(bytes(data[1][d], encoding='utf8'))
                    else:
                        feature[d] = _int64_feature(data[1][d])
                # Add data
                for n in range(numdeg):
                    for j in range(2):  # Flatten degree, axis
                        val = data[0][n][:, j].reshape(-1)
                        feature['img_%s_%s' % (n, j)] = tf.train.Feature(float_list=tf.train.FloatList(value=val))
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                # Write TFrecord file
                writer.write(example.SerializeToString())

        with tf.python_io.TFRecordWriter(tfrecord_eval_name) as writer:
            for data in eval_data:
                feature = {'degree': _int64_feature(degree),
                           'length': _int64_feature(coordinate_length)}
                # Add labels
                for d in configs['data_type']:
                    if d == "name":
                        feature[d] = _bytes_feature(bytes(data[1][d], encoding='utf8'))
                    else:
                        feature[d] = _int64_feature(data[1][d])
                # Add data
                for n in range(numdeg):
                    for j in range(2):
                        val = data[0][n][:, j].reshape(-1)
                        feature['img_%s_%s' % (n, j)] = tf.train.Feature(float_list=tf.train.FloatList(value=val))
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                # Write TFrecord file
                writer.write(example.SerializeToString())
        print("TFrecords created: %s, %s" % (tfrecord_train_name, tfrecord_eval_name))


if __name__ == '__main__':
    get_image = False
    # Select type of label to use
    label_data = ["name", "Occ_B_median", "Occ_F_median", "Occ_L_median", "BL_median", "MD_median", "Integrity_median",
                  "Width_median", "Surface_median", "Sharpness_median"]
    # label_type = ["median"]
    configs['train_eval_ratio'] = 0.8
    # configs['label_data'] = "Taper_Sum"
    # configs['label_type'] = "median"
    k_fold = False
    # label_datas = ["Taper_Sum", "BL", "MD", "Occ_Sum", "Occ_L", "Occ_F",
    #                "Occ_B"]  # Too lazy to do all of these one at a time
    # label_datas = ["BL"]  # For debug

    configs['data_type'] = label_data
    print("Use label from %s with (%s) train:eval ratio" % (
        configs['data_type'], configs['train_eval_ratio']))

    if get_image:
        image_to_tfrecord(tfrecord_name="preparation_img_test", dataset_folder="./data/cross_section",
                          k_fold=k_fold)
    else:
        # coordinate_to_tfrecord(tfrecord_name="preparation_coor_newer",
        #                        dataset_folder="./data/coordinate_newer", k_fold=k_fold)
        coordinate_to_tfrecord(tfrecord_name="left_segment",
                               dataset_folder="./data/segment_2/left_point", k_fold=k_fold)
    print("Complete")

    # for label_data_index in label_datas:
    #     configs['label_data'] = label_data_index
    #     print("Use label from %s category '%s' with (%s) train:eval ratio" % (
    #         configs['label_data'], configs['label_type'], configs['train_eval_ratio']))
    #     # File name will be [tfrecord_name]_train_Taper_sum_median
    #
    #     # tfrecord_name = "original_preparation_data"
    #     # csv_name = "../global_data/Ground Truth Score_50.csv"
    #     # Directory of image
    #
    #     if get_image:
    #         image_to_tfrecord(tfrecord_name="preparation_img_test", dataset_folder="./data/cross_section",
    #                           k_fold=k_fold)
    #     else:
    #         coordinate_to_tfrecord(tfrecord_name="preparation_coor_debug_new",
    #                                dataset_folder="./data/coordinate_debug_new", k_fold=k_fold)
    #     print("Complete")
