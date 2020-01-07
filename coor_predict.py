from utils.stl_slicer import get_cross_section
import argparse
import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from utils.open_save_file import predict_get_cross_section, get_file_name
from preprocess.stl_to_image import point_sampling, stl_point_to_movement
from utils.coor_get_data import get_data_from_path

data_type_option = ["stl", "npy", "numpy", "tfrecord"]

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test_directory', type=str, help="Directory of folder or file of test dataset")
parser.add_argument('-m', '--model_directory', help="Directory of trained model")
parser.add_argument('-dt', '--dataset_type', choices=data_type_option, type=str,
                    help="Type of dataset whether it's stl or image")
parser.add_argument('-d', '--degree', type=int, nargs='+', default=[0, 135, 45, 90],
                    help="List of degree used for dataset_type:stl")
parser.add_argument('-l', '--length', type=int, default=300, help="Length of data for dataset_type:tfrecord")
parser.add_argument('-f', '--fix_amount', type=int, default=0,
                    help="Number of point to be sampled for dataset_type:stl")

args = parser.parse_args()
degree = [0, 135, 45, 90]


# tf.enable_eager_execution()
def deserialize(example):
    d_feature = {'degree': tf.FixedLenFeature([], tf.int64),
                 'length': tf.FixedLenFeature([], tf.int64),
                 "name": tf.FixedLenFeature([], tf.string)}  # Becareful if tfrecord doesn't have name

    for i in range(4):
        for j in range(2):
            d_feature['img_%s_%s' % (i, j)] = tf.VarLenFeature(tf.float32)

    return tf.parse_single_example(example, d_feature)


def decode(data_dict):
    single_slice = False  # Unused parameter
    numdegree = len(args.degree)
    # Create initial image, then stacking it for 1dCNN model
    image_decoded = []

    # Stacking the rest
    for i in range(numdegree):
        for j in range(2):
            img = data_dict['img_%s_%s' % (i, j)]
            img = tf.sparse.to_dense(img)
            img = tf.reshape(img, [args.length])
            image_decoded.append(img)

    if single_slice:  # Fetch only the first cross-section
        image_stacked = tf.stack(image_decoded[0:numdegree * 2:numdegree], axis=1)
    else:  # Fetch all cross-section
        image_stacked = tf.stack(image_decoded[0:numdegree * 2], axis=1)

    image_stacked = tf.cast(image_stacked, tf.float32)
    name = tf.cast(data_dict['name'], tf.string)
    d_feature = {'image': image_stacked, 'name': name}
    return d_feature


def create_dataset(features, file_name):
    stacked_image = []
    for img in features:
        single_image = None
        for d in range(len(degree)):
            if single_image is None:
                single_image = img[d]
            else:
                single_image = np.concatenate((single_image, img[d]), axis=1)
        stacked_image.append(single_image)
    return tf.data.Dataset.from_tensor_slices({'image': np.stack(stacked_image, axis=0), 'name': file_name})


def read_data(dataset):
    iterator = dataset.make_one_shot_iterator()
    next_image_data = iterator.get_next()
    features = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            # Keep extracting data till TFRecord is exhausted
            while True:
                data = sess.run(next_image_data)
                features.append(data)
        except tf.errors.OutOfRangeError:
            pass
    return features


if __name__ == '__main__':
    print("Degree", args.degree)
    data_type = args.dataset_type.lower()
    if data_type == "stl":
        predict_cross_section, file_name_all, error_name = predict_get_cross_section(degree, augment_config=None,
                                                                                     folder_name=args.test_directory)
        # Sampling point to a fixed number
        if args.fix_amount > 0:
            print("Adjusting number of coordinates... Takes a long time")
            for data in range(len(predict_cross_section)):
                for d_index in range(len(degree)):
                    predict_cross_section[data][d_index] = point_sampling(predict_cross_section[data][d_index],
                                                                          args.fix_amount + 1)
                if data % 50 == 0:
                    print("Done %s out of %s" % (data + 1, len(predict_cross_section)))

        predict_cross_section = [stl_point_to_movement(p) for p in predict_cross_section]
        my_dataset = create_dataset(predict_cross_section, file_name_all)
        feature = read_data(dataset=my_dataset)
        print("Error:", error_name)
    elif data_type == "npy" or data_type == "numpy":
        name_dir, image_name = get_file_name(folder_name=args.test_directory,
                                             exception_file=["score.csv", "config.json"])
        predict_cross_section = [np.load(n) for n in name_dir]
        file_name = [os.path.basename(n).split(".")[:-1] for n in name_dir]
        my_dataset = create_dataset(predict_cross_section, file_name)
        feature = read_data(dataset=my_dataset)
    elif data_type == "tfrecord":
        my_dataset = tf.data.TFRecordDataset(args.test_directory)
        my_dataset = my_dataset.map(deserialize, num_parallel_calls=7)
        my_dataset = my_dataset.map(decode, num_parallel_calls=7)
        feature = read_data(dataset=my_dataset)
    else:
        raise ValueError("data_type not in the option")

    subdirs = [x for x in Path(args.model_directory).iterdir() if x.is_dir() and 'temp' not in str(x)
               and 'eval' not in str(x) and 'train_final' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    predict_fn = tf.contrib.predictor.from_saved_model(latest)

    print(feature)
    for f in feature:
        predictions = predict_fn({"image": np.expand_dims(f["image"], axis=0)})
        predict_class = predictions['score'][0][0]
        predict_confident = predictions['probabilities'][0][predict_class]
        print("Prediction of %s: Score = %s with probability %s" % (
            f['name'], predict_class * 2 + 1, predictions['probabilities'][0]))
