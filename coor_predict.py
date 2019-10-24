from utils.stl_slicer import getSlicer
import argparse
import tensorflow as tf
import numpy as np
import os
from utils.open_save_file import predict_get_cross_section
from preprocess.stl_to_image import point_sampling, stl_point_to_movement
from utils.coor_get_data import get_data_from_path

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test_directory', help="Directory of folder or file to test")
parser.add_argument('-f', '--fix_amount', default=0, help="Number of point to be sampled")
args = parser.parse_args()
degree = [0, 45, 90, 135]
fix_amount = int(args.fix_amount)


# tf.enable_eager_execution()


def create_dataset(features):
    stacked_image = []
    for img in features:
        single_image = None
        for d in range(len(degree)):
            if single_image is None:
                single_image = img[d]
            else:
                single_image = np.concatenate((single_image, img[d]), axis=1)
        single_image = single_image.flatten('C')
        stacked_image.append(single_image)
        print("Single image", np.shape(single_image))
    print("Debug", np.shape(np.stack(stacked_image, axis=0)))
    return tf.data.Dataset.from_tensor_slices({'image': stacked_image})


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
    predict_cross_section, error_name = predict_get_cross_section(degree, augment_config=None,
                                                                  folder_name=args.test_directory)
    # Sampling point to a fixed number
    if fix_amount > 0:
        print("Adjusting number of coordinates... Takes a long time")
        for i in range(len(predict_cross_section)):
            for d_index in range(len(degree)):
                predict_cross_section[i][d_index] = point_sampling(predict_cross_section[i][d_index], fix_amount + 1)
            if i % 50 == 0:
                print("Done %s out of %s" % (i + 1, len(predict_cross_section)))

    predict_cross_section = stl_point_to_movement(predict_cross_section)

    my_dataset = create_dataset(predict_cross_section)
    print(my_dataset)

    # for i in my_dataset:
    #     print(i['img_0_0'])
    feature = read_data(dataset=my_dataset)
    for i in feature:
        print(type(i))
        print(np.shape(i["image"]))
    # print(np.shape(output[0]["image"]))
    # print(np.shape(output[0]["img_0_0"]))
