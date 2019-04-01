import tensorflow as tf
# tf.enable_eager_execution()
import numpy as np
import os
import argparse
from shutil import copy2
import csv
import datetime

from protobuf_helper import protobuf_to_list, protobuf_to_channels
from proto import tooth_pb2
from google.protobuf import text_format

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

from cifar10_model import my_model
from cifar10_get_data import train_input_fn, eval_input_fn, get_data_from_path


# These are important parameters
run_params = {'batch_size': 4,
              'checkpoint_min': 5,
              'early_stop_step': 10000,
              'input_path': './data/cifar10.tfrecords',
              'result_path_new': '/home/pasin/Documents/Pasin/model/cifar10',
              'config_path': '',
              'steps': 100000}


model_configs = {'learning_rate': 0.0001,
                 'dropout_rate': 0.1,
                 'activation': tf.nn.relu,
                 'channels': [16, 16, 32, 32, 32, 32, 32, 32, 32, 2048, 2048]
                 }


def run(model_params={}):
    # Add exception in case params is missing
    if len(model_params['channels']) != 11:
        raise Exception("Number of channels not correspond to number of layers [Need size of 11, got %s]"
                        % len(model_params['channels']))

    # Type in file name
    train_data_path = run_params['input_path'].replace('.tfrecords', '') + '_train.tfrecords'
    eval_data_path = run_params['input_path'].replace('.tfrecords', '') + '_eval.tfrecords'
    print("Getting training data from %s" % train_data_path)
    print("Saved model at %s" % run_params['result_path_new'])

    tf.logging.set_verbosity(tf.logging.INFO)  # To see some additional info
    # Setting checkpoint config
    my_checkpoint_config = tf.estimator.RunConfig(
        save_checkpoints_secs=run_params['checkpoint_min'] * 60,
        # save_summary_steps=pareval_data_pathams['checkpoint_min'] * 10,
        keep_checkpoint_max=10,
        log_step_count_steps=500,
        session_config=tf.ConfigProto(allow_soft_placement=True)
    )
    # Or set up the model directory
    #   estimator = DNNClassifier(
    #       config=tf.estimator.RunConfig(
    #           model_dir='/my_model', save_summary_steps=100),
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params=model_params,
        model_dir=run_params['result_path_new'],
        config=my_checkpoint_config
    )
    train_hook = tf.contrib.estimator.stop_if_no_decrease_hook(classifier, "loss", run_params['early_stop_step'])
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_input_fn(train_data_path, batch_size=run_params['batch_size']),
        max_steps=run_params['steps'], hooks=[train_hook])
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: eval_input_fn(eval_data_path, batch_size=16), steps=None,
        start_delay_secs=0, throttle_secs=0)
    # classifier.train(input_fn=lambda: train_input_fn(train_data_path, batch_size=params['batch_size']),
    #     max_steps=params['steps'], hooks=[train_hook])
    # eval_result = classifier.evaluate(input_fn=lambda: eval_input_fn(eval_data_path, batch_size=32))

    eval_result = tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    print("Eval result:")
    print(eval_result)
    try:
        accuracy = eval_result[0]['accuracy']
        global_step = eval_result[0]['global_step']
    except TypeError:
        print("Warning, does receive evaluation result")
        accuracy = 0
        global_step = 0

    predictions = classifier.predict(input_fn=lambda: eval_input_fn(eval_data_path, batch_size=1))
    print(predictions)
    images, expected = get_data_from_path(eval_data_path)
    predict_score = ['Prediction']
    probability_score = ['Probability']
    label_score = ['Label']
    for pred_dict, expec in zip(predictions, expected):
        # print(pred_dict)
        print("Score" + str(pred_dict['score']))
        class_id = pred_dict['score'][0]
        probability = pred_dict['probabilities'][class_id]
        print("Actual score: %s, Predicted score: %s with probability %s" % (expec, class_id, probability))
        predict_score.append(class_id)
        label_score.append(expec)
        probability_score.append(probability)

    predict_result = zip(label_score, predict_score, probability_score)

    # print(eval_result[0]['accuracy'])
    # print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    return accuracy, global_step, predict_result


if __name__ == '__main__':
    # read_file()
    acc, steps, predict_result = run(model_configs)
    print(predict_result)
    print("train.py completed")
