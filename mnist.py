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

# Read tooth.config file
parser = argparse.ArgumentParser()
parser.add_argument('config', help="Directory of config file")
# parser.add_argument('--config',help="Config directory")
args = parser.parse_args()

activation_dict = {'0': tf.nn.relu, '1': tf.nn.leaky_relu}  # Declare global dictionary
configs = tooth_pb2.TrainConfig()
with open(args.config, 'r') as f:
    text_format.Merge(f.read(), configs)

# Convert protobuf data to list
learning_rate_list = protobuf_to_list(configs.learning_rate)
keep_prob_list = protobuf_to_list(configs.keep_prob)
activation_list = protobuf_to_list(configs.activation, activation_dict)
channel_list = protobuf_to_channels(configs.channels)


# Check if parameters exist, if not, give default parameters or raiseError
# params: dictionary, dict_name: string, default: value (if None will raiseError)
def check_exist(params, dict_name, default=None):
    try:
        output = params[dict_name]
    except (KeyError, TypeError) as error:
        if default is None:
            raise Exception("Parameter %s not defined" % dict_name)
        else:
            output = default
            print("Parameters: %s not found, use default value = %s" % (dict_name, default))
    return output


# These are important parameters
run_params = {'batch_size': configs.batch_size,
              'checkpoint_min': configs.checkpoint_min,
              'early_stop_step': configs.early_stop_step,
              'input_path': configs.input_path,
              'result_path': configs.result_path,
              'config_path': os.path.abspath(args.config),
              'steps': configs.steps}

run_params['batch_size'] = check_exist(run_params, 'batch_size', 16)
run_params['checkpoint_min'] = check_exist(run_params, 'checkpoint_min', 10)
run_params['early_stop_step'] = check_exist(run_params, 'early_stop_step', 5000)
run_params['input_path'] = check_exist(run_params, 'input_path')
run_params['result_path'] = check_exist(run_params, 'result_path')
# run_params['result_path_new'] = check_exist(run_params, 'result_path')
run_params['steps'] = check_exist(run_params, 'steps')

model_configs = {'learning_rate_list': learning_rate_list,
                 'keep_prob_list': keep_prob_list,
                 'activation_list': activation_list,
                 'channel_list': channel_list
                 }

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
model.save_weights(run_params['result_path'])
