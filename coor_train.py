import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import os
import glob
import argparse
from shutil import copy2
import csv
import datetime

from proto import tooth_pb2
from google.protobuf import text_format

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

# from model import my_model
from coor_model import my_model
from coor_get_data import train_input_fn, eval_input_fn, get_data_from_path
from open_save_file import read_file, save_file

# Read tooth.config file
parser = argparse.ArgumentParser()
parser.add_argument('config', help="Directory of config file")
# parser.add_argument('--config',help="Config directory")
args = parser.parse_args()

activation_dict = {'0': tf.nn.relu, '1': tf.nn.leaky_relu}  # Declare global dictionary
configs = tooth_pb2.TrainConfig()
with open(args.config, 'r') as f:
    text_format.Merge(f.read(), configs)

'''
# Convert protobuf data to list
learning_rate_list = protobuf_to_list(configs.learning_rate)
dropout_rate_list = protobuf_to_list(configs.dropout_rate)
activation_list = protobuf_to_list(configs.activation, activation_dict)
channel_list = protobuf_to_channels(configs.channels)
'''


# Check if parameters exist, if not, give default parameters or raiseError
# params: dictionary, dict_name: string, default: value (if None will raiseError)
def check_exist(dictionary, **kwargs):
    output_dict = dictionary
    for key, value in kwargs.items():
        try:
            output_dict[key] = dictionary[key]
            # output = params[dict_name]
        except (KeyError, TypeError) as error:
            if value is None:
                raise KeyError("Parameter '%s' not defined" % key)
            else:
                output_dict[key] = value
                print("Parameters: %s not found, use default value = %s" % (key, value))
    return output_dict


# These are important parameters
run_params = {'batch_size': configs.batch_size,
              'checkpoint_min': configs.checkpoint_min,
              'early_stop_step': configs.early_stop_step,
              'input_path': configs.input_path,
              'result_path_base': configs.result_path,
              'config_path': os.path.abspath(args.config),
              'steps': configs.steps,
              'comment': configs.comment}

model_num = configs.loss_weight  # Borrowed parameter name 0 -> dense, 1 -> 1dCNN

run_params = check_exist(run_params, batch_size=None,
                         checkpoint_min=10, early_stop_step=5000,
                         input_path=None, result_path_base=None,
                         steps=None, config_path=None)

model_configs = {'learning_rate': configs.learning_rate,
                 'dropout_rate': configs.dropout_rate,
                 'activation': activation_dict[configs.activation],
                 'channels': configs.channels * [16, 16, 32, 16, 16, 16, 16, 16, 16, 512, 512],
                 }

# Final Parameters:
# run_params: batch_size, checkpoint_min, early_stop_step, input_path, result_path_base, config_path, steps, comment
#             result_path(result_path_base + data&time), summary_file_path(doesn't use in run, just a global param)

# model_params: learning_rate, dropout_rate, activation, channels, loss_weight,
#               result_path(same as in run_params), result_file_name,
#               model_num(Special params) -> Use to determine which model to run (1dCNN or dense)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


class_value = ['1', '3', '5']


def run(model_params=None):
    if model_params is None:
        raise ValueError("No model_params found")
    # Check if all values exist
    model_params = check_exist(model_params, learning_rate=None,
                               dropout_rate=None, activation=None,
                               channels=None, result_path=None)
    # Note on some model_params:    loss_weight is calculated inside
    #                               channels (in CNN case) is [CNN channels, Dense channels]

    # Type in file name
    train_data_path = run_params['input_path'].replace('.tfrecords', '') + '_train.tfrecords'
    eval_data_path = run_params['input_path'].replace('.tfrecords', '') + '_eval.tfrecords'
    info_path = run_params['input_path'].replace('.tfrecords', '.txt')
    label_hist = dict()
    with open(info_path) as f:
        filehandle = f.read().splitlines()
        if not (filehandle[0] == 'distribution'):
            print(filehandle[0])
            raise KeyError("File does not have correct format, need 'train' and 'eval' keyword within file")
        for line in filehandle[1:]:
            if line == 'train':
                break
            else:
                [key, val] = line.split('_')
                label_hist[key] = int(val)
    total = label_hist['1'] + label_hist['3'] + label_hist['5']
    # model_params['loss_weight'] = [1, 1, 1] # Custom value
    # model_params['loss_weight'] = [3, 1, 1.8] # Only for BL_361 data: real ratio (22:1:1.8)
    # model_params['loss_weight'] = [5, 1.43, 1] # Only for MD_361 data: real ratio (43:1.43:1)
    model_params['loss_weight'] = [total / label_hist['1'], total / label_hist['3'], total / label_hist['5']]
    print("Getting training data from %s" % train_data_path)
    print("Saved model at %s" % run_params['result_path'])

    tf.logging.set_verbosity(tf.logging.INFO)  # To see some additional info
    # Setting for multiple GPUs
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=get_available_gpus())
    # Setting checkpoint config
    my_checkpoint_config = tf.estimator.RunConfig(
        save_checkpoints_secs=run_params['checkpoint_min'] * 60,
        # save_summary_steps=params['checkpoint_min'] * 10,
        keep_checkpoint_max=10,
        log_step_count_steps=500,
        session_config=tf.ConfigProto(allow_soft_placement=True),
        train_distribute=mirrored_strategy
    )

    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params=model_params,
        model_dir=run_params['result_path'],
        config=my_checkpoint_config
    )

    if model_num == 0:
        data_type = 0  # 0 is getting vectorized data, used with Dense model
    else:
        data_type = 1  # 1 is getting stacked data, used with 1d CNN model

    train_hook = tf.contrib.estimator.stop_if_no_decrease_hook(classifier, "loss", run_params['early_stop_step'])
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_input_fn(train_data_path, batch_size=run_params['batch_size'], data_type=data_type),
        max_steps=run_params['steps'], hooks=[train_hook])
    # TODO: Evaluate only once? why?
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: eval_input_fn(eval_data_path, batch_size=run_params['batch_size'], data_type=data_type), steps=100,
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

    # Evaluate using train set
    model_params['result_file_name'] = 'train_result.csv'
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params=model_params,
        model_dir=run_params['result_path'],
        config=my_checkpoint_config
    )

    eval_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn(train_data_path, batch_size=run_params['batch_size'], data_type=data_type))

    # No need to use predict since we can get data from hook now
    # # images, expected = get_data_from_path(eval_data_path)
    # score_address = run_params['input_path'].replace('.tfrecords', '') + '_score.npy'
    # print(score_address)
    # expected = np.load(score_address)
    # print(expected)
    # predictions = classifier.predict(input_fn=lambda: eval_input_fn(eval_data_path, batch_size=1))
    #
    # predict_score = ['Prediction']
    # label_score = ['Label']
    # for pred_dict, expec in zip(predictions, expected):
    #     # print(pred_dict)
    #     # print("Score: " + str(pred_dict['score']))
    #     class_id = pred_dict['score']
    #     # probability = pred_dict['probabilities'][class_id]
    #     # print("Actual score: %s, Predicted score: %s " % (expec, class_id))
    #     predict_score.append(class_id)
    #     label_score.append(expec)
    #
    # predict_result = zip(label_score, predict_score)

    # print(eval_result[0]['accuracy'])
    # print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    predict_result = None

    # Save necessary info to csv file, as reference
    info_dict = run_params.copy()
    info_dict['learning_rate'] = model_params['learning_rate']
    info_dict['dropout_rate'] = model_params['dropout_rate']
    info_dict['activation'] = model_params['activation']
    info_dict['channels'] = model_params['channels']
    info_dict['loss_weight'] = model_params['loss_weight']
    info_dict['steps'] = global_step
    info_dict['accuracy'] = accuracy
    with open((run_params['result_path'] + "config.csv"), "w") as csvfile:
        writer = csv.writer(csvfile)
        for key, val in info_dict.items():
            writer.writerow([key, val])

    return accuracy, global_step, predict_result


# # Run with multiple parameters (Grid-search)
# def run_multiple_params(model_config):
#     for lr in model_config['learning_rate_list']:
#         for dr in model_config['dropout_rate_list']:
#             for act_count, act in enumerate(model_config['activation_list']):
#                 for ch_count, ch in enumerate(model_config['channel_list']):
#                     name = ("%s/learning_rate_%s_dropout_%s_activation_%s_channels_%s/"
#                             % (run_params['result_path_base'], round(lr, 5), dr, act_count, ch_count))
#                     md_config = {'learning_rate': round(lr, 5),
#                                  'dropout_rate': dr,
#                                  'activation': act,
#                                  'channels': ch}
#                     run_params['result_path'] = name
#                     run(md_config)
#                     # Copy config file to result path as well
#                     copy2(run_params['config_path'], run_params['result_path'])


dim_learning_rate = Real(low=1e-5, high=1e-2, prior='log-uniform', name='learning_rate')
dim_dropout_rate = Real(low=0, high=0.875, name='dropout_rate')
dim_activation = Categorical(categories=['0', '1'],
                             name='activation')
dim_channel = Integer(low=1, high=3, name='channels')
dim_loss_weight = Real(low=0.8, high=2, name='loss_weight')
dimensions = [dim_learning_rate,
              dim_dropout_rate,
              dim_activation,
              dim_channel]
default_parameters = [1e-3, 0.125, '0', 2]


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, dropout_rate, activation, channels):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    dropout_rate:
    activation:        Activation function for all layers.
    channels
    """
    # Create the neural network with these hyper-parameters
    print("Learning_rate, Dropout_rate, Activation, Channels = %s, %s, %s, %s" % (
        learning_rate, dropout_rate, activation, channels))

    # Set result path combine with current time of running
    run_params['result_path'] = run_params['result_path_base'] + "/" + datetime.datetime.now().strftime(
        "%Y%m%d_%H_%M_%S") + "/"

    if model_num == 0:
        channels_full = [channels]
    else:
        channels_full = [channels, 2]
    md_config = {'learning_rate': learning_rate,
                 'dropout_rate': dropout_rate,
                 'activation': activation_dict[activation],
                 'channels': channels_full,
                 'result_path': run_params['result_path'],
                 'result_file_name': 'result.csv',
                 'model_num': model_num}
    accuracy, global_step, result = run(md_config)
    # Save info of hyperparameter search in a specific csv file
    save_file(run_params['summary_file_path'], [accuracy, learning_rate, dropout_rate, activation,
                                                channels], write_mode='a', one_row=True)
    return -accuracy


def run_hyper_parameter_optimize():
    # Name of the summary result from hyperparameter search (This variable is not used in run)
    run_params['summary_file_path'] = run_params['result_path_base'] + '/' + "hyperparameters_result_" \
                                      + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S") + ".csv"
    field_name = [i.name for i in dimensions]
    field_name.insert(0, 'accuracy')

    n_calls = 20  # Expected number of trainings

    previous_record_files = []
    for file in glob.glob(run_params['result_path_base'] + '/' + "hyperparameters_result_" + '*'):
        previous_record_files.append(file)
    previous_record_files.sort()
    if len(previous_record_files) > 0:
        prev_data, header = read_file(previous_record_files[-1], header=True)
        try:
            if prev_data[-1][0] != 'end':  # Check if the previous file doesn't end properly
                n_calls = n_calls - len(prev_data)
                l_data = prev_data[-1][1:]  # Latest_data
                default_param = [float(l_data[0]), float(l_data[1]), l_data[2], int(l_data[3])]
                run_params['summary_file_path'] = previous_record_files[-1]
            else:
                save_file(run_params['summary_file_path'], [], field_name=field_name,
                          write_mode='w', create_folder=True)  # Create new summary file
                default_param = default_parameters
        except IndexError:
            save_file(previous_record_files[-1], ['end', 'Error from previous run', datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")],
                      write_mode='a', one_row=True)
            raise ValueError("Previous file doesn't end completely")
    else:
        save_file(run_params['summary_file_path'], [], field_name=field_name, write_mode='w', create_folder=True)  # Create new summary file
        default_param = default_parameters

    print("Saving hyperparameters_result in %s" % run_params['summary_file_path'])
    print("Running remaining: %s time" % n_calls)
    if n_calls < 11:
        print("Hyper parameter optimize ENDED: run enough calls already")
        save_file(run_params['summary_file_path'], ['end', 'Completed (faster than expected)', datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")],
                  write_mode='a', one_row=True)
    else:
        search_result = gp_minimize(func=fitness,
                                    dimensions=dimensions,
                                    acq_func='EI',  # Expected Improvement.
                                    n_calls=n_calls,
                                    x0=default_param)
        print(search_result)
        print("Best hyper-parameters: %s" % search_result.x)
        searched_parameter = list(
            zip(search_result.func_vals, search_result.x_iters))  # List of tuple of (Acc, [Hyperparams])
        print("All hyper-parameter searched: %s" % searched_parameter)

        new_data = []

        for i in searched_parameter:
            data = {field_name[0]: i[0]* -1}
            for j in range(1, len(field_name)):
                data[field_name[1]] = i[1][j-1]
            # data = {field_name[0]: i[0] * -1,
            #         field_name[1]: i[1][0],
            #         field_name[2]: i[1][1],
            #         field_name[3]: i[1][2],
            #         field_name[4]: i[1][3],
            #         field_name[5]: i[1][4]}
            new_data.append(data)

        save_file(run_params['summary_file_path'], ['end', 'Completed', datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")],
                  write_mode='a', one_row=True)
        # space = search_result.space
        # print("Best result: %s" % space.point_to_dict(search_result.x))


if __name__ == '__main__':
    run_single = False

    if run_single:
        run_params['result_path'] = run_params['result_path_base'] + '/' + datetime.datetime.now().strftime(
            "%Y%m%d_%H_%M_%S") + '/'
        model_configs['result_path'] = run_params['result_path_new']
        model_configs['result_file_name'] = 'result.csv',
        run(model_configs)
    else:
        run_hyper_parameter_optimize()
    print("train.py completed")
