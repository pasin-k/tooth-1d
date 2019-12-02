import tensorflow as tf
from tensorflow.python.client import device_lib
import os
import glob
import argparse
import json
import csv
import datetime

from proto import tooth_pb2
from google.protobuf import text_format

import skopt
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

# from model import my_model
from coor_model import my_model
from utils.coor_get_data import train_input_fn, eval_input_fn
from utils.open_save_file import read_file, save_file, check_exist

# Read tooth.config file
parser = argparse.ArgumentParser()
parser.add_argument('config', help="Directory of config file")
args = parser.parse_args()

configs = tooth_pb2.TrainConfig()
with open(args.config, 'r') as f:
    text_format.Merge(f.read(), configs)

# Settings used in run
run_configs = {'batch_size': configs.batch_size,
               'checkpoint_min': configs.checkpoint_min,
               'early_stop_step': configs.early_stop_step,
               'input_path': configs.input_path,
               'result_path_base': configs.result_path,
               'config_path': os.path.abspath(args.config),
               'steps': configs.steps,
               'label_type': configs.label_type,
               'comment': configs.comment}

# Change folder of input_path into a filename
assert os.path.isdir(run_configs['input_path']), \
    "Input path should be folder directory, not file directory %s" % run_configs['input_path']
run_configs['input_path'] = os.path.join(run_configs['input_path'], os.path.basename(run_configs['input_path']))

run_mode = configs.run_mode  # Run mode (Single, search, etc.)

run_configs = check_exist(run_configs, batch_size=None, checkpoint_min=10, early_stop_step=5000,
                          input_path=None, result_path_base=None, steps=None, config_path=None)

channels_full = [configs.channels, 2]

activation_dict = {'0': tf.nn.relu, '1': tf.nn.leaky_relu}


# For multi-gpu
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_time_and_date(use_current_time):
    if use_current_time:
        return datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    else:
        return "file"

def run(model_params):
    print("Beginning run..")
    # Check if all values exist
    model_params = check_exist(model_params, learning_rate=None, dropout_rate=None, activation=None,
                               channels=None, result_path=None)
    # Note on some model_params:    loss_weight is calculated inside
    #                               channels (in CNN case) is [CNN channels, Dense channels]

    # Type in file name
    train_data_path = run_configs['input_path'] + '_train.tfrecords'
    eval_data_path = run_configs['input_path'] + '_eval.tfrecords'
    info_path = run_configs['input_path'] + '.json'

    # Add some more parameters from config file
    with open(info_path) as filehandle:
        tfrecord_info = json.load(filehandle)
        model_params['loss_weight'] = tfrecord_info['class_weight'][run_configs['label_type']]
        model_params["data_degree"] = tfrecord_info['data_degree']
        model_params["data_length"] = tfrecord_info['data_length']
        model_params["dataset_name"] = tfrecord_info['dataset_name']
        try:
            run_configs["dataset_timestamp"] = tfrecord_info['data_timestamp']
        except KeyError:
            run_configs["dataset_timestamp"] = "None"

    assert len(model_params['loss_weight']) == 3, "Label does not have 3 unique value, found %s" % len(
        model_params['loss_weight'])
    model_params['label_type'] = run_configs['label_type']

    print("Getting training data from %s" % train_data_path)
    print("Saved model at %s" % model_params['result_path'])

    # tf.logging.set_verbosity(tf.logging.INFO)  # To see some additional info
    # Setting for multiple GPUs
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=get_available_gpus())
    # Setting checkpoint config
    my_checkpoint_config = tf.estimator.RunConfig(
        save_checkpoints_secs=run_configs['checkpoint_min'] * 60,
        # save_checkpoints_steps=100,
        keep_checkpoint_max=10,
        log_step_count_steps=500,
        session_config=tf.ConfigProto(allow_soft_placement=True),
        train_distribute=mirrored_strategy
    )

    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params=model_params,
        model_dir=model_params['result_path'],
        config=my_checkpoint_config
    )

    train_hook = tf.contrib.estimator.stop_if_no_decrease_hook(classifier, "loss", run_configs['early_stop_step'])

    # Fetch training_data and evaluation
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_input_fn(train_data_path, batch_size=run_configs['batch_size'], configs=model_params),
        max_steps=run_configs['steps'], hooks=[train_hook])
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: eval_input_fn(eval_data_path, batch_size=32, configs=model_params),
        steps=None, throttle_secs=0)
    # classifier.train(input_fn=lambda: train_input_fn(train_data_path, batch_size=params['batch_size']),
    #     max_steps=params['steps'], hooks=[train_hook])
    # eval_result = classifier.evaluate(input_fn=lambda: eval_input_fn(eval_data_path, batch_size=32))

    # Train and evaluate
    eval_result = tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    # Show result
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
        model_dir=model_params['result_path'],
        config=my_checkpoint_config
    )

    eval_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn(train_data_path, batch_size=run_configs['batch_size'], configs=model_params))

    model_params['result_file_name'] = 'result.csv'

    # Save necessary info to csv file, as reference
    info_dict = run_configs.copy()
    info_dict['accuracy'] = accuracy
    info_dict['learning_rate'] = model_params['learning_rate']
    info_dict['dropout_rate'] = model_params['dropout_rate']
    info_dict['activation'] = model_params['activation']
    info_dict['channels'] = model_params['channels']
    info_dict['loss_weight'] = model_params['loss_weight']
    info_dict['steps'] = global_step
    info_dict['data_length'] = model_params['data_length']
    info_dict['data_degree'] = model_params['data_degree']

    # Save information in config.csv
    with open(os.path.join(model_params['result_path'], "config.csv"), "w") as csvfile:
        writer = csv.writer(csvfile)
        for key, val in info_dict.items():
            writer.writerow([key, val])
    print("Run completed, finished saving csv file ")
    return accuracy, global_step


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
#                     model_params['result_path'] = name
#                     run(md_config)
#                     # Copy config file to result path as well
#                     copy2(run_params['config_path'], model_params['result_path'])


dim_learning_rate = Real(low=5e-5, high=5e-2, prior='log-uniform', name='learning_rate')
dim_dropout_rate = Real(low=0, high=0.875, name='dropout_rate')
dim_activation = Categorical(categories=['0', '1'],
                             name='activation')
dim_channel = Integer(low=1, high=3, name='channels')
dim_loss_weight = Real(low=0.8, high=2, name='loss_weight')
dimensions = [dim_learning_rate,
              dim_dropout_rate,
              dim_activation,
              dim_channel]
default_parameters = [configs.learning_rate, configs.dropout_rate, configs.activation, configs.channels]


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, dropout_rate, activation, channels):
    """
    Hyper-parameters search
    learning_rate:     Learning-rate for the optimizer.
    dropout_rate:      Dropout rate
    activation:        Activation function for all layers.
    channels           Amount of channel
    """
    # Create the neural network with these hyper-parameters
    print("Learning_rate, Dropout_rate, Activation, Channels = %s, %s, %s, %s" % (
        learning_rate, dropout_rate, activation, channels))

    run_configs['current_time'] = get_time_and_date(True)
    # Set result path combine with current time of running
    md_config = {'learning_rate': learning_rate,
                 'dropout_rate': dropout_rate,
                 'activation': activation_dict[activation],
                 'channels': [channels, 2],
                 'result_path': os.path.join(run_configs['result_path_base'], run_configs['current_time']),
                 'result_file_name': 'result.csv',
                 }

    accuracy, global_step = run(md_config)
    # Save info of hyperparameter search in a specific csv file
    save_file(run_configs['summary_file_path'], [accuracy, learning_rate, dropout_rate, activation,
                                                 channels, run_configs['current_time']], write_mode='a',
              data_format="one_row")
    return -accuracy


def run_hyper_parameter_optimize():
    # Name of the summary result from hyperparameter search (This variable is not used in run)
    current_time = get_time_and_date(configs.use_current_time)
    run_configs['summary_file_path'] = os.path.join(run_configs[
                                                        'result_path_base'],
                                                    "hyperparameters_result_" + current_time + ".csv")

    field_name = [i.name for i in dimensions]
    field_name.insert(0, 'accuracy')
    field_name.append('timestamp')

    n_calls = 20  # Expected number of trainings

    # Check if there is previous unfinished file or not. If exist, continue training for last model
    previous_record_files = []
    for file in glob.glob(os.path.join(run_configs['result_path_base'], "hyperparameters_result_" + '*')):
        previous_record_files.append(file)
    previous_record_files.sort()
    if len(previous_record_files) > 0:  # Check if file has previous result
        if not os.stat(previous_record_files[-1]).st_size == 0:  # Check if previous file is empty or not
            prev_data, header = read_file(previous_record_files[-1], header=True)
            if not prev_data:
                prev_data = [['']]
            try:
                # Resume only if run_mode is single
                if prev_data[-1][0] != 'end' and run_mode == "single":
                    n_calls = n_calls - len(prev_data)
                    l_data = prev_data[-1][1:]  # Latest_data
                    default_param = [float(l_data[0]), float(l_data[1]), l_data[2], int(l_data[3])]
                    run_configs['summary_file_path'] = previous_record_files[-1]
                    current_time = previous_record_files[-1].split("/")[-1].replace('.csv', '').replace(
                        "hyperparameters_result_", '')
                    print("Continue from %s" % current_time)
                else:  # Otherwise, create new file
                    save_file(run_configs['summary_file_path'], [], field_name=field_name,
                              write_mode='w', create_folder=True, data_format="header_only")  # Create new summary file
                    default_param = default_parameters
                    print("Creating new runs")
            except IndexError:  # If error, stop and end the file, usually occur when the first run is interrupted
                save_file(previous_record_files[-1],
                          ['end', 'Error from previous run', get_time_and_date(configs.use_current_time)],
                          write_mode='a', data_format="one_row")
                raise ValueError("Previous file doesn't end completely")
        else:
            os.remove(previous_record_files[-1])  # Delete empty file
            save_file(run_configs['summary_file_path'], [], field_name=field_name, write_mode='w',
                      create_folder=True)  # Create new summary file
            default_param = default_parameters
            print("Creating new runs, deleting previous empty file")
    else:  # If no file in folder, create new file
        save_file(run_configs['summary_file_path'], [], field_name=field_name, write_mode='w',
                  create_folder=True)  # Create new summary file
        default_param = default_parameters
        print("Creating new runs (new folder)")

    print("Saving hyperparameters_result in %s" % run_configs['summary_file_path'])
    print("Running remaining: %s time" % n_calls)

    best_accuracy = 0
    if n_calls < 11:
        print("Hyper parameter optimize ENDED: run enough calls already")
        save_file(run_configs['summary_file_path'],
                  ['end', 'Completed (faster than expected)', get_time_and_date(configs.use_current_time)],
                  write_mode='a', data_format="one_row")
    else:
        # Start hyperparameter search. Save each file in seperate folder
        run_configs['result_path_base'] = os.path.join(run_configs['result_path_base'], current_time)
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
            if i[0] > best_accuracy:
                best_accuracy = i[0]
            data = {field_name[0]: i[0] * -1}
            for j in range(1, len(field_name) - 1):
                data[field_name[1]] = i[1][j - 1]
            new_data.append(data)

        save_file(run_configs['summary_file_path'],
                  ['end', 'Completed', get_time_and_date(configs.use_current_time)],
                  write_mode='a', data_format="one_row")
    print("Saving hyperparameters_result in %s" % run_configs['summary_file_path'])
    return best_accuracy


def run_kfold(model_params, k_num=5):
    base_input_path = run_configs['input_path']
    result_path = run_configs['result_path']
    kfold_path = os.path.join(run_configs['result_path'], "kfold.csv")
    if os.path.exists(kfold_path):
        data = read_file(kfold_path)
        current_run = len(data)
    else:
        current_run = 0
    all_accuracy = []
    for i in range(current_run, k_num):
        run_configs['input_path'] = base_input_path + ("_%s" % i)
        run_configs['result_path'] = os.path.join(result_path, str(i))
        model_params['result_path'] = run_configs['result_path']
        accuracy, _ = run(model_params)
        save_file(kfold_path, ["%s_%s" % (i, accuracy)], write_mode='a')
        all_accuracy.append(accuracy)
    print(all_accuracy)


def run_hyper_parameter_optimize_kfold(k_num=5):
    base_input_path = run_configs['input_path']
    result_path_base = run_configs['result_path_base']
    all_accuracy = []
    for i in range(k_num):
        run_configs['input_path'] = base_input_path + ("_%s" % i)
        run_configs['result_path_base'] = os.path.join(result_path_base, str(i))
        accuracy = run_hyper_parameter_optimize()
        all_accuracy.append(accuracy)
    print(all_accuracy)


model_configs = {'learning_rate': configs.learning_rate,
                 'dropout_rate': configs.dropout_rate,
                 'activation': activation_dict[configs.activation],
                 'channels': channels_full,
                 }

if __name__ == '__main__':
    if run_mode == "single":
        run_configs['input_path'] = run_configs['input_path'] + "_0"
        model_configs['result_file_name'] = 'result.csv'
        model_configs['result_path'] = os.path.join(run_configs['result_path_base'],
                                                    get_time_and_date(configs.use_current_time))
        run(model_configs)
    elif run_mode == "kfold":
        model_configs['result_file_name'] = 'result.csv'
        model_configs['result_path'] = run_configs['result_path']
        run_kfold(model_configs)
    elif run_mode == "search":
        run_configs['input_path'] = run_configs['input_path'] + "_0"
        run_hyper_parameter_optimize()
    elif run_mode == "kfold_search":
        run_hyper_parameter_optimize_kfold()
    else:
        raise ValueError('run_mode invalid. Expect "single", "kfold", "search", "kfold_search". Found %s' % run_mode)
    print("train.py completed")
