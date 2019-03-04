import tensorflow as tf
# tf.enable_eager_execution()
import numpy as np
import os
import argparse
from shutil import copy2
from proto import tooth_pb2
from google.protobuf import text_format
import tensorflow.contrib.slim as slim

# import cv2

################# Import Section
numdegree = 4  # Number of rotations
image_height = 240  # Used for cropping
image_width = 360  # Used for cropping

parser = argparse.ArgumentParser()
parser.add_argument('config', help="Config directory")
# parser.add_argument('--config',help="Config directory")
args = parser.parse_args()


# Changed 'google.protobuf.pyext._message.RepeatedScalarContainer' to list
# options is dict which the repeated_scalar contain index of
def protobuf_to_list(repeated_scalar, options=None):
    output = list()
    for elem in repeated_scalar:
        if options is None:
            output.append(elem)
        else:
            output.append(options[elem])
    return output


# Specifically set for transforming
def protobuf_to_channels(repeated_channel):
    channel_list = list()
    for channel in repeated_channel:
        ch = list()
        for layers in channel.channel:
            ch.append(layers)
        channel_list.append(ch)
    return channel_list


# Check if parameters exist, if not, give default parameters or raiseError
# params: dictionary, dict_name: string, default: value (if None will raiseError)
def check_exist(params, dict_name, default=None):
    try:
        output = params[dict_name]
    except KeyError:
        if default is None:
            raise Exception("Parameter %s not defined" % dict_name)
        else:
            output = default
            print("Parameters: %s not found, use default value = %s" % (dict_name, default))
    return output


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
    image_decoded = list()

    # Stacking the rest
    for i in range(0, numdegree):
        img = data_dict['img' + str(i)]
        file_decoded = tf.image.decode_png(img, channels=1)
        file_cropped = tf.squeeze(tf.image.resize_image_with_crop_or_pad(file_decoded, image_height, image_width))
        image_decoded.append(file_cropped)

    image_stacked = tf.stack([image_decoded[0], image_decoded[1], image_decoded[2], image_decoded[3]], axis=2)
    image_stacked = tf.cast(image_stacked, tf.float32)
    label = data_dict['label']
    # output = (image_stacked, label)
    # return {'images': image_stacked, 'label': label}  # Output is [Channel, Height, Width]
    return image_stacked, label


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# Default stride of 1, padding:same
def cnn_2d(x,
           conv_filter_size,  # [Scalar]
           num_filters,  # [Scalar]
           activation=tf.nn.relu,
           stride=1,
           name=''):  # Stride of CNN
    # We shall define the weights that will be trained using create_weights function.
    layer = tf.layers.conv2d(x, num_filters, conv_filter_size, strides=(stride, stride), padding="same",
                             activation=activation, name=name)

    # cnn_sum = tf.summary.histogram(name+'_activation',layer)
    return layer
    # TODO: Find way to show weight


def flatten_layer(layer):  # Flatten from 2D/3D to 1D (not count batch dimension)
    layer = tf.layers.flatten(layer)
    return layer


def fc_layer(x,  #
             num_outputs,
             activation=tf.nn.relu,
             name=''):
    # Let's define trainable weights and biases.
    layer = tf.layers.dense(x, num_outputs, activation=activation, name=name)
    return layer


def avg_pool_layer(layer, pooling_size, name=None, stride=-1):
    # Set stride equals to pooling size unless specified
    if stride == -1:
        stride = pooling_size
    return tf.layers.average_pooling2d(layer, pooling_size, stride, padding="same", name=name)


def max_pool_layer(layer, pooling_size, name=None, stride=-1):
    # Set stride equals to pooling size unless specified
    if stride == -1:
        stride = pooling_size
    return tf.layers.max_pooling2d(layer, pooling_size, stride, padding="same", name=name)


def dropout(layer, keep_prob, name):
    return tf.nn.dropout(layer, keep_prob=keep_prob, name=name)


# Define Model
def my_model(features, labels, mode, params, config):
    # TODO: Early stopping
    #

    # Input: (Batch_size,240,360,4)

    # (1) Filter size: 5x5x64
    conv1 = cnn_2d(features, 5, params['channels'][0], activation=params['activation'], name="conv1")
    pool1 = avg_pool_layer(conv1, 4, "pool1")
    # Output: 60x90x64

    # (2) Filter size: 3x3x64
    conv2 = cnn_2d(pool1, 3, params['channels'][1], activation=params['activation'], name="conv2")
    pool2 = avg_pool_layer(conv2, 2, "pool2")
    # Output: 30x45x64

    # (3.1) Max Pool, then Filter size: 64
    pool3_1 = max_pool_layer(pool2, 3, "pool3_1", stride=1)  # Special stride to keep same dimension
    conv3_1 = cnn_2d(pool3_1, 1, params['channels'][2], activation=params['activation'], name="conv3_1")
    # Output: 30x45x64

    # (3.2) Filter size: 1x1x64, then 3x3x64
    conv3_2 = cnn_2d(pool2, 1, params['channels'][3], activation=params['activation'], name="conv3_2_1")
    conv3_2 = cnn_2d(conv3_2, 3, params['channels'][4], activation=params['activation'], name="conv3_2")
    # Output: 30x45x64

    # (3.3) Filter size: 1x1x64, then 5x5x64
    conv3_3 = cnn_2d(pool2, 1, params['channels'][5], activation=params['activation'], name="conv3_3_1")
    conv3_3 = cnn_2d(conv3_3, 3, params['channels'][6], activation=params['activation'], name="conv3_3_2")
    conv3_3 = cnn_2d(conv3_3, 3, params['channels'][7], activation=params['activation'], name="conv3_3_3")
    # conv3_3 = cnn_2d(conv3_3, 5, 64)  # Might use 2 3x3 CNN instead, look at inception net paper
    # Output: 30x45x64

    # (3.4) Filter size: 1x1x256
    conv3_4 = cnn_2d(pool2, 1, params['channels'][8], activation=params['activation'], name="conv3_4")
    # Output: 30x45x64

    concat4 = tf.concat([conv3_1, conv3_2, conv3_3, conv3_4], 3)
    pool4 = avg_pool_layer(concat4, 3, name="pool4")
    # Output: 10x15x256 = 38400

    fc5 = flatten_layer(pool4)
    fc5 = fc_layer(fc5, params['channels'][9], activation=params['activation'], name='fc5')
    dropout5 = dropout(fc5, params['keep_prob'], name="dropout5")

    fc6 = fc_layer(dropout5, params['channels'][10], activation=params['activation'], name='fc6')
    dropout6 = dropout(fc6, params['keep_prob'], name="dropout6")

    logits = fc_layer(dropout6, 11, activation=params['activation'], name='predict')
    # Predict Mode
    predicted_class = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'score': predicted_class[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
            'label': labels
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    accuracy = tf.metrics.accuracy(labels, predicted_class)
    my_accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predicted_class), dtype=tf.float32))  # The one that is not complicated
    acc = tf.summary.scalar("my_accuracy", my_accuracy)
    # acc2 = tf.summary.scalar("Accuracy_update", accuracy[1])

    img1 = tf.summary.image("Input_image1", tf.expand_dims(features[:, :, :, 0], 3))
    img2 = tf.summary.image("Input_image2", tf.expand_dims(features[:, :, :, 1], 3))
    img3 = tf.summary.image("Input_image3", tf.expand_dims(features[:, :, :, 2], 3))
    img4 = tf.summary.image("Input_image4", tf.expand_dims(features[:, :, :, 3], 3))

    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    summary_name = ["conv1", "conv2", "conv3_1", "conv3_2_1", "conv3_2", "conv3_3_1",
                    "conv3_3_2", "conv3_3_3", "conv4", "fc5", "fc6", "predict"]
    if len(summary_name) == int(len(d_vars) / 2):
        for i in range(len(summary_name)):
            tf.summary.histogram(summary_name[i] + "_weights", d_vars[2 * i])
            tf.summary.histogram(summary_name[i] + "_biases", d_vars[2 * i + 1])

    summary = tf.summary.histogram("Prediction", predicted_class)
    summary2 = tf.summary.histogram("Ground_Truth", labels)
    # global_step = tf.summary.scalar("Global steps",tf.train.get_global_step())

    # Train Mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        steps = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(params['learning_rate'], steps,
                                                   20000, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=steps)
        saver_hook = tf.train.SummarySaverHook(save_steps=1000, summary_op=tf.summary.merge_all(),
                                               output_dir=config.model_dir)
        # model_vars = tf.trainable_variables()
        # slim.model_analyzer.analyze_vars(model_vars, print_info=True)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[saver_hook])

    # Evaluate Mode
    saver_hook = tf.train.SummarySaverHook(save_steps=1000, summary_op=tf.summary.merge_all(),
                                           output_dir=config.model_dir + 'eval')
    return tf.estimator.EstimatorSpec(mode=mode, eval_metric_ops={'accuracy': accuracy}, loss=loss,
                                      evaluation_hooks=[saver_hook])


def train_input_fn(data_path, batch_size):
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(deserialize, num_parallel_calls=7)
    dataset = dataset.map(decode, num_parallel_calls=7)
    dataset = dataset.batch(batch_size, drop_remainder=False)  # Maybe batch after repeat?
    dataset = dataset.shuffle(100)
    dataset = dataset.repeat(None)
    dataset = dataset.prefetch(buffer_size=None)
    return dataset


def eval_input_fn(data_path, batch_size):
    eval_dataset = tf.data.TFRecordDataset(data_path)
    eval_dataset = eval_dataset.map(deserialize)
    eval_dataset = eval_dataset.map(decode)
    eval_dataset = eval_dataset.batch(batch_size, drop_remainder=False)  # No need to shuffle this time
    return eval_dataset


def run(run_params=None, model_params=None):
    # Add exception in case params is missing
    run_params['batch_size'] = check_exist(run_params, 'batch_size', 16)
    run_params['checkpoint_min'] = check_exist(run_params, 'checkpoint_min', 10)
    run_params['early_stop_step'] = check_exist(run_params, 'early_stop_step', 5000)
    run_params['input_path'] = check_exist(run_params, 'input_path')
    run_params['result_path'] = check_exist(run_params, 'result_path')
    run_params['steps'] = check_exist(run_params, 'steps')
    model_params['learning_rate'] = check_exist(model_params, 'learning_rate', 0.0003)
    model_params['keep_prob'] = check_exist(model_params, 'keep_prob', 1)
    model_params['activation'] = check_exist(model_params, 'activation', tf.nn.relu)
    model_params['channels'] = check_exist(model_params, 'channel', [64, 64, 128, 64, 64, 64, 64, 64, 64, 2048, 2048])
    if len(model_params['channels']) != 11:
        raise Exception("Number of channels not correspond to number of layers [Need size of 11, got %s]"
                        % len(model_params['channels']))

    # Type in file name
    train_data_path = run_params['input_path'].replace('.tfrecords', '') + '_train.tfrecords'
    eval_data_path = run_params['input_path'].replace('.tfrecords', '') + '_eval.tfrecords'
    print("Getting training data from %s" % train_data_path)
    print("Saved model at %s" % run_params['result_path'])

    tf.logging.set_verbosity(tf.logging.INFO)  # To see some additional info
    # Setting checkpoint config
    my_checkpoint_config = tf.estimator.RunConfig(
        save_checkpoints_secs=run_params['checkpoint_min'] * 60,
        # save_summary_steps=params['checkpoint_min'] * 10,
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
        model_dir=run_params['result_path'],
        config=my_checkpoint_config
    )
    train_hook = tf.contrib.estimator.stop_if_no_decrease_hook(classifier, "loss", run_params['early_stop_step'])
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_input_fn(train_data_path, batch_size=run_params['batch_size']),
        max_steps=run_params['steps'], hooks=[train_hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(eval_data_path, batch_size=run_params['batch_size']), steps=None,
                                      start_delay_secs=0, throttle_secs=0)
    # classifier.train(input_fn=lambda: train_input_fn(train_data_path, batch_size=params['batch_size']),
    #     max_steps=params['steps'], hooks=[train_hook])
    # eval_result = classifier.evaluate(input_fn=lambda: eval_input_fn(eval_data_path, batch_size=32))

    eval_result = tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    # Copy config file to result path as well
    copy2(run_params['config_path'], run_params['result_path'])
    print("Eval result:")
    print(eval_result)
    # print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


# Run with multiple parameters
def run_multiple_params(run_config, model_config):
    for lr in model_config['learning_rate_list']:
        for dr in model_config['keep_prob_list']:
            for act_count, act in enumerate(model_config['activation_list']):
                for ch_count, ch in enumerate(model_config['channel_list']):
                    name = ("%s/learning_rate_%s_dropout_%s_activation_%s_channels_%s/"
                            % (run_config['result_path'], round(lr, 5), dr, act_count, ch_count))
                    md_config = {'learning_rate': round(lr, 5),
                                 'keep_prob': dr,
                                 'activation': act,
                                 'channel': ch}
                    rn_config = run_config.copy()
                    rn_config['result_path'] = name
                    run(rn_config, md_config)


if __name__ == '__main__':
    activation_dict = {'1': tf.nn.relu}
    config = tooth_pb2.TrainConfig()
    with open(args.config, 'r') as f:
        text_format.Merge(f.read(), config)

    # Convert protobuf data to list
    learning_rate_list = protobuf_to_list(config.learning_rate)
    keep_prob_list = protobuf_to_list(config.keep_prob)
    activation_list = protobuf_to_list(config.activation, activation_dict)
    channel_list = protobuf_to_channels(config.channels)

    run_configs = {'batch_size': config.batch_size,
                   'checkpoint_min': config.checkpoint_min,
                   'early_stop_step': config.early_stop_step,
                   'input_path': config.input_path,
                   'result_path': config.result_path,
                   'config_path': os.path.abspath(args.config),
                   'steps': config.steps}
    model_configs = {'learning_rate_list': learning_rate_list,
                     'keep_prob_list': keep_prob_list,
                     'activation_list': activation_list,
                     'channel_list': channel_list
                     }
    # run()
    run_multiple_params(run_configs, model_configs)
    print("train.py completed")

####################################
