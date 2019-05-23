import tensorflow as tf
import csv
import os

tf.enable_eager_execution()
import numpy as np
from custom_hook import EvalResultHook

# In case of needing l2-regularization: https://stackoverflow.com/questions/44232566/add-l2-regularization-when-using-high-level-tf-layers/44238354#44238354


# Default stride of 1, padding:same
def cnn_1d(layer,
           conv_filter_size,  # [Scalar]
           num_filters,  # [Scalar]
           activation=tf.nn.relu,
           stride=1,
           padding='same',
           name=''):  # Stride of CNN
    # We shall define the weights that will be trained using create_weights function.
    layer = tf.keras.layers.Conv1D(num_filters, conv_filter_size, strides=stride, padding=padding,
                                   activation=activation)(layer)

    # cnn_sum = tf.summary.histogram(name+'_activation',layer)
    return layer


def max_pool_layer_1d(layer, pooling_size, name=None, stride=-1):
    # Set stride equals to pooling size unless specified
    if stride == -1:
        stride = pooling_size
    return tf.keras.layers.MaxPool1D(pooling_size, stride, padding="same")(layer)


# Default stride of 1, padding:same
def cnn_2d(layer,
           conv_filter_size,  # [Scalar]
           num_filters,  # [Scalar]
           activation=tf.nn.relu,
           stride=1,
           padding='valid',
           name=''):  # Stride of CNN
    # We shall define the weights that will be trained using create_weights function.
    layer = tf.keras.layers.Conv1D(num_filters, kernel_size=conv_filter_size, strides=stride, padding=padding,
                                   activation=activation)(layer)

    # cnn_sum = tf.summary.histogram(name+'_activation',layer)
    return layer


def flatten_layer(layer):  # Flatten from 2D/3D to 1D (not count batch dimension)
    layer = tf.keras.layers.Flatten()(layer)
    return layer


def fc_layer(layer,  #
             num_outputs,
             activation=tf.nn.relu,
             name=''):
    # Let's define trainable weights and biases.
    layer = tf.keras.layers.Dense(num_outputs, activation=activation)(layer)
    return layer


def avg_pool_layer(layer, pooling_size, name=None, stride=-1):
    # Set stride equals to pooling size unless specified
    if stride == -1:
        stride = pooling_size
    return tf.keras.layers.AveragePooling2D(pooling_size, stride, padding="same")(layer)


def max_pool_layer(layer, pooling_size, name=None, stride=-1):
    # Set stride equals to pooling size unless specified
    if stride == -1:
        stride = pooling_size
    return tf.keras.layers.MaxPooling2D(pooling_size, stride, padding="same")(layer)


def max_and_cnn_layer(layer, pl_size, num_filters, activation, name):
    pool = tf.keras.layers.MaxPooling2D(pl_size, strides=pl_size, padding="same")(layer)
    conv = tf.keras.layers.Conv2D(num_filters, pl_size, strides=pl_size, padding="same",
                                  activation=activation)(layer)
    concat = tf.keras.layers.concatenate([pool, conv], 3)
    return concat


'''
def dropout(layer, dropout_rate, training, name):
    return tf.layers.dropout(layer, rate=dropout_rate, training=training, name=name)
'''


# Using average pooling
def standard_fc(features, mode, params):
    if len(params['channels']) != 1:
        raise ValueError("This model need 1 channels input, current input: %s" % params['channels'])
    layer1 = fc_layer(features, num_outputs=params['channels'][0]*64, activation=params['activation'])
    layer2 = fc_layer(layer1, num_outputs=params['channels'][0]*128, activation=params['activation'])
    dropout2 = tf.keras.layers.Dropout(rate=params['dropout_rate'])(layer2)
    layer3 = fc_layer(dropout2, num_outputs=params['channels'][0]*64, activation=params['activation'])
    dropout3 = tf.keras.layers.Dropout(rate=params['dropout_rate'])(layer3)
    logits = fc_layer(dropout3, 3, activation=tf.sigmoid, name='predict')
    return logits


# Using max pooling
def model_cnn_1d(features, mode, params):
    if len(params['channels']) != 2:
        raise ValueError("This model need 1 channels input, current input: %s" % params['channels'])

    # Input size:300x8
    '''
    This model is based on "A Comparison of 1-D and 2-D Deep Convolutional Neural Networks in ECG Classification"
    '''
    # (1) Filter size: 7x32, max pooling of k3 s2
    print(params)
    conv1 = cnn_1d(features, 7, params['channels'][0] * 16, activation=params['activation'], name="conv1")
    pool1 = max_pool_layer_1d(conv1, 3, name="pool1", stride=2)
    # Output: 294x32 -> 147x32

    # (2) Filter size: 5x64, max pooling of k3 s2
    conv2 = cnn_1d(pool1, 5, params['channels'][0] * 32, activation=params['activation'], name="conv2")
    pool2 = max_pool_layer_1d(conv2, 3, "pool2", stride=2)
    # Output: 143x64 -> 71x64

    # (3) Filter size: 3x128 (3 times), max pooling of k3 s2
    conv3 = cnn_1d(pool2, 3, params['channels'][0] * 64, activation=params['activation'], name="conv3_1")
    conv3 = cnn_1d(conv3, 3, params['channels'][0] * 64, activation=params['activation'], name="conv3_2")
    conv3 = cnn_1d(conv3, 3, params['channels'][0] * 64, activation=params['activation'], name="conv3_3")
    pool3 = max_pool_layer_1d(conv3, 3, "pool2", stride=2)
    print(pool3)
    # Output: 65x128 -> 32x128 = 4096

    fc4 = flatten_layer(pool3)
    fc4 = fc_layer(fc4, params['channels'][1] * 1024, activation=params['activation'], name='fc5')
    dropout4 = tf.keras.layers.Dropout(rate=params['dropout_rate'])(fc4)
    # Output: 4096 -> 4096 -> 3

    fc5 = fc_layer(dropout4, params['channels'][1] * 1024, activation=params['activation'], name='fc6')
    dropout5 = tf.keras.layers.Dropout(rate=params['dropout_rate'])(fc5)

    logits = fc_layer(dropout5, 3, activation=tf.nn.tanh, name='predict')
    return logits


# Define Model
def my_model(features, labels, mode, params, config):
    # Input: (Batch_size,240,360,4)
    if params['model_num'] == 0:
        logits = standard_fc(features, mode, params)
    else:
        logits = model_cnn_1d(features, mode, params)
    # Predict Mode
    predicted_class = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'score': predicted_class[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    one_hot_label = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
    labels = tf.cast((labels - 1) / 2, tf.int64)

    weight = tf.constant([[params['loss_weight'][0], params['loss_weight'][1], params['loss_weight'][2]]],
                         dtype=tf.float32)
    loss_weight = tf.matmul(one_hot_label, weight, transpose_b=True, a_is_sparse=True)

    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits,
                                                  weights=loss_weight)  # labels is int of class, logits is vector

    accuracy = tf.metrics.accuracy(labels, predicted_class)

    my_accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predicted_class), dtype=tf.float32))
    acc = tf.summary.scalar("accuracy_manual", my_accuracy)  # Number of correct answer
    # acc2 = tf.summary.scalar("Accuracy_update", accuracy[1])

    # img1 = tf.summary.image("Input_image1", tf.expand_dims(features[:, :, :, 0], 3))
    # img2 = tf.summary.image("Input_image2", tf.expand_dims(features[:, :, :, 1], 3))
    # img3 = tf.summary.image("Input_image3", tf.expand_dims(features[:, :, :, 2], 3))
    # img4 = tf.summary.image("Input_image4", tf.expand_dims(features[:, :, :, 3], 3))

    ex_prediction = tf.summary.scalar("example_prediction", predicted_class[0])
    # print(predicted_class[0])
    ex_ground_truth = tf.summary.scalar("example_ground_truth", labels[0])
    # print(labels[0])

    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # print(d_vars)
    if params['model_num'] == 0:
        summary_name = ["fc1", "fc2", "fc3", "predict"]
    else:
        summary_name = ["conv1", "conv2", "conv3_1", "conv3_2", "conv3_3", "fc4",
                        "fc5", "predict"]
    print(d_vars)
    if len(summary_name) == int(len(d_vars) / 2):
        for i in range(len(summary_name)):
            tf.summary.histogram(summary_name[i] + "_weights", d_vars[2 * i])
            tf.summary.histogram(summary_name[i] + "_biases", d_vars[2 * i + 1])
    else:
        print("Warning, expected weight&variable not equals: amount of var = %s" % len(d_vars))
        print(d_vars)

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

    # Create result(.csv) file, if not exist
    if not os.path.isfile(params['result_path']):
        with open(params['result_path'] + params['result_file_name'], "w") as csvfile:
            fieldnames = ['Label', 'Predicted Class', 'Confident level']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Create hooks
    saver_hook = tf.train.SummarySaverHook(save_steps=10, summary_op=tf.summary.merge_all(),
                                           output_dir=config.model_dir + 'eval')
    csv_name = tf.convert_to_tensor(params['result_path'] + params['result_file_name'], dtype=tf.string)
    eval_hook = EvalResultHook(labels, predicted_class, tf.nn.softmax(logits), csv_name)
    return tf.estimator.EstimatorSpec(mode=mode, eval_metric_ops={'accuracy': accuracy}, loss=loss,
                                      evaluation_hooks=[saver_hook, eval_hook])

