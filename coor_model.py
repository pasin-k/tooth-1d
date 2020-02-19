import tensorflow as tf
import csv
import os
import numpy as np
from utils.custom_hook import EvalResultHook, PrintValueHook

# In case of needing l2-regularization: https://stackoverflow.com/questions/44232566/add-l2-regularization-when-using-high-level-tf-layers/44238354#44238354
initilizer = "he_uniform"


# Default stride of 1, padding:same
def cnn_1d(inp,
           conv_filter_size,  # [Scalar]
           num_filters,  # [Scalar]
           mode,
           activation=tf.nn.relu,
           stride=1,
           padding='valid',
           input_shape=None,
           name='', kernel_regularizer=0.0):  # Stride of CNN
    # We shall define the weights that will be trained using create_weights function.
    if input_shape is None:
        layer = tf.keras.layers.Conv1D(num_filters, conv_filter_size, strides=stride, padding=padding,
                                       activation=activation, kernel_initializer=initilizer,
                                       kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer), name=name)
        output = layer(inp)
    else:
        layer = tf.keras.layers.Conv1D(num_filters, conv_filter_size, strides=stride, padding=padding,
                                       activation=activation, input_shape=input_shape, kernel_initializer=initilizer,
                                       kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer), name=name)
        output = layer(inp)
    return output


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
           name='',
           kernel_regularizer=0.0):  # Stride of CNN
    # We shall define the weights that will be trained using create_weights function.
    layer = tf.keras.layers.Conv1D(num_filters, kernel_size=conv_filter_size, strides=stride, padding=padding,
                                   activation=activation,
                                   kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer))(layer)

    # cnn_sum = tf.summary.histogram(name+'_activation',layer)
    return layer


def flatten_layer(layer):  # Flatten from 2D/3D to 1D (not count batch dimension)
    layer = tf.keras.layers.Flatten()(layer)
    return layer


def fc_layer(inp,  #
             num_outputs,
             mode,
             activation=tf.nn.relu,
             name='',
             kernel_regularizer=0.0):
    # Let's define trainable weights and biases.
    layer = tf.keras.layers.Dense(num_outputs, activation=activation, kernel_initializer=initilizer,
                                  kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer))
    output = layer(inp)
    return output


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


l2_regularizer = 0.001


# Using max pooling
def model_cnn_1d(features, mode, params, config):
    # print(features)
    require_channel = 2
    params['dropout_rate'] = 0.3
    assert len(params['channels']) == require_channel, \
        "This model need {} channels input, current input: {}".format(require_channel, params['channels'])
    # Input size:300x8
    '''
    This model is based on "A Comparison of 1-D and 2-D Deep Convolutional Neural Networks in ECG Classification"
    '''
    # (1) Filter size: 7x32, max pooling of k3 s2
    conv1 = cnn_1d(features['image'], 7, params['channels'][0] * 16,
                   mode=mode,
                   activation=params['activation'],
                   name="conv1",
                   input_shape=(300, 8),
                   kernel_regularizer=l2_regularizer)
    conv1 = tf.layers.batch_normalization(conv1)
    pool1 = max_pool_layer_1d(conv1, 3, name="pool1", stride=2)
    # Output: 294x32 -> 147x32
    # (2) Filter size: 5x64, max pooling of k3 s2
    conv2 = cnn_1d(pool1, 5, params['channels'][0] * 32,
                   mode=mode,
                   activation=params['activation'], name="conv2",
                   kernel_regularizer=l2_regularizer)
    conv2 = tf.layers.batch_normalization(conv2)
    pool2 = max_pool_layer_1d(conv2, 3, "pool2", stride=2)
    # Output: 143x64 -> 71x64

    # (3) Filter size: 3x128 (3 times), max pooling of k3 s2
    conv3 = cnn_1d(pool2, 3, params['channels'][0] * 64,
                   mode=mode,
                   activation=params['activation'], name="conv3",
                   kernel_regularizer=l2_regularizer)
    conv3 = tf.layers.batch_normalization(conv3)
    conv4 = cnn_1d(conv3, 3, params['channels'][0] * 64,
                   mode=mode,
                   activation=params['activation'], name="conv4",
                   kernel_regularizer=l2_regularizer)
    conv4 = tf.layers.batch_normalization(conv4)
    conv5 = cnn_1d(conv4, 3, params['channels'][0] * 64,
                   mode=mode,
                   activation=params['activation'], name="conv5",
                   kernel_regularizer=l2_regularizer)
    conv5 = tf.layers.batch_normalization(conv5)
    pool5 = max_pool_layer_1d(conv5, 3, "pool2", stride=2)
    # Output: 65x128 -> 32x128 = 4096
    fc6 = flatten_layer(pool5)
    fc6 = fc_layer(fc6, params['channels'][1] * 128,  # 1024
                   mode=mode,
                   activation=params['activation'], kernel_regularizer=l2_regularizer,
                   name='fc6', )
    dropout6 = tf.keras.layers.Dropout(rate=params['dropout_rate'])(fc6)
    # Output: 4096 -> 4096 -> 3
    fc7 = fc_layer(dropout6, params['channels'][1] * 64,  # 1024
                   mode=mode,
                   activation=params['activation'], name='fc7',
                   kernel_regularizer=l2_regularizer)
    dropout7 = tf.keras.layers.Dropout(rate=params['dropout_rate'])(fc7)
    logits = fc_layer(dropout7, 3,
                      mode=mode,
                      activation=None, name='predict', kernel_regularizer=l2_regularizer)
    return logits


def model_deep_sleep_net(features, mode, params, config):
    # print(features)
    require_channel = 2
    params['dropout_rate'] = 0.3
    assert len(params['channels']) == require_channel, \
        "This model need {} channels input, current input: {}".format(require_channel, params['channels'])
    lstm_unit = 128
    init = tf.initializers.truncated_normal(stddev=0.1)
    regularizer = tf.nn.l2_loss
    conv1 = cnn_1d(features['image'], 5, params['channels'][0] * 16,
                   mode=mode,
                   activation=params['activation'],
                   name="conv1",
                   input_shape=(300, 8),
                   kernel_regularizer=l2_regularizer)
    conv2 = cnn_1d(conv1, 5, params['channels'][0] * 16,
                   mode=mode,
                   activation=params['activation'], name="conv2",
                   kernel_regularizer=l2_regularizer)
    conv3 = cnn_1d(conv2, 5, params['channels'][0] * 16,
                   mode=mode,
                   activation=params['activation'], name="conv3",
                   kernel_regularizer=l2_regularizer)
    pool3 = max_pool_layer_1d(conv3, 2, name="pool3", stride=-1)
    pool3 = tf.layers.batch_normalization(pool3)

    conv4 = cnn_1d(pool3, 5, params['channels'][0] * 32,
                   mode=mode,
                   activation=params['activation'], name="conv4",
                   kernel_regularizer=l2_regularizer)
    conv5 = cnn_1d(conv4, 5, params['channels'][0] * 32,
                   mode=mode,
                   activation=params['activation'], name="conv5",
                   kernel_regularizer=l2_regularizer)
    conv6 = cnn_1d(conv5, 5, params['channels'][0] * 32,
                   mode=mode,
                   activation=params['activation'], name="conv6",
                   kernel_regularizer=l2_regularizer)
    pool6 = max_pool_layer_1d(conv6, 2, name="pool6", stride=-1)
    pool6 = tf.layers.batch_normalization(pool6)

    conv7 = cnn_1d(pool6, 5, params['channels'][0] * 64,
                   mode=mode,
                   activation=params['activation'], name="conv7",
                   kernel_regularizer=l2_regularizer)
    conv8 = cnn_1d(conv7, 5, params['channels'][0] * 64,
                   mode=mode,
                   activation=params['activation'], name="conv8",
                   kernel_regularizer=l2_regularizer)
    conv9 = cnn_1d(conv8, 5, params['channels'][0] * 64,
                   mode=mode,
                   activation=params['activation'], name="conv9",
                   kernel_regularizer=l2_regularizer)
    pool9 = max_pool_layer_1d(conv9, 2, name="pool6", stride=-1)
    pool9 = tf.layers.batch_normalization(pool9)

    #LSTM network
    cell_1 = tf.keras.layers.LSTMCell(lstm_unit)
    cell_2 = tf.keras.layers.LSTMCell(lstm_unit)
    cell_3 = tf.keras.layers.LSTMCell(lstm_unit)
    multicell = tf.nn.rnn_cell.MultiRNNCell([cell_1,cell_2,cell_3])

    nn, state = tf.nn.dynamic_rnn(multicell, pool9, dtype=tf.float32)
    nn = tf.transpose(nn, [1,0,2])
    nn = tf.gather(nn,int(nn.get_shape()[0])-1)

    #Dense
    logits = fc_layer(nn, 3,
                      mode=mode,
                      activation=None, name='predict', kernel_regularizer=l2_regularizer)
    return logits


def softmax_focal_loss(labels_l, logits_l, gamma=2., alpha=4.):
    """Focal loss for multi-classification
    https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002

    Arguments:
        labels_l {tensor} -- ground truth labels_l, shape of [batch_size, num_class] <- Integer of class
        logits_l {tensor} -- model's output, shape of [batch_size, num_class] <- Before softmax

    Keyword Arguments:
        gamma {float} -- (default: {2.0})
        alpha {float} -- (default: {4.0})

    Returns:
        [tensor] -- loss.
    """

    gamma = float(gamma)

    epsilon = 1e-32
    labels_l = tf.one_hot(indices=tf.cast(labels_l, tf.int32), depth=3)
    logits_l = tf.cast(logits_l, tf.float32)

    logits_l = tf.nn.softmax(logits_l)
    logits_l = tf.add(logits_l, epsilon)  # Add epsilon so log is valid
    ce = tf.multiply(labels_l, -tf.log(logits_l))  # Cross entropy, shape of [batch_size, num_class]
    fl_weight = tf.multiply(labels_l, tf.pow(tf.subtract(1., logits_l), gamma))  # This is focal loss part
    fl = tf.multiply(alpha, tf.multiply(fl_weight, ce))  # Add alpha weight here
    reduced_fl = tf.reduce_max(fl, axis=1)
    return tf.reduce_mean(reduced_fl)


def get_loss_weight(labels):  # Calculate loss weight of a single batch
    score_one = tf.reduce_sum(tf.cast(tf.equal(labels, tf.constant(0, dtype=tf.int64)), dtype=tf.float32))
    score_three = tf.reduce_sum(tf.cast(tf.equal(labels, tf.constant(1, dtype=tf.int64)), dtype=tf.float32))
    score_five = tf.reduce_sum(tf.cast(tf.equal(labels, tf.constant(2, dtype=tf.int64)), dtype=tf.float32))
    sum_total = score_one + score_three + score_five
    # Add 1 to all denominator to prevent overflow
    weight = tf.stack(
        [tf.math.divide(sum_total, score_one + 1), tf.math.divide(sum_total, score_three + 1),
         tf.math.divide(sum_total, score_five + 1)],
        axis=0)
    return tf.expand_dims(weight, axis=0)


def custom_l2_reg(loss, lambda_=0.01):
    # Reference: https://stackoverflow.com/questions/55029716/how-to-regularize-loss-function
    ys = tf.reduce_mean(loss)
    l2_norms = [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
    l2_norm = tf.reduce_sum(l2_norms)
    loss = ys + lambda_ * l2_norm
    return loss, lambda_ * l2_norm


# Define Model
def my_model(features, labels, mode, params, config):
    # features['image'] = features['image']*100  # Since the difference is too small
    params['activation'] = tf.nn.leaky_relu
    # Input: (Batch_size,300,8)
    logits = model_cnn_1d(features, mode, params, config)
    # Predict Mode
    predicted_class = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'score': predicted_class[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    labels = (labels - 1) / 2  # Convert score from 1,3,5 to 0,1,2
    one_hot_label = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
    labels = tf.cast(labels, tf.int64)

    # Create loss weight to help imbalance dataset between each class
    # clamp_val = 5  # Max loss weight cannot be more than 5 times of min value
    # if max(params['loss_weight']) / min(params['loss_weight']) > clamp_val:
    #     params['loss_weight'][params['loss_weight'].index(max(params['loss_weight']))] = clamp_val * min(
    #         params['loss_weight'])
    # weight = tf.constant([[params['loss_weight'][0], params['loss_weight'][1], params['loss_weight'][2]]],
    #                      dtype=tf.float32)

    # Use loss weight based on each batch
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss_weight_raw = get_loss_weight(labels)
        loss_weight = tf.matmul(one_hot_label, loss_weight_raw, transpose_b=True, a_is_sparse=True)
    else:
        loss_weight = 1.0
    # Cross-entropy loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits,
                                                  weights=loss_weight)  # labels is int of class, logits is vector
    loss, reg_loss = custom_l2_reg(loss, lambda_=0.01)

    # Focal loss
    # loss = softmax_focal_loss(labels, logits, gamma=0., alpha=loss_weight)

    accuracy = tf.metrics.accuracy(labels, predicted_class)

    my_accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predicted_class), dtype=tf.float32))
    acc = tf.summary.scalar("accuracy_manual", my_accuracy)  # Number of correct answer

    # Create parameters to show in Tensorboard
    ex_prediction = tf.summary.scalar("Prediction Output", predicted_class[0])
    ex_ground_truth = tf.summary.scalar("Mean Ground Truth", tf.reduce_mean(tf.cast(labels,tf.float32)))
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # print("d_vars", d_vars)
    # global_step = tf.summary.scalar("Global steps",tf.train.get_global_step())

    trainable_variable_name = [v.name for v in tf.trainable_variables()]

    # tf.summary for all weight and bias
    summary_weight = []
    for i, t in enumerate(trainable_variable_name):
        summary_weight.append(tf.summary.histogram(t, tf.trainable_variables()[i]))
    steps = tf.train.get_global_step()

    # Train Mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(params['learning_rate'], steps,
                                                   20000, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        loss_gradient = [optimizer.compute_gradients(loss, tf.trainable_variables()[
            trainable_variable_name.index('dense_1/kernel:0')]),
                         optimizer.compute_gradients(loss, tf.trainable_variables()[
                             trainable_variable_name.index('conv1/kernel:0')])]
        train_op = optimizer.minimize(loss, global_step=steps)

        save_steps = 1000
        saver_hook = tf.train.SummarySaverHook(save_steps=save_steps, summary_op=tf.summary.merge_all(),
                                               output_dir=config.model_dir)
        print_input_hook = PrintValueHook(features['image'], "Input value", tf.train.get_global_step(), save_steps)
        print_input_name_hook = PrintValueHook(features['name'], "Input name", tf.train.get_global_step(), save_steps)
        print_logits_hook = PrintValueHook(tf.nn.softmax(logits), "Training logits", tf.train.get_global_step(),
                                           save_steps)
        print_label_hook = PrintValueHook(labels, "Labels", tf.train.get_global_step(), save_steps)
        print_lr_hook = PrintValueHook(learning_rate, "Learning rate", tf.train.get_global_step(), save_steps)
        print_loss_hook = PrintValueHook(loss, "Total Loss", tf.train.get_global_step(), save_steps)
        print_reg_loss_hook = PrintValueHook(reg_loss, "Regularization Loss", tf.train.get_global_step(), save_steps)

        print_weight_balance_hook = PrintValueHook(loss_weight_raw, "Loss weight", tf.train.get_global_step(),
                                                   save_steps)
        print_lg_hook = PrintValueHook(loss_gradient[0][0][0][0, 0:16], "FC6 Loss gradient", tf.train.get_global_step(),
                                       save_steps)
        print_lg2_hook = PrintValueHook(loss_gradient[0][0][1][0, 0:16], "FC6 Variable", tf.train.get_global_step(),
                                        save_steps)
        print_lg3_hook = PrintValueHook(loss_gradient[1][0][0][0, 0, :], "Conv1 Loss gradient",
                                        tf.train.get_global_step(),
                                        save_steps)
        print_lg4_hook = PrintValueHook(loss_gradient[1][0][1][0, 0, :], "Conv1 Variable", tf.train.get_global_step(),
                                        save_steps)
        # Setting logging parameters
        train_hooks = [print_input_hook, print_input_name_hook,
                       saver_hook, print_logits_hook, print_label_hook,
                       print_lr_hook,
                       print_loss_hook,  print_reg_loss_hook,
                       print_weight_balance_hook,
                       # print_lg_hook, print_lg2_hook,
                       # print_lg3_hook, print_lg4_hook,
                       ]

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                          training_hooks=train_hooks)

    # Evaluate Mode
    print("Evaluation Mode")
    eval_save_steps = 10
    # Create result(.csv) file, if not exist
    # If change any header here, don't forget to change data in EvalResultHook (custom_hook.py)
    if not os.path.isfile(params['result_path']):
        with open(os.path.join(params['result_path'], params['result_file_name']), "w") as csvfile:
            fieldnames = ['Name', 'Label', 'Predicted Class', 'Confident level', 'All confident level']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Create hooks
    if params['result_file_name'] == 'train_result.csv':
        saver_hook = tf.train.SummarySaverHook(save_steps=eval_save_steps, summary_op=tf.summary.merge_all(),
                                               output_dir=os.path.join(config.model_dir, 'train_final'))
    else:
        saver_hook = tf.train.SummarySaverHook(save_steps=eval_save_steps, summary_op=tf.summary.merge_all(),
                                               output_dir=os.path.join(config.model_dir, 'eval'))
    tensorboard_hook = tf.train.SummarySaverHook(save_steps=eval_save_steps, summary_op=tf.summary.merge_all(),
                                           output_dir=config.model_dir)
    csv_name = tf.convert_to_tensor(os.path.join(params['result_path'], params['result_file_name']), dtype=tf.string)
    print_result_hook = EvalResultHook(features['name'], labels, predicted_class, tf.nn.softmax(logits), csv_name)
    print_logits_hook = PrintValueHook(tf.nn.softmax(logits), "Validation Training logits", tf.train.get_global_step(),
                                       0)
    print_label_hook = PrintValueHook(labels, "Validation Labels", tf.train.get_global_step(), 0)

    eval_hooks = [saver_hook, tensorboard_hook, print_result_hook,
                  # print_logits_hook, print_label_hook,
                  ]
    return tf.estimator.EstimatorSpec(mode=mode, eval_metric_ops={'accuracy': accuracy}, loss=loss,
                                      evaluation_hooks=eval_hooks)
