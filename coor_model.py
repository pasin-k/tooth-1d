import tensorflow as tf
import csv
import os

# tf.enable_eager_execution()
from custom_hook import EvalResultHook, PrintValueHook


# In case of needing l2-regularization: https://stackoverflow.com/questions/44232566/add-l2-regularization-when-using-high-level-tf-layers/44238354#44238354


# Default stride of 1, padding:same
def cnn_1d(layer,
           conv_filter_size,  # [Scalar]
           num_filters,  # [Scalar]
           activation=tf.nn.relu,
           stride=1,
           padding='same',
           name='', kernel_regularizer=0.0):  # Stride of CNN
    # We shall define the weights that will be trained using create_weights function.
    layer = tf.keras.layers.Conv1D(num_filters, conv_filter_size, strides=stride, padding=padding,
                                   activation=activation,
                                   kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer))(layer)

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


def fc_layer(layer,  #
             num_outputs,
             activation=tf.nn.relu,
             name='',
             kernel_regularizer=0.0):
    # Let's define trainable weights and biases.
    layer = tf.keras.layers.Dense(num_outputs, activation=activation,
                                  kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer))(layer)
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
    require_channel = 1
    if len(params['channels']) != require_channel:
        raise ValueError("This model need %s channels input, current input: %s" % (require_channel, params['channels']))

    layer1 = fc_layer(features, num_outputs=params['channels'][0] * 64, activation=params['activation'])
    layer2 = fc_layer(layer1, num_outputs=params['channels'][0] * 128, activation=params['activation'])
    dropout2 = tf.keras.layers.Dropout(rate=params['dropout_rate'])(layer2)
    layer3 = fc_layer(dropout2, num_outputs=params['channels'][0] * 64, activation=params['activation'])
    dropout3 = tf.keras.layers.Dropout(rate=params['dropout_rate'])(layer3)
    logits = fc_layer(dropout3, 3, activation=tf.sigmoid, name='predict')

    print("Fully conneted layer")
    return logits


# Using max pooling
def model_cnn_1d(features, mode, params):
    # print(features)
    require_channel = 2
    if len(params['channels']) != require_channel:
        raise ValueError("This model need %s channels input, current input: %s" % (require_channel, params['channels']))

    # Input size:300x8
    '''
    This model is based on "A Comparison of 1-D and 2-D Deep Convolutional Neural Networks in ECG Classification"
    '''
    # (1) Filter size: 7x32, max pooling of k3 s2
    # print(params)
    conv1 = cnn_1d(features, 7, params['channels'][0] * 16, activation=params['activation'], name="conv1",
                   kernel_regularizer=0.01)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    pool1 = max_pool_layer_1d(conv1, 3, name="pool1", stride=2)
    # Output: 294x32 -> 147x32

    # (2) Filter size: 5x64, max pooling of k3 s2
    conv2 = cnn_1d(pool1, 5, params['channels'][0] * 32, activation=params['activation'], name="conv2",
                   kernel_regularizer=0.01)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    pool2 = max_pool_layer_1d(conv2, 3, "pool2", stride=2)
    # Output: 143x64 -> 71x64

    # (3) Filter size: 3x128 (3 times), max pooling of k3 s2
    conv3 = cnn_1d(pool2, 3, params['channels'][0] * 64, activation=params['activation'], name="conv3_1",
                   kernel_regularizer=0.01)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = cnn_1d(conv3, 3, params['channels'][0] * 64, activation=params['activation'], name="conv3_2",
                   kernel_regularizer=0.01)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = cnn_1d(conv3, 3, params['channels'][0] * 64, activation=params['activation'], name="conv3_3",
                   kernel_regularizer=0.01)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    pool3 = max_pool_layer_1d(conv3, 3, "pool2", stride=2)
    # print("Pool: %s"% pool3)
    # Output: 65x128 -> 32x128 = 4096

    fc4 = flatten_layer(pool3)
    print("debug")
    print(fc4)
    fc4 = fc_layer(fc4, params['channels'][1] * 1024, activation=params['activation'], name='fc5',
                   kernel_regularizer=0.01)
    dropout4 = tf.keras.layers.Dropout(rate=params['dropout_rate'])(fc4)
    # Output: 4096 -> 4096 -> 3

    fc5 = fc_layer(dropout4, params['channels'][1] * 1024, activation=params['activation'], name='fc6',
                   kernel_regularizer=0.01)
    dropout5 = tf.keras.layers.Dropout(rate=params['dropout_rate'])(fc5)

    logits = fc_layer(dropout5, 3, activation=tf.nn.tanh, name='predict', kernel_regularizer=0.01)
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
    labels = (labels - 1) / 2
    one_hot_label = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
    labels = tf.cast(labels, tf.int64)

    # for i in range(len(params['loss_weight'])):
    #     if params['loss_weight'][i]:
    #         # TODO: Clamp max weight
    weight = tf.constant([[params['loss_weight'][0], params['loss_weight'][1], params['loss_weight'][2]]],
                         dtype=tf.float32)
    loss_weight = tf.matmul(one_hot_label, weight, transpose_b=True, a_is_sparse=True)

    loss = softmax_focal_loss(labels, logits, gamma=0., alpha=loss_weight)  # labels is int of class, logits is vector

    # loss = tf.losses.sparse_softmax_cross_entropy(labels, logits,
    #                                               weights=loss_weight)  # labels is int of class, logits is vector

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
        train_hooks = []
        print_variable_hook = PrintValueHook(tf.nn.softmax(logits), "Training logits", tf.train.get_global_step(), 5000)
        train_hooks.append(saver_hook)
        train_hooks.append(print_variable_hook)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                          training_hooks=train_hooks)

    # Evaluate Mode
    print("Evaluation Mode")
    # Create result(.csv) file, if not exist
    if not os.path.isfile(params['result_path']):
        with open(params['result_path'] + params['result_file_name'], "w") as csvfile:
            fieldnames = ['Label', 'Predicted Class', 'Confident level']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Create hooks
    eval_hooks = []
    if params['result_file_name'] == 'train_result.csv':
        saver_hook = tf.train.SummarySaverHook(save_steps=10, summary_op=tf.summary.merge_all(),
                                               output_dir=config.model_dir + 'train_final')
    else:
        saver_hook = tf.train.SummarySaverHook(save_steps=10, summary_op=tf.summary.merge_all(),
                                               output_dir=config.model_dir + 'eval')
    csv_name = tf.convert_to_tensor(params['result_path'] + params['result_file_name'], dtype=tf.string)
    print_result_hook = EvalResultHook(labels, predicted_class, tf.nn.softmax(logits), csv_name)
    eval_hooks.append(saver_hook)
    eval_hooks.append(print_result_hook)
    return tf.estimator.EstimatorSpec(mode=mode, eval_metric_ops={'accuracy': accuracy}, loss=loss,
                                      evaluation_hooks=eval_hooks)
