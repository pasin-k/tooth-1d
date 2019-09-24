import tensorflow as tf
import numpy as np
import os


def conv_encoder(inputs, num_filters, scope=None):
    net = inputs
    with tf.variable_scope(scope, 'encoder', [inputs]):
        for layer_id, num_outputs in enumerate(num_filters):
            with tf.variable_scope('block{}'.format(layer_id)):
                net = repeat(net, 2, tf.layers.conv2d, num_outputs=num_outputs, kernel_size=3, stride=1, padding="SAME")
                net = tf.layers.max_pool2d(net, kernel_size=2)
                # Alternatives instead of pooling
                # net = tf.layers.conv2d(net, num_outputs=num_outputs, kernel_size=3, stride=2, padding="SAME")

        net = tf.identity(net, name='output')
    return net


def conv_decoder(inputs, num_filters, output_shape, scope=None):
    net = inputs
    with tf.variable_scope(scope, 'decoder', [inputs]):
        for layer_id, num_outputs in enumerate(num_filters):
            with tf.variable_scope('block_{}'.format(layer_id),
                                   values=(net,)):
                net = tf.layers.conv2d_transpose(
                    net, num_outputs,
                    kernel_size=3, stride=2, padding='SAME')

        with tf.variable_scope('linear', values=(net,)):
            net = tf.layers.conv2d_transpose(
                net, 1, activation_fn=None)

    return net


def conv_autoencoder(inputs, num_filters, activation_fn, weight_decay, mode):
    weights_init = slim.initializers.variance_scaling_initializer()
    if weight_decay is None:
        weights_reg = None
    else:
        weights_reg = tf.contrib.layers.l2_regularizer(weight_decay)

    with slim.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.conv2d_transpose],
                        weights_initializer=weights_init,
                        weights_regularizer=weights_reg,
                        activation_fn=activation_fn):
        net = tf.reshape(inputs, [-1, 28, 28, 1])
        net = conv_encoder(net, num_filters)
        net = conv_decoder(net, num_filters[::-1], [-1, 28, 28, 1])

        net = tf.reshape(net, [-1, 28 * 28])
    return net


if __name__ == '__main__':
    autoencoder()
    print("Autoencoder.py completed")
