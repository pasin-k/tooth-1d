import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.enable_eager_execution()


def softmax_focal_loss(labels_l, logits_l, gamma=2., alpha=4.):
    """Focal loss for multi-classification
    https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002

    Arguments:
        labels_l {tensor} -- ground truth labels_l, shape of [batch_size, num_class] <- One hot vector
        logits_l {tensor} -- model's output, shape of [batch_size, num_class] <- Before softmax

    Keyword Arguments:
        gamma {float} -- (default: {2.0})
        alpha {float} -- (default: {4.0})

    Returns:
        [tensor] -- loss.
    """

    gamma = float(gamma)

    epsilon = 1e-32
    # labels_l = tf.cast(labels_l, tf.float32)
    labels_l = tf.one_hot(indices=tf.cast(labels_l, tf.int32), depth=3)
    logits_l = tf.cast(logits_l, tf.float32)

    # logits_l = tf.nn.softmax(logits_l)
    # print("Softmax: %s" % logits_l)
    logits_l = tf.add(logits_l, epsilon)  # Add epsilon so log is valid
    ce = tf.multiply(labels_l, -tf.log(logits_l))  # Cross entropy, shape of [batch_size, num_class]
    fl_weight = tf.multiply(labels_l, tf.pow(tf.subtract(1., logits_l), gamma))  # This is focal loss part
    fl = tf.multiply(alpha, tf.multiply(fl_weight, ce))  # Add alpha weight here
    reduced_fl = tf.reduce_max(fl, axis=1)
    return tf.reduce_mean(reduced_fl)


# labels = tf.constant([1,5,3])
# logits = tf.constant([[50,0,0],[1,1,2],[90,0,90]],dtype=tf.float32)
x = []
y = []
y2 = []
y3 = []
y4 = []

for i in np.linspace(0, 1, 1001):
    x.append(i)
    labels = tf.constant([1])
    logits = tf.constant([i, 0, 1 - i])
    labels = (labels - 1) / 2
    one_hot_label = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
    labels = tf.cast(labels, tf.int64)

    weight = tf.constant([[1, 1, 1]],
                         dtype=tf.float32)
    loss_weight = tf.matmul(one_hot_label, weight, transpose_b=True, a_is_sparse=True)

    loss_fc = softmax_focal_loss(labels, logits, gamma=0, alpha=loss_weight)  # labels is int of class, logits is vector
    y.append(loss_fc)

    loss_fc = softmax_focal_loss(labels, logits, gamma=1, alpha=loss_weight)  # labels is int of class, logits is vector
    y2.append(loss_fc)

    loss_fc = softmax_focal_loss(labels, logits, gamma=2, alpha=loss_weight)  # labels is int of class, logits is vector
    y3.append(loss_fc)

    loss_fc = softmax_focal_loss(labels, logits, gamma=5, alpha=loss_weight)  # labels is int of class, logits is vector
    y4.append(loss_fc)
    # loss_ce = tf.losses.sparse_softmax_cross_entropy(labels, logits,
    #                                               weights=loss_weight)  # labels is int of class, logits is vector
    # print("Focal loss:")
    # print(loss_fc)
    # print("Cross entropy")
    # print(loss_ce)
    # print("" )

# with tf.Session() as sess:
#     print(sess.run(loss_fc))
#     print(sess.run(loss_ce))
fig = plt.figure(figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')
plt.plot(x, y, color='skyblue', label='Cross Entropy')
plt.plot(x, y2, color='olive', label='gamma = 1')
plt.plot(x, y3, color='green', label='gamma = 2')
plt.plot(x, y4, color='red', label='gamma = 5')
plt.legend(loc=2, prop={'size': 6})
# naming the x axis
plt.xlabel('Probability', fontsize=40)
plt.xlim(0, 1)
# naming the y axis
plt.ylabel('Loss', fontsize=40)
plt.ylim(0, 5)
# giving a title to my graph
plt.title('Focal Loss', fontsize=40)
plt.legend()
# function to show the plot
# plt.show()

plt.savefig('focal_loss.png')
