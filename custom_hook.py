import tensorflow as tf


class EvalResultHook(tf.train.SessionRunHook):
    def __init__(self, labels, predicted_class):
        self.labels = labels
        self.predicted_class = predicted_class

    def before_run(self, run_context):
        return tf.train.SessionRunArgs([self.labels, self.predicted_class])

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        labels = run_values.results[0]
        predicted_classes = run_values.results[1]
        for label, classes in zip(labels, predicted_classes):
            label = (label * 2) + 1
            classes = (classes * 2) + 1
            print("Prediction: %s" % classes)
            print("Labels: %s" % label)
