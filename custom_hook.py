import tensorflow as tf
import csv


class EvalResultHook(tf.train.SessionRunHook):
    def __init__(self, labels, predicted_class, probability, result_path):
        self.labels = labels
        self.predicted_class = predicted_class
        self.probability = probability
        self.result_path = result_path

    def before_run(self, run_context):
        return tf.train.SessionRunArgs([self.labels, self.predicted_class, self.probability, self.result_path])

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        labels = run_values.results[0]
        predicted_classes = run_values.results[1]
        probabilities = run_values.results[2]
        result_path = run_values.results[3]
        with open(result_path, "a") as csvfile:
            writer = csv.writer(csvfile)
            for label, pred, prob in zip(labels, predicted_classes, probabilities):
                prob = prob[pred]  # Show probability of the predicted class
                label = (label * 2) + 1
                pred = (pred * 2) + 1
                writer.writerow([label, pred, prob])

