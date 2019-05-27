import tensorflow as tf
import csv


# Save result from evaluation into csv file
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

        with open(result_path, "a") as csvFile:
            writer = csv.writer(csvFile)
            for label, pred, prob in zip(labels, predicted_classes, probabilities):
                prob = prob[pred]  # Show probability of the predicted class
                label = (label * 2) + 1
                pred = (pred * 2) + 1
                writer.writerow([label, pred, prob])
                # print("Label: %s, Prediction: %s" % (label, pred))


# Print any variable inside for debugging
class PrintValueHook(tf.train.SessionRunHook):
    def __init__(self, value, variable_name):
        self.value = value
        self.variable_name = tf.convert_to_tensor(variable_name, dtype=tf.string)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs([self.value, self.variable_name])

    def after_run(self, run_context, run_values):
        print("Variable %s: %s" % (run_values[0], run_values[1]))