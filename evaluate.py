import csv
import numpy as np
from sklearn.metrics import f1_score


def get_f1_score(result_path, average=None):
    """
    Return f1_score
    :param result_path:
    :param average: None or 'micro' or 'macro'
    :return:
    """
    with open(result_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        y_true = []
        y_pred = []
        for row in reader:
            y_true.append(row[1])
            y_pred.append(row[2])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    score = f1_score(y_true, y_pred, labels=['1', '3', '5'], average=average)
    return score


if __name__ == '__main__':
    final_score = get_f1_score(
        "/home/pasin/Documents/Pasin/model/augment14_width_lr/20190927_12_25_0920190927_15_44_43_best/result.csv",
        average=None)
    print(final_score)
