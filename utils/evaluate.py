import csv
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


def get_metric_score(file_path, metric_type, average=None):
    """
    Return precision score of result.csv file from our work
    :param file_path:
    :param average:
    :return:
    """
    metric_option = ['precision', 'recall', 'f1']
    assert metric_type in metric_option, "metric_type need to be one of the available option"
    data = pd.read_csv(file_path, usecols=['Label', 'Predicted Class'])
    data.columns = ['label', 'predict']
    if metric_type == 'precision':
        score = precision_score(data[['label']].to_numpy(), data[['predict']].to_numpy(), labels=[1, 3, 5],
                                average=average)
    elif metric_type == 'recall':
        score = recall_score(data[['label']].to_numpy(), data[['predict']].to_numpy(), labels=[1, 3, 5],
                             average=average)
    else:
        score = f1_score(data[['label']].to_numpy(), data[['predict']].to_numpy(), labels=[1, 3, 5], average=average)
    return score


# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     # print(cm)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=90)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')


def get_multi_result():
    base_path = "/home/pasin/Documents/Pasin/model/Workstation_Result"
    mid_path = [["coor_0aug_bl",
                 "/home/pasin/Documents/Pasin/model/Workstation_Result/coor_0aug_md/file/20200128_18_44_46_usethisone",
                 "/home/pasin/Documents/Pasin/model/Workstation_Result/coor_0aug_width/file/20200128_18_40_32",
                 "/home/pasin/Documents/Pasin/model/Workstation_Result/coor_0aug_sharpness/file/20200128_16_45_32_usethisone"],
                [
                    "/home/pasin/Documents/Pasin/model/Workstation_Result/coor_42aug_real_point_bl_65/file/20200127_22_55_33_best_65",
                    "/home/pasin/Documents/Pasin/model/Workstation_Result/coor_42aug_real_point_md_66/file/20200128_07_11_36_Best",
                    "/home/pasin/Documents/Pasin/model/Workstation_Result/coor_42aug_real_point_width_61/file/20200125_07_27_38_best",
                    "/home/pasin/Documents/Pasin/model/Workstation_Result/coor_42aug_real_point_sharpness_65/file/20200128_02_00_59_best"],
                [
                    "/home/pasin/Documents/Pasin/model/Workstation_Result/coor_42aug_diff_point_bl/file/20200129_23_08_26_best",
                    "/home/pasin/Documents/Pasin/model/Workstation_Result/coor_42aug_diff_point_md/file/20200130_04_59_14_best",
                    "/home/pasin/Documents/Pasin/model/Workstation_Result/coor_42aug_diff_point_width/file/20200128_03_45_06_best_77",
                    "/home/pasin/Documents/Pasin/model/Workstation_Result/coor_42aug_diff_point_sharpness/file/20200128_05_38_16_best_67"],
                [
                    "/home/pasin/Documents/Pasin/model/Workstation_Result/coor_42aug_real_point_new_data_taper/file/20200128_18_44_30_best",
                    None,
                    "/home/pasin/Documents/Pasin/model/Workstation_Result/coor_42aug_real_point_new_data_width/file/20200128_20_12_07_best",
                    "/home/pasin/Documents/Pasin/model/Workstation_Result/coor_42aug_real_point_new_data_sharpness/file/20200129_05_36_04_best"],
                ]
    file_name = "result.csv"
    metric_type = 'precision'
    average = 'micro'
    score_list = []
    for m_list in mid_path:
        score_temp = []
        for m in m_list:
            if m is None:
                score_temp.append(None)
            else:
                data_path = os.path.join(base_path, m, file_name)
                score_temp.append(get_metric_score(data_path, metric_type, average=average))
        score_list.append(score_temp)

    score_data = pd.DataFrame.from_records(score_list)
    score_data = score_data.rename(columns={0: 'Taper(BL)', 1: 'Taper(MD)', 2: 'Width', 3: 'Sharpness'})

    print(score_data)
    score_data.to_csv(
        "/home/pasin/Documents/Google_Drive/Aa_TIT_LAB_Comp/TIT_Files/Thesis Material/summary_{}_{}.csv".format(metric_type,average))


if __name__ == '__main__':
    # get_multi_result()

    metric_type = 'precision'
    average = 'macro'
    data_path = "/home/pasin/Documents/Pasin/model/Workstation_Result/coor_42aug_real_point_no_batchnorm/file/20191219_17_49_19_best/result.csv"

    print(get_metric_score(data_path, metric_type, average=average))