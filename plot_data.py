import numpy as np
import os
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def plot_graph(folder, filename):
    if type(filename) is not list:
        filename = [filename]
    fig, ax = plt.subplots(2, len(filename))

    for i, f in enumerate(filename):
        data = np.load(os.path.join(folder,f))
        print(np.shape(data))
        data_x = data[:, 0]
        data_y = data[:, 1]


        fig.suptitle(f.split('/')[-1])
        ax[0,i].plot(data_x)
        ax[1,i].plot(data_y)

    plt.show()

folder_name  = "/home/pasin/Documents/Link_to_my2DCNN/data/coor_42aug"
file_name = ["PreparationScan_304101_0-0_0.npy","PreparationScan_304101_0-0_45.npy",
             "PreparationScan_304101_0-0_90.npy","PreparationScan_304101_0-0_135.npy"]

plot_graph(folder_name,file_name)
