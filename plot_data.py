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


        fig.suptitle(''.join(f.split('/')[-1].split('_')[0:-1]))
        ax[0,i].plot(data_x)
        ax[1,i].plot(data_y)

    # print(np.shape(ax.flat))
    for i, a in enumerate(ax.flat):
        if i == 0:
            ylabel = 'x-axis'
        elif i == 4:
            ylabel = 'y-axis'
        a.set(xlabel="{} degree".format(degree[i % 4]), ylabel=ylabel)

    for a in ax.flat:
        a.label_outer()

    plt.show()


degree = [0,45,90,135]
# folder_name = "/home/pasin/Documents/Link_to_my2DCNN/data/coor_14aug"
# folder_name = "/home/pasin/Documents/Link_to_my2DCNN/data/coor_14aug_real_point"
# folder_name = "/home/pasin/Documents/Link_to_my2DCNN/data/coor_debug_nofix"
folder_name = "/home/pasin/Documents/Link_to_my2DCNN/data/coor_debug_real_nofix"
file_name = ["PreparationScan_304101_0-0_0.npy","PreparationScan_304101_0-0_45.npy",
             "PreparationScan_304101_0-0_90.npy","PreparationScan_304101_0-0_135.npy"]

plot_graph(folder_name,file_name)
