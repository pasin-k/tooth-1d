import numpy as np
import os
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from preprocess.stl_to_image import point_sampling

def plot_graph(folder, filename):
    if type(filename) is not list:
        filename = [filename]
    plt.rcParams.update({'font.size': 35})
    fig, ax = plt.subplots(2, len(filename))
    for i, f in enumerate(filename):
        data = np.load(os.path.join(folder,f))
        print(np.shape(data))
        data= point_sampling(data, 300)

        data_x = data[:, 0]
        data_y = data[:, 1]


        # fig.suptitle(''.join(f.split('/')[-1].split('_')[0:-1]))
        if len(filename) > 1:
            ax[0,i].plot(data_x)
            ax[1,i].plot(data_y)
        else:
            ax[0].plot(data_x)
            ax[1].plot(data_y)

    # print(np.shape(ax.flat))
    for i, a in enumerate(ax.flat):
        if i == 0:
            ylabel = 'x-axis'
        elif i == len(filename):
            ylabel = 'y-axis'
        a.set(xlabel="# of point".format(degree[i % len(filename)]), ylabel=ylabel)

    for a in ax.flat:
        a.label_outer()

    plt.show()


degree = [0,45,90,135]
folder_name = "./data/coor_debug_real_nofix"
file_name = ["PreparationScan_304101_0-0_0.npy"]

plot_graph(folder_name, file_name)
