import os
import numpy as np
# import matplotlib
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

v = '1.1.0'
# 1.0 start with savePyplot
# 1.1 Debugged
print("imgSave.py version: " + str(v))


# Plot the list of coordinates and save it as PNG image
# Input     CoorList        -> List of {List of numpy coordinates <- get from stlSlicer}
#           outDirectory    -> String, Directory to save output
#           fileName        -> String, name of file to save
#           Degree          -> List of angles used in
#           fileType        -> [Optional], such as png,jpeg,...
# Output    none            -> Only save as output outside
def saveplot(coorList, outDirectory, fileName, degree, fileType="png"):
    outDirectory = os.path.abspath(outDirectory)
    if len(degree) != len(coorList[0]):
        print("# of Degree expected: %d" % len(degree))
        print("# of Degree found: %d" % len(coorList[0]))
        raise Exception('Number of degree specified is not equals to coordinate ')
    # Start saving image
    print("Start saving images")
    for i in range(len(coorList)):
        for d in range(len(degree)):
            coor = coorList[i][d]
            plt.plot(coor[:, 0], coor[:, 1], color='black', linewidth=1)
            plt.axis('off')
            fullname = "%s_%s_%d.%s" % (
            fileName, str(i).zfill(3), degree[d], fileType)  # Name with some additional data
            outName = os.path.join(outDirectory, fullname)

            plt.savefig(outName, bbox_inches='tight')
            plt.clf()
    print("Finished plotting for %d images with %d rotations" % (len(coorList), len(degree)))
