import numpy as np
from stl import mesh
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def getSlicer(stl_file_name, Zplane, degree, augment=None, is_rearrange=True, axis=1):
    """
    Get list of np.array(X,2) coordinates from the slice  from a .stl file; X is varies
    :param stl_file_name:   String, Folder name
    :param Zplane:          Scalar value, selected plane
    :param degree:          List of degree of cross-section you want
    :param augment:         List of rotation degree to increase datasize
    :param is_rearrange:       Boolean, Rearrange coordinate from bottom left to right
    :param axis:            Axis of rotation (0 = X,1 = Y,2 = Z) (Default at 1, based from preliminary result)
    :return: reP_all:       List of list of numpy array with size of [len(augment),len(degree),[N,2])
    """
    # Import data and change to np array of N * 9 dimension
    # print(stl_file_name)
    prep_mesh = mesh.Mesh.from_file(stl_file_name)

    # Tmain is list of all triangles
    # Tmains = np.concatenate((prep_mesh.v0, prep_mesh.v1, prep_mesh.v2), axis=1).tolist()  # list of N * 9 dimension
    Tmain_temp = np.concatenate((prep_mesh.v0, prep_mesh.v1, prep_mesh.v2), axis=1)  # numpy array of N * 9 dimension
    if augment is not None:
        Tmain_all = []
        for a in augment:
            Tmain_all.append(rotatestl(Tmain_temp, axis, a))
    else:
        Tmain_all = [Tmain_temp]

    reP_all = []
    for Tmain in Tmain_all:
        reP = list()  # Output
        # vtOne = np.zeros([3])  # vertex #1
        # vtTwo1 = np.zeros([3])  # vertex #2
        # vtTwo2 = np.zeros([3])  # vertex #3
        for d in degree:
            P = np.empty([0, 3])  # Unarranged coordinates
            T = rotatestl(Tmain, axis, d).tolist()  # Default as Z-axis
            i = 1  # Special index added as a third column which will be used in 'rearrange' function
            while len(T) != 0:
                t = np.array(T.pop(0))  # Select some element from t
                Zcoor = np.array((t[2], t[5], t[8]))
                if (not ((Zcoor[0] < Zplane and Zcoor[1] < Zplane and Zcoor[
                    2] < Zplane) or  # Check if triangle is in the selected plane
                         (Zcoor[0] > Zplane and Zcoor[1] > Zplane and Zcoor[2] > Zplane))):
                    idxUp = np.argwhere(Zcoor > Zplane)  # Index of vertex ABOVE plane
                    idxDown = np.argwhere(Zcoor < Zplane)  # Index of vertex BELOW plane
                    if np.size(idxUp) == 1:  # If this true, one point is above plane
                        vtOne = t[idxUp[0][0] * 3:idxUp[0][0] * 3 + 3]
                        vtTwo1 = t[idxDown[0][0] * 3:idxDown[0][0] * 3 + 3]
                        vtTwo2 = t[idxDown[1][0] * 3:idxDown[1][0] * 3 + 3]
                    else:  # If this true, one point is below plane
                        vtOne = t[idxDown[0][0] * 3:idxDown[0][0] * 3 + 3]
                        vtTwo1 = t[idxUp[0][0] * 3:idxUp[0][0] * 3 + 3]
                        vtTwo2 = t[idxUp[1][0] * 3:idxUp[1][0] * 3 + 3]

                    # vtOne is vertex coordinate (3D) that is alone separate by plane (above or below alone)
                    # vtTwo1/vtTwo2 is vertex coordinate (3D) that is together after separate by plane
                    # l1,l2 is cross-section coordinate (2D) combined as 2x2 matrix
                    # The one with lower x coordinate with be the first row
                    l1 = slicecoor(Zplane, vtOne, vtTwo1, i)
                    l2 = slicecoor(Zplane, vtOne, vtTwo2, i)
                    l = np.array([l1, l2])  # This is [2,3] size numpy array of two coordinates that intersect Z plane
                    P = np.concatenate((P, l), axis=0)  # Accumulate all intersections
                    i = i + 1
            if is_rearrange:
                # print("%s_%s_%s" % (stl_file_name, d, augment))
                newP = rearrange(P)
                if newP is None:
                    print(
                        "getSlicer: %s has problem getting cross-section. Possible hole appeared in model" % stl_file_name)
                    reP = None
                    break
                else:
                    reP.append(newP)
            else:
                reP.append(P)
        reP_all.append(reP)
    return reP_all


# Rotate the stl data (expect np.array(N*9)), and return multiple rotated data on 'axis' (0,1,2) with 'degree' rotated
# Input:    T       -> Numpy array of triangles coordinate (N*9 Matrix)
#           axis    -> Axis of rotation (0 = X,1 = Y,2 = Z)
#           deg     -> rotation degree (scalar value)
# Output:   out     -> Numpy of Rotated numpy array size N*9
def rotatestl(T, axis, deg):
    # We default on rotate over Z-axis, thus rotate x-y-z first then applied Z-axis rotation
    V = np.zeros([np.shape(T)[0], 9])
    out = np.zeros([np.shape(T)[0], 9])
    if axis == 0:
        for i in range(0, 3):
            V[:, 3 * i + 0] = T[:, 3 * i + 1]  # New X axis is the data Y axis
            V[:, 3 * i + 1] = T[:, 3 * i + 2]  # New Y axis is the data Z axis
            V[:, 3 * i + 2] = T[:, 3 * i + 0]  # New Z axis is the data X axis *We rotate this axis now*
    elif axis == 1:
        for i in range(0, 3):
            V[:, 3 * i + 0] = T[:, 3 * i + 2]  # New X axis is the data Z axis
            V[:, 3 * i + 1] = T[:, 3 * i + 0]  # New Y axis is the data X axis
            V[:, 3 * i + 2] = T[:, 3 * i + 1]  # New Z axis is the data Y axis *We rotate this axis now*
    else:
        V = T
    # From now we assume that we want to rotate around Z-axis only
    rotMat = np.array([[np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg)), 0],
                       [-np.sin(np.deg2rad(deg)), np.cos(np.deg2rad(deg)), 0],
                       [0, 0, 1]])  # Rotation matrix, look up on wiki for more info
    V[:, 0:3] = np.matmul(V[:, 0:3], rotMat)
    V[:, 3:6] = np.matmul(V[:, 3:6], rotMat)
    V[:, 6:9] = np.matmul(V[:, 6:9], rotMat)
    # Revert axis back to normal
    if axis == 0:
        for i in range(0, 3):
            out[:, 3 * i + 0] = V[:, 3 * i + 2]  # Transform Z axis back to X
            out[:, 3 * i + 1] = V[:, 3 * i + 0]  # Transform X axis back to Y
            out[:, 3 * i + 2] = V[:, 3 * i + 1]  # Transform Y axis back to Z
    elif axis == 1:
        for i in range(0, 3):
            out[:, 3 * i + 0] = V[:, 3 * i + 1]  # Transform Y axis back to X
            out[:, 3 * i + 1] = V[:, 3 * i + 2]  # Transform Z axis back to Y
            out[:, 3 * i + 2] = V[:, 3 * i + 0]  # Transform X axis back to Z
    else:
        out = V
    out = np.round(out, 8)
    return out


# Rearrange order of tooth shape so it start from bottom left to right
# Input     P       -> Numpy size N*2, should be called from getSlicer function
# Output    rePoint -> Numpy size N*2, rearranged
def rearrange(P):
    P = np.round(P, 8)
    np.set_printoptions(threshold=np.inf)
    ind = np.lexsort((P[:, 1], P[:, 0]))
    P = P[ind]  # Sort coordinate from left to right
    # P = P[P[:, 0].argsort()]

    # Finding the starting and ending rows
    start_row = np.array([-1, -1])
    start_coor = np.zeros([2, 3])
    for i in range(np.int32(np.round(np.size(P, 0) / 2))):
        if start_row[0] == -1:
            if not (np.array_equal(P[2 * i, 0:2], P[2 * i + 1, 0:2])):
                start_row[0] = 2 * i
                start_coor[0, :] = P[2 * i, :]
        else:
            if not (np.array_equal(P[2 * i - 1, 0:2], P[2 * i, 0:2])):
                start_row[1] = 2 * i - 1
                start_coor[1, :] = P[2 * i - 1, :]
                break
    if start_row[1] == -1:  # This case occur when final point is last row
        start_row[1] = np.round(np.size(P, 0)) - 1
        start_coor[1, :] = P[np.round(np.size(P, 0)) - 1, :]
    # Set the left most one as startRow[0]
    if P[start_row[0], 0] > P[start_row[1], 0]:
        row_temp = start_row[0]
        start_row[0] = start_row[1]
        start_row[1] = row_temp

    # Rearranging coordinate starting from startRow[0]
    re_point = np.zeros([np.int32(np.round(np.size(P, 0)) / 2) + 1, 2])  # Rearranged points (Output)
    curr_line = P[start_row[0], 2]  # Current line that we are interested in, starting from first point
    re_point[0, :] = P[start_row[0], 0:2]
    i = 1  # Start at 1 because index 0 is already initialized
    while True:
        # Find the pair coordinate of the same line
        same_line = P[np.where(P[:, 2] == curr_line)]
        if np.array_equal(re_point[i - 1, :], same_line[0, 0:2]):
            re_point[i, :] = same_line[1, 0:2]
        else:
            re_point[i, :] = same_line[0, 0:2]
        # Find the line that connect to the current line
        same_point = P[np.where((P[:, 0] == re_point[i, 0]) & (P[:, 1] == re_point[i, 1]))]
        if np.array_equal(same_point[0], P[start_row[1], :]) and (same_point.shape[0] == 1):
            if np.array_equal(re_point[-1, :], [0, 0]):
                print("WARNING: Not all expected coordinate are discovered. Some (0,0) rows may occur")
            break
        if curr_line == same_point[0, 2]:
            if np.shape(same_point)[0] == 1:  # This happens if there's a hole in the cross-section
                # print("Starting Coordinate found: %s" % start_coor)
                # print("Problem Coordinate: %s" % same_point)
                return None
                # raise Exception("Bug detected, possibly found multiple starting points")
            curr_line = same_point[1, 2]
        else:
            curr_line = same_point[0, 2]
        i = i + 1
        if i == np.int32(np.round(np.size(P, 0)) / 2) + 1:
            print("WARNING: Too many coordinate than expected. Possible bug loop happened")
            break
    return re_point


# To get coordinate of a line start at pointA and pointB (3d coordinate) which intersect at Zplane
# (i is additional variable for rearrange)
# Input     Zplane          -> Scalar, Value of Z plane to intersect
#           pointA/pointB   -> numpy array [1,3] coordinate on both ends of the line
#           i               -> Scalar, additional value (Represent which # of line this is)
# Output    coor            -> numpy array [1,3] of 2D-coordinate plus i
def slicecoor(Zplane, pointA, pointB, i):
    coor = np.array([pointB[0] + (Zplane - pointB[2]) * (pointA[0] - pointB[0]) / (pointA[2] - pointB[2]), pointB[1] +
                     (Zplane - pointB[2]) * (pointA[1] - pointB[1]) / (pointA[2] - pointB[2]), i])
    return coor


"""
    Get list of np.array(X,2) coordinates from the slice  from a .stl file; X is varies
    :param stl_file_name:   String, Folder name
    :param Zplane:          Scalar value, selected plane
    :param degree:          List of degree of cross-section you want
    :param augment:         List of rotation degree to increase datasize
    :param is_rearrange:       Boolean, Rearrange coordinate from bottom left to right
    :param axis:            Axis of rotation (0 = X,1 = Y,2 = Z) (Default at 1, based from preliminary result)
    :return: reP_all:       List of list of numpy array with size of [len(augment),len(degree),[N,2])
    """
save_path = 'sliced_img_4PerTooth/combinedstlsegment/'


def slopefunc(x1, y1, x2, y2):  # function to find top line
    if x1 == x2 or y1 == y2:
        return 0
    else:
        a = (y2 - y1) / (x2 - x1)
    return abs(a)


import os
import matplotlib.pyplot as plt
import numpy as np

cross_sections = [0, 45, 90, 135]  # degrees to cut from stl
for subdir, dirs, files in os.walk('stl_data'):
    for file in files:
        # print os.path.join(subdir, file)
        filepath = subdir + os.sep + file
        if filepath.endswith("PreparationScan.stl"):
            for ind in range(0, 4):  # cycle thru cross sections
                directory = subdir.split('\\')  # for filenames to save

                a = getSlicer(filepath, 1, [cross_sections[ind]], [0], True, 1)  # use slicer
                A = np.asarray(a[0][0])  # turn into numpy

                x_max_index = np.where(A == np.amax(A, axis=0)[0])  # find max and min x to find sharp point of tooth
                x_min_index = np.where(A == np.amin(A, axis=0)[0])
                Xmin1 = (x_min_index[0])  # index of min x
                Xmin = (Xmin1[0])
                Yleft = A[Xmin][1]  # find y at min x location
                Xmax1 = (x_max_index[0])  # index max x
                Xmax = (Xmax1[0])
                Yright = A[Xmax][1]  # find y at max x location

                y_left_index = np.where(A == Yleft)  # to find the (max and min of Y) index needed to search thru
                y_right_index = np.where(A == Yright)  # maybe can use above function, but it works.
                y_left1 = (y_left_index[0])
                y_left = (Xmin1[0])
                y_right1 = (y_right_index[0])
                y_right = (Xmax1[0])

                # initialise slope, (+20 points to avoid corner of the curve which could provide a unusually large slope)
                slope, slope1 = slopefunc(A[Xmin][0], A[Xmin][1], A[y_left + 20][0], A[y_left + 20][1]), slopefunc(
                    A[y_right - 20][0], A[y_right - 20][1], A[Xmax][0], A[Xmax][1])

                # func to find largest slope location
                for findslopecount in range(len(a[0][0]) - 1):
                    if findslopecount > y_left + 20 and findslopecount < y_right - 20:
                        if slope <= slopefunc(
                                A[Xmin][0], A[Xmin][1], A[findslopecount][0], A[findslopecount][1]):
                            slope = slopefunc(A[Xmin][0], A[Xmin][1], A[findslopecount][0],
                                              A[findslopecount][1])
                            Ytop_left = A[findslopecount][1]
                        if slope1 < slopefunc(
                                A[findslopecount][0], A[findslopecount][1], A[Xmax][0], A[Xmax][1]):
                            slope1 = slopefunc(A[findslopecount][0], A[findslopecount][1], A[Xmax][0], A[Xmax][1])
                            Ytop_right = A[findslopecount][1]
                # lower of the two top locations
                top = min(Ytop_left, Ytop_right)
                '''
                Saving func
                arr_pts1 = [Ytop_left,Ytop_right,top]
                # np.save(file=save_path + 'Y_topcut_coords/'+directory[1] +'_'+ str(cross_sections[ind]), arr=arr_pts1)
'''
                # Ycut is the lower point, Ycut1 is the higher point
                if A[Xmin][1] > A[Xmax][1]:  # takes the lower of both, could use min but idk why i used this.
                    Ycut = A[Xmax][1]
                    Ycut1 = A[Xmin][1]
                else:
                    Ycut = A[Xmin][1]
                    Ycut1 = A[Xmax][1]
                Ycut = Ycut - 0.25  # lower the threshold so some curves are retained
                '''
                    save func
                arr_pts = np.asarray(Ycut)
                
                #np.save(file=save_path + 'Y_bottomcut_coords/'+directory[1] +'_'+ str(cross_sections[ind]), arr=arr_pts)
'''
                # to shrink array of points to desired shape
                B = np.zeros(A.shape)
                for counter in range(len(a[0][0])):
                    if top < A[
                        counter, 1] >= Ycut:  # condition for B array vars possible eg.= "top" - lower of 2 top lines, "Ytop_left" - self explanatory, "Ytop_right", "Ycut" - lower of bottom cuts, "Ycut1" - higher bottom cut
                        # conditions are for example "top< A[counter,1] >= Ycut" when A's y value is higher than top copy value into B
                        #                           "top>A[counter,1]>=Ycut" when A's y value is in between top and bottom threshold copy value into B
                        B[counter, :] = A[counter, :]
                B = B[~np.all(B == 0, axis=1)]  # removes the rows with 0

                ''' im not sure if this is the best way to achieve the corner cuts
                #Left corner
                for counter in range(len(a[0][0])/2): #only run values till half of the figure
                    if top>A[counter,1]>=Ycut: 
                        B[counter,:] = A[counter,:]
                B = B[~np.all(B == 0, axis=1)] #removes the rows with 0
                '''

                # saves the points from condition above
                FigureCoords = np.asarray(B)
                np.save(file=save_path + '1dfigs/' + directory[1] + '_' + str(cross_sections[ind]), arr=FigureCoords)
                # plotting
                fig = plt.figure()

                # display cut section
                x = np.zeros(np.shape(B)[0])  # make x and y to store
                y = np.zeros(np.shape(B)[0])
                for i in range(np.shape(B)[0]):  # range is num of rows
                    x[i] = (B[i, 0])  # read from A to x,y
                    y[i] = (B[i, 1])

                ''' Display entire tooth
                x = np.zeros(np.shape(A)[0])  # make x and y to store
                y = np.zeros(np.shape(A)[0])
                for i in range(np.shape(A)[0]):
                    x[i] = (A[i, 0])  # read from A to x,y
                    y[i] = (A[i, 1])
'''
                ax = fig.gca()
                ax.set_autoscale_on(False)  # allows us to define scale
                ax.plot(x, y, linewidth=1.0)
'''
                #plot lines for viewing
                x1 = range(-5,5)
                yleft = np.full((10,), Ytop_left)
                yright = np.full((10,), Ytop_right)
                ax.plot(x1, yleft, '-c')
                ax.plot(x1, yright, '-r')
                ybottom = np.full((10,), Ycut+.25)
                ax.plot(x1, ybottom, '-p')
                ybottom = np.full((10,), Ycut)
                ax.plot(x1, ybottom, '-y')
'''
ax.axis([-5, 5, -6, 7])  # scale to same size
# remove irrelevent plot information
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
# save to this folder
fig.savefig(save_path + 'figs1/' + directory[1] + '_' + str(cross_sections[ind]))
plt.close(fig=None)
