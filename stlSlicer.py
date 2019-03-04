import numpy as np
# import stl
from stl import mesh
import sys
import matplotlib as mpl

mpl.use('TkAgg')  # Use this so we can use matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

v = '1.8.1'
# 1.3 Finished rearrange function change name to reAr, output as finalP , create slicecoor function for simplicity
# 1.4 Works for pycharm now
# 1.5 Add rotatestl to rotate 1 degree
# 1.6 Debug rotate stl, now working
# 1.7 Add augmentation
# 1.8 More stable sorting + more catch exception case
print("stlSlicer.py version: " + str(v))


# Get list of coordinates from the slice np.array(X,2) from 'a single stl file'; X is varies
# Input:    stlfilename -> String, Folder name
#           Zplane      -> Scalar value, Selected plane,
#           Degree      -> List of rotation degree
#           augment     -> Boolean, If true, will rotate 180 degree
#           reAr        -> Boolean, Rearrange coordinate from bottom left to right
#           axis        -> Axis of rotation (0 = X,1 = Y,2 = Z) (Default at 1, based from preliminary result)
# Output:   List of numpy array of coordinates, each element in layer list represent each degree rotated
def getSlicer(stl_file_name, Zplane, Degree, augment, reAr=True, axis=1):
    # Import data and change to np array of N * 9 dimension
    # print(stl_file_name)
    prep_mesh = mesh.Mesh.from_file(stl_file_name)

    # Tmain is list of all triangles
    # Tmains = np.concatenate((prep_mesh.v0, prep_mesh.v1, prep_mesh.v2), axis=1).tolist()  # list of N * 9 dimension
    Tmain = np.concatenate((prep_mesh.v0, prep_mesh.v1, prep_mesh.v2), axis=1)  # numpy array of N * 9 dimension
    if augment:
        Tmain = augmented(Tmain)
    reP = list()  # Output
    # vtOne = np.zeros([3])  # vertex #1
    # vtTwo1 = np.zeros([3])  # vertex #2
    # vtTwo2 = np.zeros([3])  # vertex #3
    for d in Degree:
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
        if reAr:
            # print("%s_%s_%s" % (stl_file_name, d, augment))
            newP = rearrange(P)
            if newP is None:
                print("getSlicer: %s has problem getting cross-section. Possible hole appeared in model" % stl_file_name)
                return None
            else:
                reP.append(newP)
        else:
            reP.append(P)
    return reP


# Double amount by doing augmentations by rotating 180 degree
# Input:    T   ->  Numpy array of triangles coordinate ([N,9] Matrix)
# Output:   Taug->  Augmented triangles ([2*N,9] Matrix)
def augmented(T):
    Taug = rotatestl(T, 1, 180)
    return Taug


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
    np.set_printoptions(threshold=np.nan)
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
