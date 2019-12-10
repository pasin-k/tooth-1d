import numpy as np
from stl import mesh


def get_cross_section(stl_file_name, z_plane, degree=None, augment=None, is_rearrange=True, axis=1):
    """
    Get cross-section of an stl file from selected x/y/z plane as well as cross-section of an rotated angle.
    We also have augmentation option to duplicate more data by rotating (Recommend small degree like 1,2,3)
    :param stl_file_name:   String, File name
    :param z_plane:         float or int, Selected plane (E.g. get cross-section at plane z = 0)
    :param degree:          List of degree of cross-section you want
    :param augment:         List of rotation degree to increase datasize, None will give only 0 degree
    :param is_rearrange:    Boolean, Rearrange coordinate from bottom left to right, else the data will be unordered
    :param axis:            Axis of rotation (0 = X,1 = Y,2 = Z) (Default at 1 for our data)
    :return: reP_all:       List of list of numpy array with size of [len(augment),len(degree),[N,2])
    """
    # Fetch data and get all triangles
    prep_mesh = mesh.Mesh.from_file(stl_file_name)
    tmain_temp = np.concatenate((prep_mesh.v0, prep_mesh.v1, prep_mesh.v2), axis=1)  # numpy array of N * 9 dimension
    if augment is not None:
        all_triangle = []
        for a in augment:
            all_triangle.append(rotatestl(tmain_temp, axis, a))
    else:
        all_triangle = [tmain_temp]

    if degree is None:  # Use default value
        degree = [0, 45, 90, 135]
        print("No degree input found, use default value")
    elif isinstance(degree, int) or isinstance(degree, float):  # In case of single value, put list over it
        degree = [degree]

    all_points = []
    for Tmain in all_triangle:  # Loop for every augmentation
        points = []  # Output
        for d in degree:
            P = np.empty([0, 3])  # Unarranged coordinates
            T = rotatestl(Tmain, axis, d).tolist()  # Default as Z-axis
            i = 1  # Special index added as a third column which will be used in 'rearrange' function
            while len(T) != 0:
                t = np.array(T.pop(0))  # Select some element from t
                Zcoor = np.array((t[2], t[5], t[8]))
                # Check if triangle is in the selected plane
                if (not ((Zcoor[0] < z_plane and Zcoor[1] < z_plane and Zcoor[2] < z_plane) or
                         (Zcoor[0] > z_plane and Zcoor[1] > z_plane and Zcoor[2] > z_plane))):
                    idxUp = np.argwhere(Zcoor > z_plane)  # Index of vertex ABOVE plane
                    idxDown = np.argwhere(Zcoor < z_plane)  # Index of vertex BELOW plane
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
                    l1 = slicecoor(z_plane, vtOne, vtTwo1, i)
                    l2 = slicecoor(z_plane, vtOne, vtTwo2, i)
                    l = np.array([l1, l2])  # This is [2,3] size numpy array of two coordinates that intersect Z plane
                    P = np.concatenate((P, l), axis=0)  # Accumulate all intersections
                    i = i + 1
            if is_rearrange:
                # print("%s_%s_%s" % (stl_file_name, d, augment))
                new_points = rearrange(P)
                if new_points is None:
                    print(
                        "getSlicer: %s has problem getting cross-section. Possible hole appeared in model" % stl_file_name)
                    points = None
                    break
                else:
                    points.append(new_points)
            else:
                points.append(P)
        all_points.append(points)
    return all_points


def rotatestl(point, axis, deg):
    """
    Rotate stl model (Data need to be in numpy format)
    :param point: Stl data point, ndarray of (N,9)
    :param axis: int, axis to rotate (axis X:0, Y:1, Z:2)
    :param deg: int or float, degree to rotate
    :return: ndarray (N,9)
    """
    # We default on rotate over Z-axis, thus rotate x-y-z first then applied Z-axis rotation
    V = np.zeros([np.shape(point)[0], 9])
    out = np.zeros([np.shape(point)[0], 9])
    if axis == 0:
        for i in range(0, 3):
            V[:, 3 * i + 0] = point[:, 3 * i + 1]  # New X axis is the data Y axis
            V[:, 3 * i + 1] = point[:, 3 * i + 2]  # New Y axis is the data Z axis
            V[:, 3 * i + 2] = point[:, 3 * i + 0]  # New Z axis is the data X axis *We rotate this axis now*
    elif axis == 1:
        for i in range(0, 3):
            V[:, 3 * i + 0] = point[:, 3 * i + 2]  # New X axis is the data Z axis
            V[:, 3 * i + 1] = point[:, 3 * i + 0]  # New Y axis is the data X axis
            V[:, 3 * i + 2] = point[:, 3 * i + 1]  # New Z axis is the data Y axis *We rotate this axis now*
    elif axis == 2:
        V = point
    else:
        raise ValueError("axis need to be 0,1,2. Found %s" % axis)
    # From now we assume that we want to rotate around Z-axis only
    rot_mat = np.array([[np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg)), 0],
                        [-np.sin(np.deg2rad(deg)), np.cos(np.deg2rad(deg)), 0],
                        [0, 0, 1]])  # Rotation matrix, look up on wiki for more info
    V[:, 0:3] = np.matmul(V[:, 0:3], rot_mat)
    V[:, 3:6] = np.matmul(V[:, 3:6], rot_mat)
    V[:, 6:9] = np.matmul(V[:, 6:9], rot_mat)
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


def rearrange(point):
    """
    Rearrange order of point from randomly scattered by connecting same point
    This will fail if the cross-section is not connected into one piece
    :param point: ndarray (N,2), unarranged
    :return: ndarray (N,2), rearranged, None if fail
    """
    point = np.round(point, 8)
    np.set_printoptions(threshold=np.inf)
    ind = np.lexsort((point[:, 1], point[:, 0]))
    point = point[ind]  # Sort coordinate from left to right

    # Finding the starting and ending rows
    start_row = np.array([-1, -1])
    start_coor = np.zeros([2, 3])
    for i in range(np.int32(np.round(np.size(point, 0) / 2))):
        if start_row[0] == -1:
            if not (np.array_equal(point[2 * i, 0:2], point[2 * i + 1, 0:2])):
                start_row[0] = 2 * i
                start_coor[0, :] = point[2 * i, :]
        else:
            if not (np.array_equal(point[2 * i - 1, 0:2], point[2 * i, 0:2])):
                start_row[1] = 2 * i - 1
                start_coor[1, :] = point[2 * i - 1, :]
                break
    if start_row[1] == -1:  # This case occur when final point is last row
        start_row[1] = np.round(np.size(point, 0)) - 1
        start_coor[1, :] = point[np.round(np.size(point, 0)) - 1, :]
    # Set the left most one as startRow[0]
    if point[start_row[0], 0] > point[start_row[1], 0]:
        row_temp = start_row[0]
        start_row[0] = start_row[1]
        start_row[1] = row_temp

    # Rearranging coordinate starting from startRow[0]
    re_point = np.zeros([np.int32(np.round(np.size(point, 0)) / 2) + 1, 2])  # Rearranged points (Output)
    curr_line = point[start_row[0], 2]  # Current line that we are interested in, starting from first point
    re_point[0, :] = point[start_row[0], 0:2]
    i = 1  # Start at 1 because index 0 is already initialized
    while True:
        # Find the pair coordinate of the same line
        same_line = point[np.where(point[:, 2] == curr_line)]
        if np.array_equal(re_point[i - 1, :], same_line[0, 0:2]):
            re_point[i, :] = same_line[1, 0:2]
        else:
            re_point[i, :] = same_line[0, 0:2]
        # Find the line that connect to the current line
        same_point = point[np.where((point[:, 0] == re_point[i, 0]) & (point[:, 1] == re_point[i, 1]))]
        if np.array_equal(same_point[0], point[start_row[1], :]) and (same_point.shape[0] == 1):
            if np.array_equal(re_point[-1, :], [0, 0]):
                print("WARNING: Not all expected coordinate are discovered. Some (0,0) rows may occur")
            break
        if curr_line == same_point[0, 2]:
            if np.shape(same_point)[0] == 1:  # This happens if there's a hole in the cross-section
                return None
            curr_line = same_point[1, 2]
        else:
            curr_line = same_point[0, 2]
        i = i + 1
        if i == np.int32(np.round(np.size(point, 0)) / 2) + 1:
            print("WARNING: Too many coordinate than expected. Possible bug loop happened")
            break
    return re_point


def slicecoor(z_plane, point_a, point_b, i):
    """
    Get 2D coordinate of an intersection on Zplane which is between two 3D coordinates
    :param z_plane: int or float, z_plane
    :param point_a: ndarray of first 3d point (shape: (1,3))
    :param point_b: ndarray of second 3d point (shape: (1,3))
    :param i: scalar, addition value to represent which # of line
    :return: ndarray of x,y coordinate plus i (shape:(1,3))
    """
    coor = np.array(
        [point_b[0] + (z_plane - point_b[2]) * (point_a[0] - point_b[0]) / (point_a[2] - point_b[2]), point_b[1] +
         (z_plane - point_b[2]) * (point_a[1] - point_b[1]) / (point_a[2] - point_b[2]), i])
    return coor


def read_data(file_dir):
    all_points = None
    num_points = 2048  # Number of point to retrieve
    for file in file_dir:
        data = mesh.Mesh.from_file(file)
        point = stl_to_point(v1=data.v0, v2=data.v2, v3=data.v1, num_points=num_points)  # Order to get upright shape
        point = np.expand_dims(point, axis=0)
        if all_points is None:
            all_points = point
        else:
            all_points = np.concatenate((all_points, point), axis=0)
    return all_points


def stl_to_point(v1, v2, v3, num_points, sampling_mode="weight"):
    """
    Function to convert stl file into point cloud
    https://medium.com/@daviddelaiglesiacastro/3d-point-cloud-generation-from-3d-triangular-mesh-bbb602ecf238
    :param v1, v2, v3 : (N,3) ndarrays, vi represent x,y,z coordinates of one vertex
    :param num_points: Number of points we want to sample
    :param sampling_mode: String, type of sampling from triangle, recommend "weight"
    :return: points: numpy array of point cloud
    """
    print_data = False
    if not (np.shape(v1)[0] == np.shape(v2)[0] == np.shape(v3)[0]):
        raise ValueError("Size of all three vertex is not the same")
    else:
        if print_data:
            print("Number of mesh: %s" % np.shape(v1)[0])
    areas = triangle_area_multi(v1, v2, v3)
    prob = areas / areas.sum()
    if sampling_mode == "weight":
        indices = np.random.choice(range(len(areas)), size=num_points, p=prob)
    else:
        indices = np.random.choice(range(len(areas)), size=num_points)
    points = select_point_from_triangle(v1[indices, :], v2[indices, :], v3[indices, :])
    return points


def triangle_area_multi(v1, v2, v3):
    """
    Find area of triangle, used for finding weights
    :param v1, v2, v3 : (N,3) ndarrays, vi represent x,y,z coordinates of one vertex
    :return: size of triangle
    """
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)


def select_point_from_triangle(v1, v2, v3):
    """
    Select one point from each three vertex
    :param v1, v2, v3 : (N,3) ndarrays, vi represent x,y,z coordinates of one vertex
    :return: ndarrays
    """
    n = np.shape(v1)[0]
    u = np.random.rand(n, 1)
    v = np.random.rand(n, 1)
    is_a_problem = u + v > 1

    u[is_a_problem] = 1 - u[is_a_problem]
    v[is_a_problem] = 1 - v[is_a_problem]

    w = 1 - (u + v)

    points = (v1 * u) + (v2 * v) + (v3 * w)

    return points
