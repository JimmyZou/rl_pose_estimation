import numpy as np
from matplotlib import pyplot as plt
import pickle
import tensorflow as tf


def print_args(args):
    """ Prints the argparse argmuments applied
    Args:
      args = parser.parse_args()
    """
    max_length = max([len(k) for k, _ in args.items()])
    for k, v in args.items():
        print(' ' * (max_length-len(k)) + k + ': ' + str(v))


def str2int_tuple(args):
    for key, value in args.items():
        if type(value) == str and value[0] == '(' and value[-1] == ')':
            tmp = tuple(map(int, value[1:-1].split(',')))
            args[key] = tmp
    return args


def saveToFlat(var_list, param_pkl_path):
    # get all the values
    var_values = np.concatenate([v.flatten() for v in tf.get_default_session().run(var_list)])
    pickle.dump(var_values, open(param_pkl_path, "wb"))


def load_from_file(param_pkl_path):
    with open(param_pkl_path, 'rb') as f:
        params = pickle.load(f)
    return params.astype(np.float32)


def loadFromFlat(var_list, param_pkl_path):
    flat_params = load_from_file(param_pkl_path)
    # print("the type of the parameters stored is", flat_params.dtype)
    shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    theta = tf.placeholder(tf.float32, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = int(np.prod(shape))
        print(v.name)
        assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns)
    tf.get_default_session().run(op, {theta: flat_params})


def expmap2rotmat(A):
    # omega to R
    theta = np.linalg.norm(A)
    if theta == 0:
        R = np.identity(3)
    else:
        A = A / theta
        cross_matrix = np.array([[0, -A[2], A[1]], [A[2], 0, -A[0]], [-A[1], A[0], 0]])
        R = np.identity(3) + np.sin(theta) * cross_matrix + (1 - np.cos(theta)) * np.matmul(cross_matrix, cross_matrix)
    return R


def lietomatrix(angle, trans):
    # xi to exp^{xi}, se(3) to SE(3)
    R = expmap2rotmat(angle)
    T = trans
    SEmatrix = np.concatenate((np.concatenate((R, T.reshape(3, 1)), axis=1), np.array([[0, 0, 0, 1]])))
    return SEmatrix


def plot_3d_points(data):
    # data [num_points, 3]
    data[:, 1] = - data[:, 1]
    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
    z_min, z_max = np.min(data[:, 2]), np.max(data[:, 2])
    plt.figure()
    ax = plt.axes(xlim=(x_min, x_max), ylim=(z_min, z_max), zlim=(y_min, y_max), projection='3d')
    ax.scatter(data[:, 0], data[:, 2], data[:, 1], color='b', marker='.', s=1)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.grid()
    plt.show()


def plot_cropped_3d_annotated_hand(xyz_pose, bbx, cropped_points, view=None):
    plt.figure()
    if bbx is None:
        ax = plt.axes(projection='3d')
    else:
        x_min, x_max, y_min, y_max, z_min, z_max = bbx
        ax = plt.axes(xlim=(x_min, x_max), ylim=(z_min, z_max), zlim=(y_min, y_max), projection='3d')
    ax.scatter(cropped_points[:, 0], cropped_points[:, 2], cropped_points[:, 1], color='b', marker='.', s=1,
               alpha=0.5)
    ax.scatter(xyz_pose[:, 0], xyz_pose[:, 2], xyz_pose[:, 1], color='r', marker='o', s=20)
    if view is not None:
        ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    plt.show()


def plot_annotated_depth_img(depth_img, jnt_uvd):
    # plot 2d gray image
    _depth_img = depth_img / np.max(depth_img)
    plt.figure()
    plt.imshow(_depth_img, cmap="gray")
    if jnt_uvd is not None:
        plt.scatter(jnt_uvd[:, 0], jnt_uvd[:, 1], c='b', s=3)
    plt.axis('off')
    plt.show()


def depth2uvd(depth):
    x, y = np.meshgrid(np.linspace(0, depth.shape[1] - 1, depth.shape[1]),
                       np.linspace(0, depth.shape[0] - 1, depth.shape[0]))
    uvd = np.stack([x, y, depth], axis=2)
    return uvd


def uvd2xyz(uvd, camera_cfg):
    """
    convert uvd coordinates to xyz
    return:
        points in xyz coordinates, shape [N, 3]
    """
    # fx, fy, cx, cy, w, h
    # 0,  1,  2,  3,  4, 5
    # z = d
    # x = (u - cx) * d / fx
    # y = (v - cy) * d / fy
    _bpro = lambda pt2, cfg : [(pt2[0] - cfg[2]) * pt2[2] / cfg[0], (pt2[1] - cfg[3]) * pt2[2] / cfg[1], pt2[2]]
    uvd = np.reshape(uvd, [-1, 3])
    xyz = [_bpro(pt2, camera_cfg) for pt2 in uvd]
    return np.array(xyz)


def xyz2uvd(xyz, camera_cfg):
    # fx, fy, cx, cy, w, h
    # 0,  1,  2,  3,  4, 5
    # d = z
    # u = fx * x / z + cx
    # v = fy * y / z + cy
    _pro = lambda pt3, cfg: [pt3[0] * cfg[0] / pt3[2] + cfg[2], pt3[1] * cfg[1] / pt3[2] + cfg[3], pt3[2]]
    xyz = xyz.reshape((-1, 3))
    # perspective projection function
    uvd = [_pro(pt3, camera_cfg) for pt3 in xyz]
    return np.array(uvd)

