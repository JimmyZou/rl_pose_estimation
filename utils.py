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


"""=================Lie groups related functions=================="""


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


def findrot(u, v):
    if (v == np.array([-1, 0, 0])).all():
        return np.array([0, 0, np.pi])
    # compute omega
    w = np.cross(u, v)
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-6:
        A = np.zeros(3)
    else:
        A = w / w_norm * np.arccos(np.dot(u, v))
    return A


def rotmat2expmap(R):
    # R to omega
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1, 1))
    if theta < 1e-6:
        A = np.zeros((3, 1))
    else:
        A = theta / (2 * np.sin(theta)) * np.array([[R[2, 1] - R[1, 2]], [R[0, 2] - R[2, 0]], [R[1, 0] - R[0, 1]]])
    return A


def inverse_kinematics(joint_xyz, chain_idx, warning=False):
    # joint_xyz: [num_joints, 3]
    njoints = joint_xyz.shape[0]
    lie_paras = np.zeros([njoints, 6])

    root_idx = chain_idx[0][0]
    lie_paras[root_idx, 3:6] = joint_xyz[root_idx, :]

    for _idx in chain_idx[1:]:
        # append root joint idx
        idx = [root_idx] + _idx
        # t
        for j in range(len(idx)-1):
            lie_paras[idx[j+1], 3] = np.linalg.norm(joint_xyz[idx[j+1], :] - joint_xyz[idx[j], :])
        # omega
        for j in range(len(idx)-2, -1, -1):
            v = joint_xyz[idx[j+1], :] - joint_xyz[idx[j], :]
            vhat = v / np.linalg.norm(v)

            if j == 0:
                uhat = np.array([1, 0, 0])
            else:
                u = joint_xyz[idx[j], :] - joint_xyz[idx[j-1], :]
                uhat = u / np.linalg.norm(u)
            A = expmap2rotmat(findrot(np.array([1, 0, 0]), uhat))
            B = expmap2rotmat(findrot(np.array([1, 0, 0]), vhat))
            lie_paras[idx[j+1], 0:3] = np.squeeze(rotmat2expmap(np.matmul(A.T, B)))
    return lie_paras


def computelie(lie_params):
    # lie_params [N, 6], compute g_{1:N} [N, 4, 4]
    njoints = np.shape(lie_params)[0]
    A = np.zeros((njoints, 4, 4))

    for j in range(njoints):
        if j == 0:
            A[j, :, :] = lietomatrix(lie_params[j+1, 0:3], lie_params[j, 3:6])
        elif j == njoints-1:
            tmp = lietomatrix(np.array([0, 0, 0]), lie_params[j, 3:6])
            A[j, :, :] = np.matmul(A[j-1, :, :], tmp)
        else:
            tmp = lietomatrix(lie_params[j+1, 0:3], lie_params[j, 3:6])
            A[j, :, :] = np.matmul(A[j-1, :, :], tmp)

    tmp_coor = np.array([0, 0, 0, 1]).reshape((4, 1))
    joint_xyz = np.matmul(A, tmp_coor)[:, 0:3, 0]
    return joint_xyz


def forward_kinematics(lie_paras, chain_idx):
    # lie_params [njoints, 6]
    njoints = lie_paras.shape[0]
    joint_xyz_f = np.zeros([njoints, 3])

    root_idx = chain_idx[0][0]
    for k, _idx in enumerate(chain_idx):
        if k == 0:
            # root joint
            A = lietomatrix(lie_paras[root_idx, 0:3], lie_paras[root_idx, 3:6])
            tmp = np.array([0, 0, 0, 1]).reshape((4, 1))
            joint_xyz_f[root_idx, :] = np.matmul(A, tmp)[0:3, 0]
        else:
            # append root joint idx
            idx = [root_idx] + _idx
            joint_xyz = computelie(lie_paras[idx, :])
            joint_xyz_f[idx[1:], :] = joint_xyz[1:]
    return joint_xyz_f


"""=================hand pose related functions=================="""


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

