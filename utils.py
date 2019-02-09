import numpy as np
from scipy.io import loadmat, savemat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc


def rotmat2expmap(R):
    # R to omega
    theta = np.arccos((np.trace(R) - 1) / 2.0)
    if theta < 1e-6:
        A = np.zeros((3, 1))
    else:
        A = theta / (2 * np.sin(theta)) * np.array([[R[2, 1] - R[1, 2]], [R[0, 2] - R[2, 0]], [R[1, 0] - R[0, 1]]])

    return A


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
    # compute omega
    w = np.cross(u, v)
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-6:
        A = np.zeros(3)
    else:
        A = w/w_norm*np.arccos(np.dot(u, v))

    return A


def computelie(lie_params):
    # lie_params [N, 6], compute g_{1:N} [N, 4, 4]
    njoints = np.shape(lie_params)[0]
    A = np.zeros((njoints, 4, 4))

    for j in range(njoints):
        if j == 0:
            A[j, :, :] = lietomatrix(lie_params[j, 0: 3].T, lie_params[j, 3:6].T)
        else:
            A[j, :, :] = np.matmul(np.squeeze(A[j - 1, :, :]),
                                   lietomatrix(lie_params[j, 0:3].T, lie_params[j, 3:6].T))

    joint_xyz = np.zeros((njoints, 3))

    for j in range(njoints):
        coor = np.array([0, 0, 0, 1]).reshape((4, 1))
        xyz = np.matmul(np.squeeze(A[j, :, :]), coor)
        joint_xyz[j, :] = xyz[0:3, 0]

    return joint_xyz


def inverse_kinematics(joint_xyz, config):
    # joint_xyz [nframes, njoints*3]
    index = config.chain_idx
    nframes = joint_xyz.shape[0]
    joint_xyz = joint_xyz.reshape([nframes, -1, 3])
    njoints = joint_xyz.shape[1]

    lie_parameters = np.zeros((nframes, njoints, 6))

    for i in range(nframes):
        for k in range(len(index)):

            lie_parameters[i, index[k][0], 3:6] = joint_xyz[i, index[k][0], :]

            for j in range(len(index[k])-1):
                lie_parameters[i, index[k][j+1], 3] = np.linalg.norm(joint_xyz[i, index[k][j+1], :] - joint_xyz[i, index[k][j], :])

            for j in range(len(index[k])-2, -1, -1):
                v = np.squeeze(joint_xyz[i, index[k][j+1], :] - joint_xyz[i, index[k][j], :])
                vhat = v/np.linalg.norm(v)

                if j == 0:
                    uhat = np.array([1, 0, 0])
                else:
                    u = np.squeeze(joint_xyz[i, index[k][j], :] - joint_xyz[i, index[k][j-1], :])
                    uhat = u/np.linalg.norm(u)
                A = expmap2rotmat(findrot(np.array([1, 0, 0]), np.squeeze(uhat))).T
                B = expmap2rotmat(findrot(np.array([1, 0, 0]), np.squeeze(vhat)))
                lie_parameters[i, index[k][j], 0:3] = np.squeeze(rotmat2expmap(np.matmul(A, B)))

    return lie_parameters


def forward_kinematics(lie_params, config):
    # lie_params [nframes, njoints, 6]
    nframes = lie_params.shape[0]
    njoints = lie_params.shape[1]
    skip = config.skip

    joint_xyz_f = np.zeros([nframes, njoints, 3])

    for i in range(nframes):
        for j in range(len(skip)-1):
            joint_xyz_f[i, np.arange(skip[j], skip[j + 1]), :] = computelie(
                np.squeeze(lie_params[i, np.arange(skip[j], skip[j + 1]), :]))
    return joint_xyz_f


# Define the kinematic chain configuration
class bone_config(object):
    def __init__(self):
        """Define kinematic chain configurations"""
        self.chain_config = np.array([6, 6, 5, 5, 5])

        self.nchains = self.chain_config.shape[0]
        self.skip = np.zeros([self.nchains])

        for i in range(self.nchains):
            if i == 0:
                self.skip[i] = self.chain_config[i]
            else:
                self.skip[i] = self.skip[i - 1] + self.chain_config[i]

        self.skip = np.concatenate((np.array([0]), self.skip))
        self.skip = self.skip.astype(int)

        self.chain_idx = []
        for j in range(self.skip.shape[0] - 1):
            self.chain_idx.append(np.arange(self.skip[j], self.skip[j + 1]))

        self.idx = [0]
        for j in range(len(self.chain_idx)):
            self.idx.append(self.chain_idx[j][-1] - j)


class plot_human(object):

    def __init__(self, predict, labels, config):
        self.joint_xyz = labels
        self.nframes = labels.shape[0]
        self.joint_xyz_f = predict

        # set up the axes
        xmin = -1000
        xmax = 1000
        ymin = -1000
        ymax = 1000
        zmin = -1000
        zmax = 1000

        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax), projection='3d')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        self.chain = config.chain_idx
        self.scats = []
        self.lns = []

    def update(self, frame):
        for scat in self.scats:
            scat.remove()
        for ln in self.lns:
            self.ax.lines.pop(0)

        self.scats = []
        self.lns = []

        xdata = np.squeeze(self.joint_xyz[frame, :, 0])
        ydata = np.squeeze(self.joint_xyz[frame, :, 1])
        zdata = np.squeeze(self.joint_xyz[frame, :, 2])

        xdata_f = np.squeeze(self.joint_xyz_f[frame, :, 0])
        ydata_f = np.squeeze(self.joint_xyz_f[frame, :, 1])
        zdata_f = np.squeeze(self.joint_xyz_f[frame, :, 2])

        # self.scats.append(self.ax.scatter3D(xdata, ydata, zdata, color='b'))

        for i in range(len(self.chain)):
            self.lns.append(
                self.ax.plot3D(xdata_f[self.chain[i][:],], ydata_f[self.chain[i][:],], zdata_f[self.chain[i][:],],
                               linewidth=2.0, color='#f94e3e'))
            self.lns.append(self.ax.plot3D(xdata[self.chain[i][:],], ydata[self.chain[i][:],], zdata[self.chain[i][:],],
                                           linewidth=2.0, color='#0780ea'))

    def plot(self):
        ani = FuncAnimation(self.fig, self.update, frames=self.nframes, interval=40)
        plt.show()


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


def plot_cropped_3d_annotated_hand(xyz_pose, bbx, cropped_points):
    x_min, x_max, y_min, y_max, z_min, z_max = bbx
    plt.figure()
    ax = plt.axes(xlim=(x_min, x_max), ylim=(z_min, z_max), zlim=(y_min, y_max), projection='3d')
    ax.scatter(cropped_points[:, 0], cropped_points[:, 2], cropped_points[:, 1], color='b', marker='.', s=0.8,
               alpha=0.5)
    ax.scatter(xyz_pose[:, 0], xyz_pose[:, 2], xyz_pose[:, 1], color='r', marker='o', s=20)
    # ax.view_init(elev=0, azim=270)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    plt.show()


def plot_annotated_depth_img(depth_img, jnt_uvd, camera_cfg, max_depth):
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

