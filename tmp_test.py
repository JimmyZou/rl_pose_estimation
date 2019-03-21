from utils import *


def inverse_kinematics(joint_xyz, chain_idx):
    # joint_xyz: [num_joints, 3]
    # chain_idx: [[13], [12], [11], [10, 9, 8], [7, 6], [5, 4], [3, 2], [1, 0]]
    # chains_idx [[0], [17, 18, 19, 20], [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
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


from model2.environment import HandEnv
from data_preprocessing.nyu_dataset import NYUDataset
dataset = NYUDataset(subset='training', root_dir='/home/data/nyu/', predefined_bbx=(63, 63, 31))
env = HandEnv(dataset='nyu',
              subset='training',
              max_iters=5,
              predefined_bbx=dataset.predefined_bbx)

# from model2.environment import HandEnv
# from data_preprocessing.mrsa_dataset import MRSADataset
# dataset = MRSADataset(subset='training', test_fold='P0',
#                       root_dir='/hand_pose_data/mrsa15/', predefined_bbx=(63, 63, 31))
# env = HandEnv(dataset='mrsa15',
#               subset='training',
#               max_iters=5,
#               predefined_bbx=dataset.predefined_bbx)

# from model2.environment import HandEnv
# from data_preprocessing.icvl_dataset import ICVLDataset
# dataset = ICVLDataset(subset='training', root_dir='/hand_pose_data/icvl/', predefined_bbx=(63, 63, 31))
# env = HandEnv(dataset='icvl',
#               subset='training',
#               max_iters=5,
#               predefined_bbx=dataset.predefined_bbx)



dataset.plot_skeleton(None, env.home_pose)
lie_paras = inverse_kinematics(env.home_pose, env.chains_idx)
print(lie_paras)
joint_xyz = forward_kinematics(lie_paras, env.chains_idx)
dataset.plot_skeleton(None, joint_xyz)
# print(env.home_pose)
# print(joint_xyz)

# print(expmap2rotmat(np.array([0, 0, 2.60117315])))
# print(expmap2rotmat(np.array([0, 0, -2.60117315])))















