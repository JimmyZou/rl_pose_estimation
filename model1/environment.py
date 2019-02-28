import numpy as np
import utils
import copy
import collections


class HandEnv(object):
    def __init__(self, dataset, subset, obs_width=(30, 30, 20), iter_per_joint=1, reward_beta=0.1):
        self.dataset = dataset
        assert subset in ["training", "testing"]
        self.subset = subset  # ["training", "testing"]
        self.home_pose, self.chains_idx, self.root_idx = self.init_home_pose()
        self.iter_per_joint = iter_per_joint
        self.num_chains = len(self.chains_idx)
        self.num_joint = self.home_pose.shape[0]
        self.obs_width = obs_width
        self.reward_beta = reward_beta

        # data of an example
        self.pose = copy.copy(self.home_pose)
        self.filename = None
        self.orig_pose = None
        self.depth_img = None
        self.orig_bbx = None
        self.orig_points = None
        self.rotate_mat = None
        self.norm_pose = None
        self.norm_points = None
        self.norm_bbx = None
        self.volume = None
        self.current_chain = 0
        self.current_joint = 0
        self.current_iter = 0
        self.dis2targ = collections.defaultdict(list)

    def set_iter_per_joint(self, iterations):
        self.iter_per_joint = iterations

    def init_home_pose(self):
        if self.dataset == 'NYU':
            z = 60
            home_pose = np.array([
                [50, 10, z],  # 1-2
                [100, 10, z],
                [30, 40, z],  # 3-4
                [80, 40, z],
                [10, 70, z],  # 5-6
                [60, 70, z],
                [30, 100, z],  # 7-8
                [80, 100, z],
                [100, 120, z],  # 9-11
                [120, 120, z],
                [160, 120, z],
                [220, 20, z],  # 12-13
                [220, 70, z],
                [150, 60, z]  # 14
            ], dtype=np.float32)
            root_idx = 13
            chains_idx = [[13], [12], [11], [10, 9, 8], [7, 6], [5, 4], [3, 2], [1, 0]]
        elif self.dataset == 'MRSA15':
            z = 60
            home_pose = np.array([
                [150, 60, z],  # 1
                [100, 80, z],  # 2-5
                [75, 80, z],
                [50, 80, z],
                [25, 80, z],
                [90, 60, z],  # 6-9
                [50, 60, z],
                [30, 60, z],
                [10, 60, z],
                [90, 40, z],  # 10-13
                [60, 40, z],
                [35, 40, z],
                [15, 40, z],
                [100, 20, z],  # 14-17
                [70, 20, z],
                [55, 20, z],
                [40, 20, z],
                [135, 75, z],  # 18-21
                [120, 85, z],
                [100, 90, z],
                [85, 90, z]
            ], dtype=np.float32)
            root_idx = 0
            chains_idx = [[0], [17, 18, 19, 20], [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        elif self.dataset == 'ICVL':
            z = 30
            home_pose = np.array([
                [50, 40, z],  # 1
                [30, 50, z],  # 2-4
                [25, 80, z],
                [30, 110, z],
                [70, 80, z],  # 5-7
                [90, 110, z],
                [100, 120, z],
                [90, 70, z],  # 8-10
                [120, 85, z],
                [130, 90, z],
                [90, 50, z],  # 11-13
                [120, 60, z],
                [130, 65, z],
                [90, 25, z],  # 14-16
                [110, 30, z],
                [120, 35, z]
            ], dtype=np.float32)
            root_idx = 0
            chains_idx = [[0], [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
        else:
            raise ValueError('Unknown subset %s' % self.subset)
        return home_pose, chains_idx, root_idx

    def reset(self, example):
        """
        load a new example in the environment
        """
        # example: (filename, xyz_pose, depth_img, pose_bbx, cropped_points, coeff,
        #           normalized_rotate_pose, normalized_rotate_points, rotated_bbx, volume)
        self.filename = example[0]
        self.orig_pose = example[1]
        self.depth_img = example[2]
        self.orig_bbx = example[3]
        self.orig_points = example[4]
        self.rotate_mat = example[5]
        self.norm_pose = example[6]
        self.norm_points = example[7]
        self.norm_bbx = example[8]
        self.volume = example[9]
        self.pose = copy.copy(self.home_pose)
        # control which joint is manipulated now
        self.current_chain = 0
        self.current_joint = 0
        self.current_iter = 0
        self.dis2targ.clear()

    def step(self, ac):
        # ac is lie parameter [w, t] for current joint
        ac = np.squeeze(ac)
        SE_mat = utils.lietomatrix(ac[0:3], ac[3:6])
        curr_joint_idx = self.chains_idx[self.current_chain][self.current_joint]

        # obtain the initial distance for reward
        target_xyz = self.norm_pose[curr_joint_idx, :]
        if not self.dis2targ[curr_joint_idx]:
            # self.dis2targ[curr_joint_idx] is empty
            if self.current_chain == 0:
                init_root_xyz = self.pose[curr_joint_idx, :]
                root_dis = np.linalg.norm(target_xyz - init_root_xyz)
                other_xyz = np.delete(self.pose, curr_joint_idx, axis=0)
                target_other_xyz = np.delete(self.norm_pose, curr_joint_idx, axis=0)
                other_dis = np.mean(np.linalg.norm(other_xyz - target_other_xyz, axis=1))
                self.dis2targ[curr_joint_idx].append((root_dis, other_dis))
            else:
                init_xyz = self.pose[curr_joint_idx, :]
                self.dis2targ[curr_joint_idx].append(np.linalg.norm(target_xyz - init_xyz))

        # define reward for root joint or chain joints
        if self.current_chain == 0:
            # root joint, adjust the position of all joints as well
            assert curr_joint_idx == self.root_idx
            root_coor = self.pose[curr_joint_idx, :]
            pose = np.concatenate([self.pose - root_coor, np.ones([self.num_joint, 1])], axis=1)
            pose = np.matmul(SE_mat, pose.T).T
            self.pose = pose[:, 0:3] + root_coor

            # define root reward
            pred_xyz = self.pose[curr_joint_idx, :]
            root_dis = np.linalg.norm(target_xyz - pred_xyz)
            other_xyz = np.delete(self.pose, curr_joint_idx, axis=0)
            target_other_xyz = np.delete(self.norm_pose, curr_joint_idx, axis=0)
            other_dis = np.mean(np.linalg.norm(other_xyz - target_other_xyz, axis=1))
            self.dis2targ[curr_joint_idx].append((root_dis, other_dis))
            r = (self.dis2targ[curr_joint_idx][-2][0] - self.dis2targ[curr_joint_idx][-1][0]) + \
                self.reward_beta * (self.dis2targ[curr_joint_idx][-2][1] - self.dis2targ[curr_joint_idx][-1][1])

        else:
            # finger branch chain, adjust the position of sub-sequential joint on the chain
            current_chain = self.chains_idx[self.current_chain]
            root_coor = self.pose[curr_joint_idx, :]
            subseq_joints = current_chain[self.current_joint:]
            # print(curr_joint_idx, adjust_joints, i)
            pose = np.concatenate([self.pose[subseq_joints, :] - root_coor, np.ones([len(subseq_joints), 1])], axis=1)
            pose = np.matmul(SE_mat, pose.T).T
            self.pose[subseq_joints, :] = pose[:, 0:3] + root_coor

            # define reward for joints on a finger
            pred_xyz = self.pose[curr_joint_idx, :]
            distance = np.linalg.norm(target_xyz - pred_xyz)
            self.dis2targ[curr_joint_idx].append(distance)
            r = self.dis2targ[curr_joint_idx][-2] - self.dis2targ[curr_joint_idx][-1]
        r = r/10
        return r

    def get_obs(self):
        # decide the next joint
        all_done, is_root, chain_done, joint_done = False, False, False, False
        current_chain = self.chains_idx[self.current_chain]
        if self.current_joint == len(current_chain) - 1 and self.current_iter == self.iter_per_joint - 1:
            self.current_iter += 1
            chain_done = True
            if self.current_chain == len(self.chains_idx) - 1:
                all_done = True
        elif self.current_iter < self.iter_per_joint:
            self.current_iter += 1
        else:
            self.current_iter = 1
            if self.current_joint < len(current_chain) - 1:
                self.current_joint += 1
            else:
                self.current_joint = 0
                self.current_chain += 1
        if self.current_chain == 0:
            is_root = True

        # get the local volumetric observation
        curr_joint_idx = self.chains_idx[self.current_chain][self.current_joint]
        joint_position = self.pose[curr_joint_idx, :]
        print('current joint idx: %i' % curr_joint_idx)
        local_obs = self.crop_volume(self.volume, joint_position, self.obs_width)
        return local_obs, all_done, chain_done, is_root

    @staticmethod
    def crop_volume(volume, center_coor, width):
        if type(width) == float:
            w = [int(width)] * 3
        elif type(width) == int:
            w = [int(width)] * 3
        else:
            w = width

        x_max, y_max, z_max = volume.shape
        c = np.asarray(center_coor, dtype=np.int16)
        print('joint coordination:', c)
        local_obs = np.zeros([2 * w[0] + 1, 2 * w[1] + 1, 2 * w[2] + 1], dtype=np.int8)
        if (x_max + w[0] < c[0]) or (y_max + w[1] < c[1]) or (z_max + w[2] < c[2]):
            # the cropped space is out of the volume
            pass
        elif (w[0] < - c[0]) or (w[1] < - c[1]) or (w[2] < - c[2]):
            # the cropped space is out of the volume
            pass
        else:
            v_x_min = max(0, c[0] - w[0])
            v_x_max = min(x_max, c[0] + w[0] + 1)
            v_y_min = max(0, c[1] - w[1])
            v_y_max = min(y_max, c[1] + w[1] + 1)
            v_z_min = max(0, c[2] - w[2])
            v_z_max = min(z_max, c[2] + w[2] + 1)

            obs_x_min = w[0] + v_x_min - c[0]
            obs_x_max = w[0] + v_x_max - c[0]
            obs_y_min = w[1] + v_y_min - c[1]
            obs_y_max = w[1] + v_y_max - c[1]
            obs_z_min = w[2] + v_z_min - c[2]
            obs_z_max = w[2] + v_z_max - c[2]

            local_obs[obs_x_min: obs_x_max, obs_y_min: obs_y_max, obs_z_min: obs_z_max] = \
                volume[v_x_min: v_x_max, v_y_min: v_y_max, v_z_min: v_z_max]

            # print('volume', v_x_min, v_x_max, v_y_min, v_y_max, v_z_min, v_z_max)
            # print('local obs', obs_x_min, obs_x_max, obs_y_min, obs_y_max, obs_z_min, obs_z_max)
        return local_obs


def in_test():
    # env = HandEnv(dataset='NYU', subset='training', iter_per_joint=2)
    # while not env.get_obs()[1]:
    #     pass

    from data_preprocessing.mrsa_dataset import MRSADataset
    reader = MRSADataset(test_fold='P0', subset='pre-processing', num_cpu=4,
                         num_imgs_per_file=600, root_dir='../../data/mrsa15/')
    reader.load_annotation()
    example = reader.convert_to_example(reader._annotations[10])
    env = HandEnv(dataset='MRSA15', subset='training', iter_per_joint=1)
    env.reset(example)

    # from data_preprocessing.nyu_dataset import NYUDataset
    # reader = NYUDataset(subset='pps-testing', num_cpu=4, num_imgs_per_file=600, root_dir='../../data/nyu/')
    # reader.load_annotation()
    # example = reader.convert_to_example(reader._annotations[10])
    # env = HandEnv(dataset='NYU', subset='training', iter_per_joint=3)
    # env.reset(example)

    reader.plot_skeleton(None, env.pose)

    env.get_obs()
    r = env.step(np.array([0, 0, np.pi / 4, 10, 0, 0]))
    print(r)
    reader.plot_skeleton(None, env.pose)

    env.get_obs()
    r = env.step(np.array([0, 0, np.pi / 2, 0, 20, 0]))
    print(r)
    reader.plot_skeleton(None, env.pose)

    env.get_obs()
    r = env.step(np.array([0, 0, np.pi / 2, 0, -20, 0]))
    print(r)
    reader.plot_skeleton(None, env.pose)

    env.get_obs()
    r = env.step(np.array([0, 0, np.pi / 2, 10, 0, 0]))
    print(r)
    reader.plot_skeleton(None, env.pose)

    env.get_obs()
    r = env.step(np.array([0, 0, np.pi / 2, 10, 0, 0]))
    print(r)
    reader.plot_skeleton(None, env.pose)

    print(env.dis2targ)

    pass


if __name__ == '__main__':
    in_test()

