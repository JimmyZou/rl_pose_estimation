import numpy as np
import utils
import copy


class HandEnv(object):
    def __init__(self, dataset, subset, max_iters, predefined_bbx):
        self.dataset = dataset
        assert subset in ["training", "testing"]
        self.subset = subset  # ["training", "testing"]
        self.predefined_bbx = (predefined_bbx[0] + 1, predefined_bbx[1] + 1, predefined_bbx[2] + 1)
        self.home_pose, self.chains_idx, self.root_idx = self.init_home_pose()
        # the first element is the iterations for root joint, second for chain joints
        self.num_chains = len(self.chains_idx)
        self.num_joint = self.home_pose.shape[0]
        self.max_iters = max_iters
        self.current_iter = 0
        self.resize_ratio = np.array([[1., 1., 1.]])

        # data of an example
        self.pose = copy.copy(self.home_pose)
        self.filename = None
        self.orig_pose = None
        self.depth_img = None
        self.orig_bbx = None
        self.rotate_mat = None
        self.norm_target_pose = None
        self.norm_points = None
        self.norm_bbx = None
        self.volume = None
        self.history_pose = []
        self.dis2targ = []

    def init_home_pose(self):
        if self.dataset == 'nyu':
            # (240, 180, 70)
            z = 35
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
            home_pose = home_pose * (np.array([self.predefined_bbx]) / np.array([[240, 180, 70]]))
            root_idx = 13
            chains_idx = [[13], [12], [11], [10, 9, 8], [7, 6], [5, 4], [3, 2], [1, 0]]
        elif self.dataset == 'mrsa15':
            # (180, 120, 70)
            z = 35
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
            home_pose = home_pose * (np.array([self.predefined_bbx]) / np.array([[180, 120, 70]]))
            root_idx = 0
            chains_idx = [[0], [17, 18, 19, 20], [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        elif self.dataset == 'icvl':
            # (140, 120, 60) (80, 60)
            z = 30
            home_pose = np.array([
                [90, 80, z],  # 1
                [110, 70, z],  # 2-4
                [115, 40, z],
                [110, 10, z],
                [70, 40, z],  # 5-7
                [50, 10, z],
                [40, 5, z],
                [50, 50, z],  # 8-10
                [20, 35, z],
                [10, 30, z],
                [50, 70, z],  # 11-13
                [20, 60, z],
                [10, 55, z],
                [50, 95, z],  # 14-16
                [30, 90, z],
                [20, 85, z]
            ], dtype=np.float32)
            home_pose = home_pose * (np.array([self.predefined_bbx]) / np.array([[140, 120, 60]]))
            root_idx = 0
            chains_idx = [[0], [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
        else:
            raise ValueError('Unknown dataset %s' % self.dataset)
        return home_pose, chains_idx, root_idx

    def reset(self, example):
        """
        load a new example in the environment
        """
        # example: (filename, xyz_pose, depth_img, pose_bbx, cropped_points, coeff,
        #           normalized_rotate_pose, normalized_rotate_points, rotated_bbx, volume)
        self.filename = example[0]
        self.orig_pose = example[1]  # xyz_pose
        self.depth_img = example[2]  # depth_img
        self.orig_bbx = example[3]  # pose_bbx
        self.rotate_mat = example[5]  # coeff
        self.norm_target_pose = example[6]
        self.norm_points = example[7]
        self.norm_bbx = example[8]
        self.volume = example[9]
        self.pose = copy.copy(self.home_pose)
        self.history_pose.clear()
        self.history_pose.append(self.pose)
        # resize ratio
        x_min, x_max, y_min, y_max, z_min, z_max = self.norm_bbx
        predefined_bbx = np.asarray([self.predefined_bbx])
        point1 = np.array([[x_min, y_min, z_min]])
        point2 = np.array([[x_max, y_max, z_max]])
        self.resize_ratio = predefined_bbx / (point2 - point1)
        # current iteration
        self.current_iter = 0
        # initial average distance self.norm_pose
        self.dis2targ.clear()
        self.dis2targ.append(np.linalg.norm((self.norm_target_pose - self.pose) / self.resize_ratio, axis=1).mean())

    def get_obs(self):
        # W, H, D (x, y, z)
        pose_volume = np.zeros_like(self.volume, dtype=np.int8)
        for xyz_joint in self.pose:
            joint_coor = tuple(xyz_joint.astype(np.int32))
            pose_volume[joint_coor] = 2
        obs = np.stack([self.volume, pose_volume], axis=-1)
        return obs.astype(np.int8), self.pose

    def adjust_pose(self, ac_flat):
        ac = np.reshape(ac_flat, [-1, 6])
        # adjust each joint
        pose = np.concatenate([self.pose, np.ones([self.num_joint, 1])], axis=1)
        for idx, chain in enumerate(self.chains_idx):
            for i, curr_joint_idx in enumerate(chain):
                joint_ac = ac[curr_joint_idx, :]
                se_mat = utils.lietomatrix(joint_ac[0:3], joint_ac[3:6])

                root_coor = copy.copy(pose[curr_joint_idx, :])
                root_coor[3] = 0
                if curr_joint_idx == self.root_idx:
                    # root joint
                    tmp = pose - root_coor
                    pose = np.matmul(se_mat, tmp.T).T + root_coor
                else:
                    # chain joints
                    subseq_joints_idx = chain[i:]
                    tmp = pose[subseq_joints_idx, :] - root_coor
                    pose[subseq_joints_idx, :] = np.matmul(se_mat, tmp.T).T + root_coor

        # clip pose by predefined_bbx
        final_pose = np.clip(pose[:, 0:3], np.array([[0, 0, 0]]), np.array([[self.predefined_bbx[0] - 1,
                                                                             self.predefined_bbx[1] - 1,
                                                                             self.predefined_bbx[2] - 1]]))
        return final_pose

    def step(self, ac):
        is_done = False
        self.current_iter += 1
        if self.current_iter > self.max_iters:
            is_done = True

        # ac has the shape [N_joints, 6]
        assert ac.shape[0] == self.num_joint * 6
        self.pose = self.adjust_pose(ac)
        self.history_pose.append(self.pose)

        # define rewards
        joints_error = np.linalg.norm((self.norm_target_pose - self.pose) / self.resize_ratio, axis=1)

        # mean joint error
        self.dis2targ.append(joints_error.mean())
        # r = self.dis2targ[-1] - self.dis2targ[-2]

        # mean average precision
        r = self.mean_avg_p(joints_error)
        return r, is_done

    @staticmethod
    def mean_avg_p(joints_error, threshold=80):
        thresholds = np.arange(1, threshold + 1)[np.newaxis, :]
        errors = np.repeat(joints_error[:, np.newaxis], threshold, axis=1)
        mean_ap = np.mean(errors <= thresholds) - 0.5
        return mean_ap

    def pose_to_lie_algebras(self, pose):
        """
        inverse kinematics to lie algebras and remove redundant lie algebras
        """
        lie_algebras = utils.inverse_kinematics(pose, self.chains_idx)
        label_lie_algebras = []
        for idx, chain in enumerate(self.chains_idx):
            if idx == 0:
                label_lie_algebras.append(lie_algebras[chain, 3:6].reshape([-1]))
            else:
                label_lie_algebras.append(lie_algebras[chain, 0:4].reshape([-1]))
        label = np.hstack(label_lie_algebras)
        return label

    def lie_algebras_to_pose(self, pred_lie_algebras, num_joints):
        """
        arrange lie algebras and forward kinematics to pose
        """
        lie_algebras = np.zeros([num_joints, 6])
        i = 0
        for idx, chain in enumerate(self.chains_idx):
            if idx == 0:
                lie_algebras[chain, 3:6] = pred_lie_algebras[0:3]
                i += 3
            else:
                n = 4 * len(chain)
                lie_algebras[chain, 0:4] = pred_lie_algebras[i:i+n].reshape([-1, 4])
                i += n
        pose = utils.forward_kinematics(lie_algebras, self.chains_idx)
        return pose


def in_test():
    # from data_preprocessing.mrsa_dataset import MRSADataset
    # reader = MRSADataset(test_fold='P0', subset='pre-processing', num_cpu=4, num_imgs_per_file=600)
    # reader.load_annotation()
    # example = reader.convert_to_example(reader._annotations[10])
    # env = HandEnv(dataset='mrsa15', subset='training', max_iters=10, predefined_bbx=reader.predefined_bbx)
    # env.reset(example)

    from data_preprocessing.nyu_dataset import NYUDataset
    reader = NYUDataset(subset='pps-testing', num_cpu=4, num_imgs_per_file=600)
    reader.load_annotation()
    example = reader.convert_to_example(reader._annotations[10])
    env = HandEnv(dataset='nyu', subset='training', max_iters=10, predefined_bbx=reader.predefined_bbx)
    env.reset(example)

    # from data_preprocessing.icvl_dataset import ICVLDataset
    # reader = ICVLDataset(subset='pps-testing',  num_cpu=4, num_imgs_per_file=600)
    # reader.load_annotation()
    # example = reader.convert_to_example(reader._annotations[150])
    # env = HandEnv(dataset='icvl', subset='training', max_iters=10, predefined_bbx=reader.predefined_bbx)
    # env.reset(example)

    # obs, pose = env.get_obs()
    # reader.plot_skeleton(env.norm_points, env.home_pose)
    #
    # reader.plot_skeleton(env.norm_points, env.norm_target_pose)
    #
    # reader.plot_skeleton(None, env.pose)
    #
    # ac = np.zeros([env.num_joint, 6])
    # ac[13, :] = np.array([0, 0, 0, 10, 0, 0])
    # ac[10, :] = np.array([0, 0, np.pi / 2, 10, 0, 0])
    # ac[9, :] = np.array([0, 0, np.pi / 2, 10, 0, 0])
    # ac[8, :] = np.array([0, 0, np.pi / 2, 10, 0, 0])
    # r = env.step(ac)
    # print(r)
    # reader.plot_skeleton(None, env.pose)

    print(env.home_pose)
    a = env.pose_to_lie_algebras(env.home_pose)
    b = env.lie_algebras_to_pose(a, reader.jnt_num)
    print(b)

    pass


if __name__ == '__main__':
    in_test()
