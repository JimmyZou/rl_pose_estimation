import numpy as np
import utils
import copy
import multiprocessing


class HandEnvOld(object):
    def __init__(self, dataset, subset, max_iters, predefined_bbx, pretrained_model):
        self.pretrained_model = pretrained_model
        self.dataset = dataset
        assert subset in ["training", "testing"]
        self.subset = subset  # ["training", "testing"]
        self.predefined_bbx = (predefined_bbx[0] + 1, predefined_bbx[1] + 1, predefined_bbx[2] + 1)
        self.initial_pose, self.chains_idx, self.root_idx = self.init_home_pose()
        # the first element is the iterations for root joint, second for chain joints
        self.num_chains = len(self.chains_idx)
        self.num_joint = self.initial_pose.shape[0]
        self.max_iters = max_iters
        self.current_iter = 0
        self.resize_ratio = np.array([[1., 1., 1.]])

        # data of an example
        self.pose = copy.copy(self.initial_pose)
        self.filename = None
        self.orig_pose = None
        self.orig_bbx = None
        self.rotate_mat = None
        self.norm_target_pose = None
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
        self.orig_bbx = example[3]  # pose_bbx
        self.rotate_mat = example[5]  # coeff
        self.norm_target_pose = example[6]
        self.norm_bbx = example[8]
        self.volume = example[9]
        self.pose = copy.copy(self.initial_pose)
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
        # TODO: adjust pose

        # clip pose by predefined_bbx
        # final_pose = np.clip(pose[:, 0:3], np.array([[0, 0, 0]]), np.array([[self.predefined_bbx[0] - 1,
        #                                                                      self.predefined_bbx[1] - 1,
        #                                                                      self.predefined_bbx[2] - 1]]))
        # return final_pose
        return 0

    def step(self, ac):
        is_done = False
        self.current_iter += 1
        if self.current_iter > self.max_iters:
            is_done = True

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


class HandEnv(object):
    def __init__(self, dataset_name, subset, max_iters, predefined_bbx, pretrained_model,
                 reward_range=2, num_cpus=10):
        self.pretrained_model = pretrained_model
        self.num_cpus = num_cpus
        self.reward_range = reward_range
        self.dataset = dataset_name
        assert subset in ["training", "testing"]
        self.subset = subset  # ["training", "testing"]
        self.predefined_bbx = predefined_bbx
        self.chains_idx, self.root_idx, self.num_joints = self.dataset_info()
        # the first element is the iterations for root joint, second for chain joints
        self.num_chains = len(self.chains_idx)
        self.max_iters = max_iters
        self.current_iter = 0
        self.current_pose = None  # array [N, num_joint, 3]

        # data of mini-batch examples
        self.batch_size = None  # int
        self.filename = None  # list
        self.xyz_pose = None  # array [N, num_joint, 3]
        self.orig_bbx = None  # array [N, 6]
        self.coeff = None  # array [N, 3, 3]
        self.rotated_target_pose = None  # array [N, num_joint, 3]
        self.rotated_bbx = None  # array [N, 6]
        self.volume = None  # array NDHWC
        self.init_pose = None  # array [N, num_joint, 3]
        self.resize_ratio = None  # array [N, 3]
        self.history_pose = []
        self.lie_groups = {}  # dict of lie groups for batch-size samples
        self.init_max_error = {}  # dict of scalar
        self.root_coordinates = None

    def dataset_info(self):
        if self.dataset == 'nyu':
            root_idx = 13
            chains_idx = [[13], [12], [11], [10, 9, 8], [7, 6], [5, 4], [3, 2], [1, 0]]
            num_joint = 14
        elif self.dataset == 'mrsa15':
            root_idx = 0
            chains_idx = [[0], [17, 18, 19, 20], [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
            num_joint = 21
        elif self.dataset == 'icvl':
            root_idx = 0
            chains_idx = [[0], [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
            num_joint = 16
        else:
            raise ValueError('Unknown dataset %s' % self.dataset)
        return chains_idx, root_idx, num_joint

    def reset(self, examples, sess):
        """
        load a mini-batch of examples (list of tuples)
        example: (filename, xyz_pose, depth_img, pose_bbx, cropped_points, coeff,
                  normalized_rotate_pose, normalized_rotate_points, rotated_bbx, volume)
        sess: get initial pose using pretrained model
        """
        # reset environment setting
        self.current_iter = 0
        self.history_pose.clear()
        self.lie_groups.clear()
        self.init_max_error.clear()

        # initial batch examples
        self.batch_size = len(examples)
        self.filename, xyz_pose, _, orig_bbx, _, coeff, \
            rotated_target_pose, _, rotated_bbx, volume = zip(*examples)

        # convert to arrays [batch_size, ...]
        self.xyz_pose = np.stack(xyz_pose, axis=0)
        self.orig_bbx = np.stack(orig_bbx, axis=0)
        self.coeff = np.stack(coeff, axis=0)
        self.rotated_target_pose = np.stack(rotated_target_pose, axis=0)
        self.rotated_bbx = np.stack(rotated_bbx, axis=0)
        self.volume = np.transpose(np.expand_dims(np.stack(volume, axis=0), axis=4), [0, 3, 2, 1, 4])  # NDHWC
        self.init_pose = sess.run(self.pretrained_model.ac,  # [N, ac_dims]
                                  feed_dict={self.pretrained_model.obs: self.volume,
                                             self.pretrained_model.dropout_prob: 1.0}).reshape([self.batch_size, -1, 3])
        self.root_coordinates = self.init_pose[:, self.root_idx, :]
        self.history_pose.append(self.init_pose)
        self.current_pose = self.init_pose.copy()

        # # debug
        # results = []
        # for i in range(self.batch_size):
        #     results.append(self.arrange_initial_batch_examples(i, self.init_pose[i, :], self.num_joints,
        #                                                        self.chains_idx, self.rotated_bbx[i, :],
        #                                                        self.predefined_bbx,
        #                                                        self.rotated_target_pose[i, :, :]))
        # self.resize_ratio = np.zeros([self.batch_size, 3])  # [N, 3]
        # for result in results:
        #     idx, resize_ratio, lie_group, max_error = result
        #     self.resize_ratio[idx, :] = resize_ratio
        #     self.lie_groups[idx] = lie_group
        #     self.init_max_error[idx] = max_error
        # print('Loaded %i samples...' % self.batch_size)
        # # debug

        results = []
        pool = multiprocessing.Pool(self.num_cpus)
        for i in range(self.batch_size):
            results.append(pool.apply_async(self.arrange_initial_batch_examples,
                                            (i, self.init_pose[i, :], self.num_joints, self.chains_idx,
                                             self.rotated_bbx[i, :], self.predefined_bbx,
                                             self.rotated_target_pose[i, :, :])))
        pool.close()
        pool.join()
        self.resize_ratio = np.zeros([self.batch_size, 3])  # [N, 3]
        for result in results:
            idx, resize_ratio, lie_group, max_error = result.get()
            self.resize_ratio[idx, :] = resize_ratio
            self.lie_groups[idx] = lie_group
            self.init_max_error[idx] = max_error
        # print('Loaded %i samples...' % self.batch_size)

    def get_obs(self):
        """
        inputs:
            volume: [N, D, H, W, C=1]
            pose: [N, n_joints, 3]
        returns:
            obs: [N, D, H, W, C=2]
        """
        # convert pose coordinate to volume
        results = []
        pool = multiprocessing.Pool(self.num_cpus)
        for i in range(self.batch_size):
            results.append(pool.apply_async(self.get_pose_volume,
                                            (i, self.current_pose[i, :, :], self.predefined_bbx)))
        pool.close()
        pool.join()
        pose_volume_list, i = [], 0
        for result in results:
            idx, pose_volume = result.get()
            assert i == idx
            i += 1
            pose_volume_list.append(pose_volume)
        pose_volume = np.stack(pose_volume_list, axis=0)  # [NDHWC]
        obs = np.concatenate([self.volume, pose_volume], axis=4)
        return obs, self.current_pose

    def step(self, acs):
        """
        inputs:
            acs: [N, ac_dims]
        returns:
            r, is_done
        """
        is_done = False
        self.current_iter += 1
        if self.current_iter >= self.max_iters:
            is_done = True

        # # debug
        # results = []
        # for i in range(self.batch_size):
        #     results.append(self.iterative_pose_adjust(i, self.lie_groups[i], acs[i, :], self.num_joints,
        #                                               self.chains_idx, self.root_coordinates[i, :],
        #                                               self.rotated_target_pose[i, :, :], self.init_max_error[i]))
        # rs = np.zeros([self.batch_size, 1])
        # for result in results:
        #     batch_idx, new_lie_group, new_pose, reward = result
        #     assert len(new_lie_group) == (len(self.chains_idx)-1)
        #     self.lie_groups[batch_idx] = new_lie_group
        #     self.current_pose[batch_idx, :, :] = new_pose
        #     rs[batch_idx, 0] = reward
        # self.history_pose.append(self.current_pose)
        # # debug

        # adjust pose and obatin reward (multi-processing)
        results = []
        pool = multiprocessing.Pool(self.num_cpus)
        for i in range(self.batch_size):
            results.append(pool.apply_async(self.iterative_pose_adjust,
                                            (i, self.lie_groups[i], acs[i, :], self.num_joints, self.chains_idx,
                                             self.root_coordinates[i, :], self.rotated_target_pose[i, :, :],
                                             self.init_max_error[i])))
        pool.close()
        pool.join()
        rs = np.zeros([self.batch_size, 1])
        for result in results:
            batch_idx, new_lie_group, new_pose, reward = result.get()
            assert len(new_lie_group) == (len(self.chains_idx)-1)
            self.lie_groups[batch_idx] = new_lie_group
            self.current_pose[batch_idx, :, :] = new_pose
            rs[batch_idx, 0] = reward
        self.history_pose.append(self.current_pose)
        # clip rewards
        rs = np.clip(rs, -self.reward_range, self.reward_range)
        return rs, is_done

    def transfer_pose(self):
        raise NotImplemented

    def batch_max_error(self):
        # array [N, num_joint, 3]
        max_error = np.max(np.linalg.norm(self.current_pose - self.rotated_target_pose, axis=2), axis=1)
        return max_error

    @staticmethod
    def iterative_pose_adjust(batch_idx, prior_lie_group, ac, num_joints, chains_idx,
                              root_coordinate, target_pose, init_error):
        """
        inputs:
            prior_lie_group: dict of lie groups for each chain
            ac: lie algebras 4*(num_joints-1)
        return:
            new_lie_group
            new_pose
            reward
        """
        lie_algebras = np.zeros([num_joints, 6])
        # convert action to lie algebras
        i = 0
        for idx, chain in enumerate(chains_idx):
            if idx is not 0:
                n = 4 * len(chain)
                lie_algebras[chain, 0:4] = ac[i:i + n].reshape([-1, 4])
                i += n

        # forward kinematic
        new_pose = np.zeros([num_joints, 3])
        root_idx = chains_idx[0][0]
        new_pose[root_idx] = root_coordinate
        new_lie_groups = []
        for idx, chain in enumerate(chains_idx):
            if idx is not 0:
                # append root joint idx
                chain_idx = [root_idx] + chain
                joint_coordinates, lie_group = utils.update_lie_groups(prior_lie_group[idx-1],
                                                                       lie_algebras[chain_idx, :])
                new_pose[chain_idx[1:], :] = joint_coordinates[1:]
                new_lie_groups.append(lie_group)

        # reward
        max_error = np.max(np.linalg.norm(target_pose - new_pose, axis=1))
        reward = init_error - max_error
        return batch_idx, new_lie_groups, new_pose, reward


    @staticmethod
    def arrange_initial_batch_examples(_idx, pose, num_joints, chains_idx, norm_bbx, predefined_bbx, target_pose):
        # resize ratio
        x_min, x_max, y_min, y_max, z_min, z_max = norm_bbx
        predefined_bbx = np.asarray([predefined_bbx])
        point1 = np.array([[x_min, y_min, z_min]])
        point2 = np.array([[x_max, y_max, z_max]])
        _resize_ratio = predefined_bbx / (point2 - point1)

        # lie group of init pose
        assert pose.shape == (num_joints, 3)
        _lie_algebras = utils.inverse_kinematics(pose, chains_idx)
        joint_xyz_f, _lie_group = utils.forward_kinematics(_lie_algebras, chains_idx)
        assert len(_lie_group) == (len(chains_idx) - 1)

        # max error in current distance measurement
        max_error = np.max(np.linalg.norm(target_pose - pose, axis=1))
        return _idx, _resize_ratio, _lie_group, max_error

    @staticmethod
    def get_pose_volume(idx, xyz_pose, bbx):
        x_max, y_max, z_max = bbx
        # WHD
        pose_volume = np.zeros([x_max + 1, y_max + 1, z_max + 1], dtype=np.int8)
        xyz_pose = np.clip(xyz_pose, [0, 0, 0], [x_max, y_max, z_max])
        for j in range(xyz_pose.shape[0]):
            tmp = xyz_pose[j, :].astype(np.int32)
            joint_coor = (tmp[0], tmp[1], tmp[2])
            pose_volume[joint_coor] = 4
        # DHWC
        pose_volume = np.expand_dims(np.transpose(pose_volume, [2, 1, 0]), axis=3)
        return idx, pose_volume

    # @staticmethod
    # def mean_avg_p(joints_error, threshold=80):
    #     thresholds = np.arange(1, threshold + 1)[np.newaxis, :]
    #     errors = np.repeat(joints_error[:, np.newaxis], threshold, axis=1)
    #     mean_ap = np.mean(errors <= thresholds) - 0.5
    #     return mean_ap


def in_test():
    pass


if __name__ == '__main__':
    in_test()
