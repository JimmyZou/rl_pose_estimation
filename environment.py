import numpy as np


class HandEnv(object):
    def __init__(self, dataset, subset, iter_per_joint=1):
        self.dataset = dataset
        assert subset in ["training", "testing"]
        self.subset = subset  # ["training", "testing"]
        self.home_pose, self.chains_idx = self.init_home_pose()
        # data of an example
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
            chains_idx = [[11, 12, 13], [13, 10, 9, 8], [13, 7, 6], [13, 5, 4], [13, 3, 2], [13, 1, 0]]
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
            chains_idx = [[0, 17, 18, 19, 20], [0, 1, 2, 3, 4], [0, 5, 6, 7, 8],
                          [0, 9, 10, 11, 12], [0, 13, 14, 15, 16]]
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
            chains_idx = [[0, 1, 2, 3, 4], [0, 4, 5, 6], [0, 7, 8, 9], [0, 10, 11, 12], [0, 13, 14, 15]]
        else:
            raise ValueError('Unknown subset %s' % self.subset)
        return home_pose, chains_idx

    def reset(self, example):
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

    def step(self):
        raise NotImplemented

    def get_obs(self):
        raise NotImplemented


def in_test():
    pass


if __name__ == '__main__':
    in_test()

