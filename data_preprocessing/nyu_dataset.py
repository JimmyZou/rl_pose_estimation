import sys
sys.path.append('..')
from data_preprocessing.dataset_base import BaseDataset
import os
import time
import cv2
import numpy as np
from scipy.io import loadmat
import glob
import matplotlib.pyplot as plt


class NYUDataset(BaseDataset):
    def __init__(self, subset, predefined_bbx=(240, 180, 70), num_cpu=4,
                 num_imgs_per_file=600, root_dir="/home/data/nyu/"):
        super(NYUDataset, self).__init__(subset, num_imgs_per_file, num_cpu)

        self.camera_cfg = (588.235, 587.084, 320, 240, 640, 480)
        self.max_depth = 1500.0
        self.root_dir = root_dir
        self.num_imgs_per_file = num_imgs_per_file
        self.dataset = 'NYU'
        self.predefined_bbx = predefined_bbx

        if self.subset in ['pps-training']:
            self.src_dir = os.path.join(self.root_dir, 'dataset/train/')
            assert os.path.exists(self.src_dir)
            self.img_dir = self.src_dir
            self.store_dir = os.path.join(self.root_dir, 'train_data_ppsd/')
        elif self.subset in ['pps-testing']:
            self.src_dir = os.path.join(self.root_dir, 'dataset/test/')
            assert os.path.exists(self.src_dir)
            self.img_dir = self.src_dir
            self.store_dir = os.path.join(self.root_dir, 'test_data_ppsd/')
        elif self.subset in ['training']:
            self.train_files = glob.glob(self.root_dir + 'train_data_ppsd/')
            self.test_files = glob.glob(self.root_dir + 'test_data_ppsd/')
        else:
            raise ValueError('Unknown subset %s to %s hand datset' % (subset, self.dataset))

        if os.path.exists(self.store_dir):
            # for training or testing after pre-processing
            self.file_list = glob.glob(self.store_dir)
            print('[NyuDataset] %d %s files are loaded from %s' % (len(self.file_list), self.subset, self.store_dir))
        else:
            os.makedirs(self.store_dir)
            print('File %s is created to save preprocessed data.' % self.store_dir)

        # select a part of 14 joints
        self.joint_truncate_idxes = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]
        self.jnt_num = len(self.joint_truncate_idxes)
        self.pose_dim = 3 * self.jnt_num
        print('[NyuDataset] %d joints, with %d dim' % (self.jnt_num, self.pose_dim))

    def load_annotation(self):
        time_begin = time.time()
        # load joint_data.mat in either train or test fold.
        _dir = os.path.join(self.src_dir, 'joint_data.mat')
        joint_data = loadmat(_dir)
        # print('%i joints names:' % self.jnt_num, joint_data['joint_names'])
        camera_num = 1 if self.subset == 'testing' else 3
        joints = [joint_data['joint_xyz'][idx] for idx in range(camera_num)]
        filenames = [
            ['depth_{}_{:07d}.png'.format(camera_idx + 1, idx + 1) for idx in range(joints[camera_idx].shape[0])]
            for camera_idx in range(camera_num)]

        for c_j, c_file in zip(joints, filenames):
            for j, n in zip(c_j, c_file):
                j = j[self.joint_truncate_idxes]
                j = j.reshape((-1, 3))
                j[:, 1] *= -1.0
                j = j.reshape((-1,))
                self._annotations.append((n, j))
        print('[data.%sDataset] annotation has been loaded with %d samples, %fs' %
              (self.dataset, len(self._annotations), time.time() - time_begin))

    def _decode_png(self, img_data):
        # The top 8 bits of depth are packed into green and the lower 8 bits into blue.
        b, g = img_data[:, :, 0].astype(np.uint16), img_data[:, :, 1].astype(np.uint16)
        depth_img = (g * 256 + b).astype(np.float32)
        # utils.plot_depth_img(depth_img, None, self.camera_cfg, self.max_depth)
        return depth_img

    def convert_to_example(self, label):
        """
        convert one example (image and pose) to target format
        """
        filename, pose = label
        img_dir = os.path.join(self.img_dir, filename)
        img_data = cv2.imread(img_dir, -1)  # BGR order
        depth_img = self._decode_png(img_data)

        # show joints on uvd depth image
        # jnt_uvd = utils.xyz2uvd(label.pose, self.camera_cfg)
        # utils.plot_annotated_depth_img(depth_img, jnt_uvd, self.camera_cfg, self.max_depth)

        # tuple (filename, xyz_pose, depth_img, bbox, cropped_points)
        example = self.crop_from_xyz_pose(filename, depth_img, pose)
        # utils.plot_cropped_3d_annotated_hand(example[1], example[3], example[4])

        # preprocessed_example (filename, xyz_pose, depth_img, pose_bbx, cropped_point,
        # coeff, normalized_rotate_pose, normalized_rotate_points, rotated_bbx, volume)
        preprocessed_example = self.consistent_orientation(example, self.predefined_bbx)

        # import utils
        # utils.plot_cropped_3d_annotated_hand(preprocessed_example[6], None, preprocessed_example[7])
        # self.plot_skeleton(preprocessed_example[7], preprocessed_example[6])
        return preprocessed_example

    @staticmethod
    def plot_skeleton(points, pose):
        lw = 1.5
        plt.figure()
        if points is not None:
            plt.scatter(points[:, 0], points[:, 1], color='gray', marker='.', s=2, alpha=0.5)
        # plot pose skeleton
        plt.scatter(pose[:, 0], pose[:, 1], color='black', marker='h', s=30)
        # little finger
        plt.plot([pose[0, 0], pose[1, 0]], [pose[0, 1], pose[1, 1]], color='red', linewidth=lw)
        plt.plot([pose[1, 0], pose[13, 0]], [pose[1, 1], pose[13, 1]], color='red', linewidth=lw)
        # ring finger
        plt.plot([pose[2, 0], pose[3, 0]], [pose[2, 1], pose[3, 1]], color='orangered', linewidth=lw)
        plt.plot([pose[3, 0], pose[13, 0]], [pose[3, 1], pose[13, 1]], color='orangered', linewidth=lw)
        # middle finger
        plt.plot([pose[4, 0], pose[5, 0]], [pose[4, 1], pose[5, 1]], color='orange', linewidth=lw)
        plt.plot([pose[5, 0], pose[13, 0]], [pose[5, 1], pose[13, 1]], color='orange', linewidth=lw)
        # fore finger
        plt.plot([pose[6, 0], pose[7, 0]], [pose[6, 1], pose[7, 1]], color='yellow', linewidth=lw)
        plt.plot([pose[7, 0], pose[13, 0]], [pose[7, 1], pose[13, 1]], color='yellow', linewidth=lw)
        # thumb
        plt.plot([pose[8, 0], pose[9, 0]], [pose[8, 1], pose[9, 1]], color='cyan', linewidth=lw)
        plt.plot([pose[9, 0], pose[10, 0]], [pose[9, 1], pose[10, 1]], color='cyan', linewidth=lw)
        plt.plot([pose[10, 0], pose[13, 0]], [pose[10, 1], pose[13, 1]], color='cyan', linewidth=lw)
        # palm
        plt.plot([pose[13, 0], pose[12, 0]], [pose[13, 1], pose[12, 1]], color='purple', linewidth=lw)
        plt.plot([pose[13, 0], pose[11, 0]], [pose[13, 1], pose[11, 1]], color='purple', linewidth=lw)
        plt.show()


def in_test():
    reader = NYUDataset(subset='pps-training', num_cpu=15, num_imgs_per_file=600)
    # reader = NYUDataset(subset='pps-testing', num_cpu=30, num_imgs_per_file=600)
    reader.load_annotation()
    # for i in range(5):
    #     gap = 250
    #     print(reader._annotations[i * gap][0])
    #     example = reader.convert_to_example(reader._annotations[i * gap])
    #     print(example[-1].shape)

    # reader.store_preprocessed_data_per_file(reader._annotations[0:5], 1, reader.store_dir)
    reader.store_multi_processors(reader.store_dir)

    # a = reader.get_batch_samples_training(3)
    # for data in reader.get_samples_testing():
    #     print(len(data))


if __name__ == '__main__':
    in_test()
