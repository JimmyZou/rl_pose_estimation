from data_preprocessing.dataset_base import BaseDataset
import os
import time
import cv2
import numpy as np
from scipy.io import loadmat
import utils


class NYUDataset(BaseDataset):
    def __init__(self, subset, num_cpu=4, num_imgs_per_file=700, root_dir="../../data/nyu/"):
        super(NYUDataset, self).__init__(subset, num_imgs_per_file, num_cpu)

        self.camera_cfg = (588.235, 587.084, 320, 240, 640, 480)
        self.max_depth = 1500.0
        self.root_dir = root_dir
        self.num_imgs_per_file = num_imgs_per_file
        self.dataset = 'NYU'

        if self.subset in ['training', 'validation', 'training_small']:
            self.src_dir = os.path.join(self.root_dir, 'dataset/train/')
            assert os.path.exists(self.src_dir)
            self.img_dir = self.src_dir
            self.store_dir = os.path.join(self.root_dir, 'train_data_ppsd/')
        elif self.subset in ['testing']:
            self.src_dir = os.path.join(self.root_dir, 'dataset/test/')
            assert os.path.exists(self.src_dir)
            self.img_dir = self.src_dir
            self.store_dir = os.path.join(self.root_dir, 'test_data_ppsd/')
        else:
            raise ValueError('Unknown subset %s to NYU hand datset' % subset)

        if not os.path.exists(self.store_dir):
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

    def crop_from_xyz_pose(self, filename, depth_img, pose, pad=20):
        # bounding box
        xyz_pose = np.reshape(pose, [-1, 3])
        x_min, x_max = np.min(xyz_pose[:, 0]) - pad, np.max(xyz_pose[:, 0]) + pad
        y_min, y_max = np.min(xyz_pose[:, 1]) - pad, np.max(xyz_pose[:, 1]) + pad
        z_min, z_max = np.min(xyz_pose[:, 2]) - pad, np.max(xyz_pose[:, 2]) + pad
        bbx = (x_min, x_max, y_min, y_max, z_min, z_max)

        # crop image
        depth_uvd = utils.depth2uvd(depth_img)
        depth_xyz = utils.uvd2xyz(depth_uvd, self.camera_cfg)
        depth_xyz = depth_xyz[(depth_xyz[:, 2] < self.max_depth) & (depth_xyz[:, 2] > 0), :]
        crop_idxes = (x_min < depth_xyz[:, 0]) & (depth_xyz[:, 0] < x_max) & \
                     (y_min < depth_xyz[:, 1]) & (depth_xyz[:, 1] < y_max) & \
                     (z_min < depth_xyz[:, 2]) & (depth_xyz[:, 2] < z_max)
        cropped_points = depth_xyz[crop_idxes, :]
        example = (filename, xyz_pose, depth_img, bbx, cropped_points)

        # example: tuple (filename, xyz_pose, depth_img, bbx, cropped_points)
        return example

    def convert_to_example(self, label):
        """
        convert one example (image and pose) to target format
        return:
            a tuple: (filename, xyz_pose, depth_img, bbox, cropped_points)
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
        utils.plot_cropped_3d_annotated_hand(example[1], example[3], example[4])

        # preprocessed_example (filename, xyz_pose, depth_img, pose_bbx,
        # coeff, normalized_rotate_pose, normalized_rotate_points, rotated_bbx)
        preprocessed_example = self.consistent_orientation(example)
        utils.plot_cropped_3d_annotated_hand(preprocessed_example[5], None, preprocessed_example[6])

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.scatter(preprocessed_example[6][:, 0], preprocessed_example[6][:, 1],
        #             color='b', marker='.', s=2, alpha=0.5)
        # plt.scatter(preprocessed_example[5][:, 0], preprocessed_example[5][:, 1], color='r', marker='.', s=10)
        # plt.show()

        return preprocessed_example


def in_test():
    reader = NYUDataset(subset='testing', num_cpu=4, num_imgs_per_file=7)
    reader.load_annotation()
    for i in range(3):
        print(reader._annotations[i * 1030][0])
        reader.convert_to_example(reader._annotations[i * 1030])
    # reader.store_preprocessed_data_per_file(reader._annotations[0:5], 1, reader.store_dir)
    # reader.store_multi_processors(reader.store_dir)


if __name__ == '__main__':
    in_test()
