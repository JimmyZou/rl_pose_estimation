import sys
sys.path.append('..')
from data_preprocessing.dataset_base import BaseDataset
import os
import time
import numpy as np
import pickle
import utils
import cv2
import glob
import matplotlib.pyplot as plt


class ICVLDataset(BaseDataset):
    def __init__(self, subset, predefined_bbx=(63, 63, 31), num_cpu=4,
                 num_imgs_per_file=600, root_dir='/hand_pose_data/icvl/'):
        super(ICVLDataset, self).__init__(subset, num_imgs_per_file, num_cpu)

        # self.camera_cfg is a tuple (fx, fy, cx, cy, w, h)
        self.camera_cfg = (241.42, 241.42, 160, 120, 320, 240)
        self.max_depth = 500
        self.root_dir = root_dir
        self.num_imgs_per_file = num_imgs_per_file
        self.dataset = 'ICVL'
        self.pre_depth_img = None
        self.predefined_bbx = predefined_bbx

        if self.subset in ['pps-training']:
            self.src_dir = os.path.join(self.root_dir, 'dataset/train/')
            assert os.path.exists(self.src_dir)
            self.img_dir = os.path.join(self.src_dir, 'Depth')
            self.store_dir = os.path.join(self.src_dir, 'train_data_ppsd/')
        elif self.subset in ['pps-testing']:
            self.src_dir = os.path.join(self.root_dir, 'dataset/test/')
            assert os.path.exists(self.src_dir)
            self.img_dir = os.path.join(self.src_dir, 'Depth')
            self.store_dir = os.path.join(self.src_dir, 'test_data_ppsd/')
        elif self.subset in ['training']:
            self.train_files = glob.glob(self.root_dir + 'dataset/train/train_data_ppsd/*')
            self.test_files = glob.glob(self.root_dir + 'dataset/test/test_data_ppsd/*')
            print('[%sDataset] %i training files and %i testing files in total.'
                  % (self.dataset, len(self.train_files), len(self.test_files)))
        else:
            raise ValueError('Unknown subset %s to %s hand datset' % (subset, self.dataset))

        self.jnt_num = 16
        self.pose_dim = 3 * self.jnt_num
        print('[%sDataset] %d joints, with %d dim' % (self.dataset, self.jnt_num, self.pose_dim))

    def load_annotation(self):
        time_begin = time.time()
        if os.path.exists(self.src_dir + 'labels.pkl'):
            with open(self.src_dir + 'labels.pkl', 'rb') as f:
                self._annotations = pickle.load(f)
                print('[data.%sDataset] annotation has been loaded with %d samples, %fs' %
                      (self.dataset, len(self._annotations), time.time() - time_begin))
        else:
            print('[data.%sDataset] annotations are being loaded from text file...' % self.dataset)
            # load joint.txt in each fold.
            with open(self.src_dir + 'labels.txt', 'r') as f:
                for frm_idx, line in enumerate(f):
                    if self.subset in ['pps-training'] and not line.startswith('2014'):
                        continue
                    if line == '\n' or '201406030937/image_-001.png' in line:
                        continue
                    buf = line.strip(' \n').split(' ')
                    filename = buf[0]
                    if not len(buf[1:]) == self.pose_dim:
                        print('[error] %s annotations error...' % filename)
                    tmp = np.asarray([float(x) for x in buf[1:]]).reshape([-1, 3])
                    pose = np.reshape(utils.uvd2xyz(tmp, self.camera_cfg), [-1, 3])
                    self._annotations.append((filename, pose))
            with open(self.src_dir + 'labels.pkl', 'wb') as f:
                pickle.dump(self._annotations, f)
            print('[data.%sDataset] annotation has been loaded with %d samples, %fs' %
                  (self.dataset, len(self._annotations), time.time() - time_begin))

    def convert_to_example(self, label):
        """
        convert one example (image and pose) to target format
        """
        filename, pose = label
        img_dir = os.path.join(self.img_dir, filename)
        depth_img = cv2.imread(img_dir, -1)  # BGR order
        if depth_img is None:
            print('[warning] %s is None' % filename)
        if (depth_img < self.max_depth).sum() < 20:
            depth_img = self.pre_depth_img.copy()
            print('[warning] points %s is empty' % filename)
        else:
            self.pre_depth_img = depth_img.copy()

        # show depth image
        # jnt_uvd = utils.xyz2uvd(pose.reshape([-1, 3]), self.camera_cfg)
        # utils.plot_annotated_depth_img(depth_img, jnt_uvd)

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
        plt.plot([pose[15, 0], pose[14, 0]], [pose[15, 1], pose[14, 1]], color='red', linewidth=lw)
        plt.plot([pose[14, 0], pose[13, 0]], [pose[14, 1], pose[13, 1]], color='red', linewidth=lw)
        plt.plot([pose[13, 0], pose[0, 0]], [pose[13, 1], pose[0, 1]], color='red', linewidth=lw)
        # ring finger
        plt.plot([pose[12, 0], pose[11, 0]], [pose[12, 1], pose[11, 1]], color='orangered', linewidth=lw)
        plt.plot([pose[11, 0], pose[10, 0]], [pose[11, 1], pose[10, 1]], color='orangered', linewidth=lw)
        plt.plot([pose[10, 0], pose[0, 0]], [pose[10, 1], pose[0, 1]], color='orangered', linewidth=lw)
        # middle finger
        plt.plot([pose[9, 0], pose[8, 0]], [pose[9, 1], pose[8, 1]], color='orange', linewidth=lw)
        plt.plot([pose[8, 0], pose[7, 0]], [pose[8, 1], pose[7, 1]], color='orange', linewidth=lw)
        plt.plot([pose[7, 0], pose[0, 0]], [pose[7, 1], pose[0, 1]], color='orange', linewidth=lw)
        # fore finger
        plt.plot([pose[6, 0], pose[5, 0]], [pose[6, 1], pose[5, 1]], color='yellow', linewidth=lw)
        plt.plot([pose[5, 0], pose[4, 0]], [pose[5, 1], pose[4, 1]], color='yellow', linewidth=lw)
        plt.plot([pose[4, 0], pose[0, 0]], [pose[4, 1], pose[0, 1]], color='yellow', linewidth=lw)
        # thumb
        plt.plot([pose[3, 0], pose[2, 0]], [pose[3, 1], pose[2, 1]], color='cyan', linewidth=lw)
        plt.plot([pose[2, 0], pose[1, 0]], [pose[2, 1], pose[1, 1]], color='cyan', linewidth=lw)
        plt.plot([pose[1, 0], pose[0, 0]], [pose[1, 1], pose[0, 1]], color='cyan', linewidth=lw)
        plt.show()


def in_test():
    reader = ICVLDataset(subset='pps-training', num_cpu=10, num_imgs_per_file=600, predefined_bbx=(63, 63, 31))
    # reader = ICVLDataset(subset='pps-testing', num_cpu=3, num_imgs_per_file=600, predefined_bbx=(63, 63, 31))
    reader.load_annotation()
    # for i in range(20):
    #     gap = 101
    #     print(reader._annotations[i * gap][0])
    #     example = reader.convert_to_example(reader._annotations[i * gap])
    #     print(example[-1].shape)

    for ann in reader._annotations:
        if '201406191044/image_7438.png' in ann[0] or '201406191044/image_6936.png' in ann[0]:
            example = reader.convert_to_example(ann)
            print(example[9].shape)
            # try:
            #     example = reader.convert_to_example(ann)
            # except:
            #     print('error...%s', ann[0])

    # reader.store_multi_processors(reader.store_dir)

    # a = reader.get_batch_samples_training(3)
    # for data in reader.get_samples_testing():
    #     print(len(data))


if __name__ == '__main__':
    in_test()

