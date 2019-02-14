from data_preprocessing.dataset_base import BaseDataset
import os
import time
import numpy as np
import pickle
import utils
import cv2


class ICVLDataset(BaseDataset):
    def __init__(self, subset, num_cpu=4, num_imgs_per_file=700, root_dir="../../data/icvl/"):
        super(ICVLDataset, self).__init__(subset, num_imgs_per_file, num_cpu)

        # self.camera_cfg is a tuple (fx, fy, cx, cy, w, h)
        self.camera_cfg = (241.42, 241.42, 160, 120, 320, 240)
        self.max_depth = 500
        self.root_dir = root_dir
        self.num_imgs_per_file = num_imgs_per_file
        self.dataset = 'ICVL'

        if self.subset in ['training', 'validation', 'training_small']:
            self.src_dir = os.path.join(self.root_dir, 'dataset/train/')
            assert os.path.exists(self.src_dir)
            self.img_dir = os.path.join(self.src_dir, 'Depth')
            self.store_dir = os.path.join(self.src_dir, 'train_data_ppsd/')
        elif self.subset in ['testing']:
            self.src_dir = os.path.join(self.root_dir, 'dataset/test/')
            assert os.path.exists(self.src_dir)
            self.img_dir = os.path.join(self.src_dir, 'Depth')
            self.store_dir = os.path.join(self.src_dir, 'test_data_ppsd/')
        else:
            raise ValueError('Unknown subset %s to NYU hand datset' % subset)

        if not os.path.exists(self.store_dir):
            os.makedirs(self.store_dir)
            print('File %s is created to save preprocessed data.' % self.store_dir)

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
                    # if self.subset in ['training', 'validation', 'training_small'] and not line.startswith('2014'):
                    #     continue
                    if line == '\n':
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

        # example: tuple (filename, xyz_pose, depth_img, bbx, cropped_points)
        example = (filename, xyz_pose, depth_img, bbx, cropped_points)
        return example

    def convert_to_example(self, label):
        """
        convert one example (image and pose) to target format
        return:
            a tuple: (filename, xyz_pose, depth_img, bbox, cropped_points, mrsa_shape)
        """
        filename, pose = label
        img_dir = os.path.join(self.img_dir, filename)
        depth_img = cv2.imread(img_dir, -1)  # BGR order

        # show depth image
        # jnt_uvd = utils.xyz2uvd(pose.reshape([-1, 3]), self.camera_cfg)
        # utils.plot_annotated_depth_img(depth_img, jnt_uvd)

        # tuple (filename, xyz_pose, depth_img, bbox, cropped_points)
        example = self.crop_from_xyz_pose(filename, depth_img, pose)
        utils.plot_cropped_3d_annotated_hand(example[1], example[3], example[4])

        # preprocessed_example (filename, xyz_pose, depth_img, pose_bbx,
        # coeff, normalized_rotate_pose, normalized_rotate_points, rotated_bbx)
        preprocessed_example = self.consistent_orientation(example)
        utils.plot_cropped_3d_annotated_hand(preprocessed_example[5], None, preprocessed_example[6])

        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(preprocessed_example[6][:, 0], preprocessed_example[6][:, 1],
                    color='b', marker='.', s=2, alpha=0.5)
        plt.scatter(preprocessed_example[5][:, 0], preprocessed_example[5][:, 1], color='r', marker='.', s=10)
        plt.show()

        return preprocessed_example


def in_test():
    reader = ICVLDataset(subset='training', num_cpu=4, num_imgs_per_file=600)
    reader.load_annotation()
    for i in range(5):
        print(reader._annotations[i * 210][0])
        reader.convert_to_example(reader._annotations[i * 100])
    # reader.store_preprocessed_data_per_file(reader.annotations()[0:5], 1, reader.store_dir)
    # reader.store_multi_processors(reader.store_dir)


if __name__ == '__main__':
    in_test()

