from data_preprocessing.dataset_base import BaseDataset
import pickle, os, glob, time, cv2
import numpy as np
from collections import namedtuple
from scipy.io import loadmat
import utils
import matplotlib.pyplot as plt


class NYUDataset(BaseDataset):
    def __init__(self, subset, root_dir="../../data/nyu/"):
        super(NYUDataset, self).__init__(subset)

        # bbx denotes bounding box
        self._annotations = []
        self.camera_cfg = self.Camera(fx=588.235, fy=587.084, cx=320, cy=240, w=640, h=480)
        self.num_per_file = 730
        self.max_depth = 1500.0
        self.root_dir = root_dir

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

        self.jnt_num = 36
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
        filenames = [['depth_{}_{:07d}.png'.format(camera_idx+1, idx+1) for idx in range(joints[camera_idx].shape[0])]
                     for camera_idx in range(camera_num)]

        for c_j, c_file in zip(joints, filenames):
            for j, n in zip(c_j, c_file):
                j = j.reshape((-1, 3))
                j[:, 1] *= -1.0
                j = j.reshape((-1,))
                self._annotations.append(self.annotation(n, j))
        print('[data.NyuDataset] annotation has been loaded with %d samples, %fs' %
              (len(self._annotations), time.time() - time_begin))

    def _decode_png(self, img_data):
        # The top 8 bits of depth are packed into green and the lower 8 bits into blue.
        g, b = img_data[:, :, 1].astype(np.uint16), img_data[:, :, 2].astype(np.uint16)
        depth_img = (g * 256 + b).astype(np.float32)
        # utils.plot_depth_img(depth_img, None, self.camera_cfg, self.max_depth)
        return depth_img

    def crop_from_xyz_pose(self, example):
        # crop_from_xyz_pose
        pass

    def convert_to_example(self, label):
        """
        convert one example (image and pose) to target format
        """
        img_dir = os.path.join(self.img_dir, label.filename)
        img_data = cv2.imread(img_dir, -1)
        depth_img = self._decode_png(img_data)
        # example = self.crop_from_xyz_pose(label.name, depth_img, label.pose)
        jnt_uvd = utils.xyz2uvd(label.pose, self.camera_cfg)
        utils.plot_depth_img(depth_img, jnt_uvd, self.camera_cfg, self.max_depth)









def in_test():
    # reader = NyuDataset('training')
    # reader.write_TFRecord_multi_thread(num_threads=30, num_shards=300)
    reader = NYUDataset('testing')
    reader.load_annotation()
    reader.convert_to_example(reader._annotations[0])



if __name__ == '__main__':
    in_test()

