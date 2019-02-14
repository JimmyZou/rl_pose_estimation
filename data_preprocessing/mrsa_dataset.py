from data_preprocessing.dataset_base import BaseDataset
import os
import time
import numpy as np
import pickle
import utils
import struct
import multiprocessing
import matplotlib.pyplot as plt


class MRSADataset(BaseDataset):
    def __init__(self, subset, num_cpu=4, num_imgs_per_file=100, root_dir="../../data/msra15/"):
        super(MRSADataset, self).__init__(subset, num_imgs_per_file, num_cpu)

        # self.camera_cfg is a tuple (fx, fy, cx, cy, w, h)
        self.camera_cfg = (241.42, 241.42, 160, 120, 320, 240)
        self.max_depth = 1000
        self.root_dir = root_dir
        self.num_imgs_per_file = num_imgs_per_file
        self.dataset = 'MRSA15'
        # self.fold_list = ['P0', 'P8']  # ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
        self.fold_list = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
        self.pose_list = '1 2 3 4 5 6 7 8 9 I IP L MP RP T TIP Y'.split(' ')

        if self.subset in ['pre-processing']:
            self.src_dir = os.path.join(self.root_dir, 'dataset/')
            assert os.path.exists(self.src_dir)
            self.img_dir = self.src_dir
            self.store_dir = os.path.join(self.root_dir, 'data_ppsd/')
            if not os.path.exists(self.store_dir):
                os.makedirs(self.store_dir)
                print('File %s is created to save preprocessed data.' % self.store_dir)
        elif self.subset in ['cross_validation']:
            # TODO: for nine-fold cross validation
            self.src_dir = os.path.join(self.root_dir, 'data_ppsd/')
            assert os.path.exists(self.src_dir)
        else:
            raise ValueError('Unknown subset %s to MRSA15 hand datset' % subset)

        self.jnt_num = 21
        self.pose_dim = 3 * self.jnt_num
        print('[%sDataset] %d joints, with %d dim' % (self.dataset, self.jnt_num, self.pose_dim))

    def load_annotation(self):
        time_begin = time.time()
        if os.path.exists(self.root_dir + 'labels.pkl'):
            with open(self.root_dir + 'labels.pkl', 'rb') as f:
                self._annotations = pickle.load(f)
                print('[data.%sDataset] annotation has been loaded with %d samples, %fs' %
                      (self.dataset, len(self._annotations), time.time() - time_begin))
        else:
            assert self.subset == 'pre-processing'
            print('[data.%sDataset] annotations are being loaded...' % self.dataset)
            # load joint.txt in each fold.
            for fold in self.fold_list:
                print('[data.%sDataset] annotations in %s' % (self.dataset, fold))
                for gesture in self.pose_list:
                    fold_dir = '%s%s/%s/' % (self.img_dir, fold, gesture)
                    with open(fold_dir + 'joint.txt', 'r') as f:
                        for frm_idx, line in enumerate(f):
                            if frm_idx == 0:
                                continue
                            buf = line.strip('\n').split(' ')
                            tmp = np.asarray([float(x) for x in buf])
                            pose = np.reshape(np.reshape(tmp, [-1, 3]) * np.array([[1.0, -1.0, -1.0]]), [-1])
                            filename = os.path.join(fold_dir, '%06i_depth.bin' % (frm_idx - 1))
                            self._annotations.append((filename, pose))
            with open(self.root_dir + 'labels.pkl', 'wb') as f:
                pickle.dump(self._annotations, f)
            print('[data.%sDataset] annotation has been loaded with %d samples, %fs' %
                  (self.dataset, len(self._annotations), time.time() - time_begin))

    def _bin2depth(self, filename):
        with open(filename, 'rb') as f:
            # bbx ('width', 'height', 'left', 'top', 'right', 'bottom')
            bbx = tuple([struct.unpack('i', f.read(4))[0] for _ in range(6)])
            cropped_depth_img = np.fromfile(f, dtype=np.float32)
        crop_height, crop_width = bbx[5] - bbx[3], bbx[4] - bbx[2]
        cropped_depth_img = np.reshape(cropped_depth_img, [crop_height, crop_width])

        # expand the cropped dm to full-size make later process in a uniformed way
        depth_img = np.zeros((bbx[1], bbx[0]), np.float32)
        np.copyto(depth_img[bbx[3]: bbx[5], bbx[2]: bbx[4]], cropped_depth_img)
        # for empty image, just copy the previous frame
        if depth_img.sum() < 10:
            print('[warning] %s is empty' % filename)
        return depth_img, cropped_depth_img, bbx

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
        depth_img, cropped_depth_img, mrsa_shape = self._bin2depth(filename)

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

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.scatter(preprocessed_example[6][:, 0], preprocessed_example[6][:, 1],
        #             color='b', marker='.', s=2, alpha=0.5)
        # plt.scatter(preprocessed_example[5][:, 0], preprocessed_example[5][:, 1], color='r', marker='.', s=10)
        # plt.show()

        return preprocessed_example

    def store_multi_processors(self, store_dir):
        print('[data.%sDataset] multi-processing starts...' % self.dataset)
        time_begin = time.time()

        # save self.num_imgs_per_file samples per file
        num_imgs_per_fold = [8499, 8492, 8412, 8488, 8500, 8497, 8497, 8498, 8492]
        # num_imgs_per_fold = [8499, 8492]
        assert self.num_imgs_per_file > 50
        num_files_per_fold = num_imgs_per_fold[0] // self.num_imgs_per_file + 1
        file_idxes = [0]
        filenames = []
        base_idx = 0
        for p, n in enumerate(num_imgs_per_fold):
            for i in range(num_files_per_fold):
                file_idxes.append(min((i + 1) * self.num_imgs_per_file, n) + base_idx)
                filenames.append('P%i-%i-%i' % (p, i, file_idxes[-1] - file_idxes[-2]))
            base_idx += n

        # # save according to gesture in each fold
        # file_idxes = [0]
        # filenames = []
        # for fold in self.fold_list:
        #     for gesture in self.pose_list:
        #         assert '%s/%s/' % (fold, gesture) in self._annotations[file_idxes[-1]][0]
        #         fold_dir = '%s%s/%s/' % (self.img_dir, fold, gesture)
        #         filenames.append('%s_%s' % (fold, gesture))
        #         n_files = len(glob.glob(fold_dir + '*_depth.bin'))
        #         file_idxes.append(file_idxes[-1] + n_files)

        results = []
        pool = multiprocessing.Pool(self.num_cpus)
        for i, filename in enumerate(filenames):
            results.append(pool.apply_async(self.store_preprocessed_data_per_file,
                             (self._annotations[file_idxes[i]: file_idxes[i + 1]], filename, store_dir, )))
        pool.close()
        pool.join()
        pool.terminate()

        for result in results:
            tmp = result.get()
            if tmp is not None:
                print(tmp)
        print('[data.%sDataset] multi-processing ends, %fs' % (self.dataset, time.time() - time_begin))


def in_test():
    reader = MRSADataset(subset='pre-processing', num_cpu=4, num_imgs_per_file=100)
    reader.load_annotation()
    for i in range(5):
        print(reader._annotations[i * 500 + 2510][0])
        reader.convert_to_example(reader._annotations[i * 500 + 10])
    # reader.store_preprocessed_data_per_file(reader._annotations[0:5], 1, reader.store_dir)
    # reader.store_multi_processors(reader.store_dir)


if __name__ == '__main__':
    in_test()

