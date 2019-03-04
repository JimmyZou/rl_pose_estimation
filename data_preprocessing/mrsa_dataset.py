from data_preprocessing.dataset_base import BaseDataset
import os
import time
import numpy as np
import pickle
import struct
import multiprocessing
import glob
import matplotlib.pyplot as plt


class MRSADataset(BaseDataset):
    def __init__(self, subset, test_fold, predefined_bbx=(180, 120, 70), num_cpu=4,
                 num_imgs_per_file=600, root_dir='/hand_pose_data/mrsa15/'):
        super(MRSADataset, self).__init__(subset, num_imgs_per_file, num_cpu)

        # self.camera_cfg is a tuple (fx, fy, cx, cy, w, h)
        self.camera_cfg = (241.42, 241.42, 160, 120, 320, 240)
        self.max_depth = 1000
        self.root_dir = root_dir
        self.num_imgs_per_file = num_imgs_per_file
        self.predefined_bbx = predefined_bbx
        self.dataset = 'MRSA15'
        self.fold_list = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
        self.pose_list = '1 2 3 4 5 6 7 8 9 I IP L MP RP T TIP Y'.split(' ')
        self.test_fold = test_fold  # for cross validation
        assert self.test_fold in self.fold_list

        if self.subset in ['pre-processing']:
            self.src_dir = os.path.join(self.root_dir, 'dataset/')
            assert os.path.exists(self.src_dir)
            self.img_dir = self.src_dir
            self.store_dir = os.path.join(self.root_dir, 'data_ppsd/')
            if not os.path.exists(self.store_dir):
                os.makedirs(self.store_dir)
                print('File %s is created to save preprocessed data.' % self.store_dir)
        elif self.subset in ['training']:
            self.src_dir = os.path.join(self.root_dir, 'data_ppsd/')
            assert os.path.exists(self.src_dir)
            all_files = glob.glob(self.src_dir + 'P*')
            self.train_files = [file for file in all_files if ('%s-' % self.test_fold) not in file]
            self.test_files = [file for file in all_files if ('%s-' % self.test_fold) in file]
        else:
            raise ValueError('Unknown subset %s to %s hand datset' % (subset, self.dataset))

        self.jnt_num = 21
        self.pose_dim = 3 * self.jnt_num
        print('[%sDataset] %d joints, with %d dim' % (self.dataset, self.jnt_num, self.pose_dim))
        self.pre_depth_image = None
        self.pre_crop_points = None

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
        if (depth_img > 0).sum() < 1000:
            print('[warning] %s is empty' % filename)
            depth_img = self.pre_depth_image.copy()
        else:
            self.pre_depth_image = depth_img.copy()
        return depth_img, cropped_depth_img, bbx

    def convert_to_example(self, label):
        """
        convert one example (image and pose) to target format
        """
        filename, pose = label
        depth_img, cropped_depth_img, mrsa_shape = self._bin2depth(filename)

        # # show depth image
        # import utils
        # jnt_uvd = utils.xyz2uvd(pose.reshape([-1, 3]), self.camera_cfg)
        # utils.plot_annotated_depth_img(depth_img, jnt_uvd)

        # tuple (filename, xyz_pose, depth_img, bbox, cropped_points)
        example = self.crop_from_xyz_pose(filename, depth_img, pose)
        if example[4].shape[0] == 0:
            print('[warning] points %s is empty' % filename)
            example[4] = self.pre_crop_points.copy()
        else:
            self.pre_crop_points = example[4].copy()
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
        plt.plot([pose[16, 0], pose[15, 0]], [pose[16, 1], pose[15, 1]], color='red', linewidth=lw)
        plt.plot([pose[15, 0], pose[14, 0]], [pose[15, 1], pose[14, 1]], color='red', linewidth=lw)
        plt.plot([pose[14, 0], pose[13, 0]], [pose[14, 1], pose[13, 1]], color='red', linewidth=lw)
        plt.plot([pose[13, 0], pose[0, 0]], [pose[13, 1], pose[0, 1]], color='red', linewidth=lw)
        # ring finger
        plt.plot([pose[12, 0], pose[11, 0]], [pose[12, 1], pose[11, 1]], color='orangered', linewidth=lw)
        plt.plot([pose[11, 0], pose[10, 0]], [pose[11, 1], pose[10, 1]], color='orangered', linewidth=lw)
        plt.plot([pose[10, 0], pose[9, 0]], [pose[10, 1], pose[9, 1]], color='orangered', linewidth=lw)
        plt.plot([pose[9, 0], pose[0, 0]], [pose[9, 1], pose[0, 1]], color='orangered', linewidth=lw)
        # middle finger
        plt.plot([pose[8, 0], pose[7, 0]], [pose[8, 1], pose[7, 1]], color='orange', linewidth=lw)
        plt.plot([pose[7, 0], pose[6, 0]], [pose[7, 1], pose[6, 1]], color='orange', linewidth=lw)
        plt.plot([pose[6, 0], pose[5, 0]], [pose[6, 1], pose[5, 1]], color='orange', linewidth=lw)
        plt.plot([pose[5, 0], pose[0, 0]], [pose[5, 1], pose[0, 1]], color='orange', linewidth=lw)
        # fore finger
        plt.plot([pose[4, 0], pose[3, 0]], [pose[4, 1], pose[3, 1]], color='yellow', linewidth=lw)
        plt.plot([pose[3, 0], pose[2, 0]], [pose[3, 1], pose[2, 1]], color='yellow', linewidth=lw)
        plt.plot([pose[2, 0], pose[1, 0]], [pose[2, 1], pose[1, 1]], color='yellow', linewidth=lw)
        plt.plot([pose[1, 0], pose[0, 0]], [pose[1, 1], pose[0, 1]], color='yellow', linewidth=lw)
        # thumb
        plt.plot([pose[20, 0], pose[19, 0]], [pose[20, 1], pose[19, 1]], color='cyan', linewidth=lw)
        plt.plot([pose[19, 0], pose[18, 0]], [pose[19, 1], pose[18, 1]], color='cyan', linewidth=lw)
        plt.plot([pose[18, 0], pose[17, 0]], [pose[18, 1], pose[17, 1]], color='cyan', linewidth=lw)
        plt.plot([pose[17, 0], pose[0, 0]], [pose[17, 1], pose[0, 1]], color='cyan', linewidth=lw)
        plt.show()

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
    reader = MRSADataset(subset='pre-processing', test_fold='P0', num_cpu=30, num_imgs_per_file=600)
    reader.load_annotation()
    # for i in range(10):
    #     gap = 501
    #     print(reader._annotations[i * gap][0])
    #     example = reader.convert_to_example(reader._annotations[i * gap])
    #     print(example[-1].shape)

    # for ann in reader._annotations:
    #     if 'P3/3/000289_depth' in ann[0] or 'P3/3/000288_depth' in ann[0]:
    #         example = reader.convert_to_example(ann)
    #         # try:
    #         #     example = reader.convert_to_example(ann)
    #         # except:
    #         #     print('error...%s', ann[0])

    # reader.store_preprocessed_data_per_file(reader._annotations[0:5], 1, reader.store_dir)
    reader.store_multi_processors(reader.store_dir)

    # a = reader.get_batch_samples_training(3)
    # for data in reader.get_samples_testing():
    #     print(len(data))


if __name__ == '__main__':
    in_test()



