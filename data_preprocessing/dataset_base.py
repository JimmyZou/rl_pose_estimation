import time
import pickle
import multiprocessing
from sklearn.decomposition import PCA
import numpy as np
import utils


class BaseDataset(object):
    """
    super class for several hand pose dataset, including NYU, MRSA15, ICVL.
    """
    def __init__(self, subset, num_imgs_per_file, num_cpus):
        """
        subset: e.g., train, validation, test
        """
        self.subset = subset
        # self.camera_cfg is a tuple (fx, fy, cx, cy, w, h)
        # fx, fy: focal length, cx, cy: center of the camera, w, h: width and height of images
        self.camera_cfg = ()
        # self._annotations is a list of tuples (filename, annotation)
        # filename is the filename of an image, pose is its corresponding annotations
        self._annotations = []
        self.num_imgs_per_file = num_imgs_per_file
        self.num_cpus = num_cpus
        self.dataset = 'Empty'
        self.pca = PCA()
        self.max_depth = None
        self.test_files = []
        self.train_files = []

    def load_annotation(self):
        raise NotImplementedError

    @property
    def get_annotations(self):
        """
        return annotations
        """
        return self._annotations

    def convert_to_example(self, label):
        """
        load the image corresponding to the label.filename
        """
        raise NotImplementedError

    def crop_from_xyz_pose(self, filename, depth_img, pose, pad=20):
        """
        crop a depth image according to pose
        """
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

        xyz_pose -= np.array([[x_min, y_min, z_min]])
        cropped_points -= np.array([[x_min, y_min, z_min]])
        # example: tuple (filename, xyz_pose, depth_img, bbx, cropped_points)
        example = (filename, xyz_pose, depth_img, bbx, cropped_points)
        return example

    @staticmethod
    def convert_to_volume(points, bbx):
        x_min, x_max, y_min, y_max, z_min, z_max = bbx
        volume = np.zeros([int(x_max - x_min) + 1, int(y_max - y_min) + 1, int(z_max - z_min) + 1])
        # print(bbx, volume.shape)
        for point in points:
            idx = tuple(point.astype(np.int16))
            volume[idx] += 1
        return volume.astype(np.int8)

    def consistent_orientation(self, example):
        filename, xyz_pose, depth_img, pose_bbx, cropped_points = example

        self.pca.fit(cropped_points)
        coeff = self.pca.components_.T
        if coeff[1, 0] < 0:
            coeff[:, 0] = - coeff[:, 0]
        if coeff[2, 2] < 0:
            coeff[:, 2] = - coeff[:, 2]
        coeff[:, 1] = np.cross(coeff[:, 0], coeff[:, 2])
        rotated_points = np.dot(cropped_points, coeff)
        rotated_pose = np.dot(xyz_pose, coeff)
        x_min, x_max = np.min(rotated_points[:, 0]), np.max(rotated_points[:, 0])
        y_min, y_max = np.min(rotated_points[:, 1]), np.max(rotated_points[:, 1])
        z_min, z_max = np.min(rotated_points[:, 2]), np.max(rotated_points[:, 2])
        rotated_bbx = (x_min, x_max, y_min, y_max, z_min, z_max)
        normalized_rotate_points = rotated_points - np.array([[x_min, y_min, z_min]])
        normalized_rotate_pose = rotated_pose - np.array([[x_min, y_min, z_min]])
        volume = self.convert_to_volume(normalized_rotate_points, rotated_bbx)
        example = (filename, xyz_pose, depth_img, pose_bbx, cropped_points, coeff,
                   normalized_rotate_pose, normalized_rotate_points, rotated_bbx, volume)
        return example

    def store_preprocessed_data_per_file(self, annotations, stored_file_idx, store_dir):
        """
        preprocess 'self.num_imgs_per_file' images and save in one file
        """
        stored_data = []
        for label in annotations:
            stored_data.append(self.convert_to_example(label))
        with open(store_dir + str(stored_file_idx) + '.pkl', 'wb') as f:
            pickle.dump(stored_data, f)
            print('[data.%sDataset] File %s is saved.' % (self.dataset, store_dir + str(stored_file_idx) + '.pkl'))

    def store_multi_processors(self, store_dir):
        print('[data.%sDataset] multi-processing starts...' % self.dataset)
        time_begin = time.time()
        N = len(self._annotations)
        if self.subset == 'testing':
            tag = 'test'
        else:
            tag = 'train'
        num_files = N // self.num_imgs_per_file + 1
        file_idxes = [(j * self.num_imgs_per_file, min((j + 1) * self.num_imgs_per_file, N)) for j in range(num_files)]
        print('[data.%sDataset] %i files in total...' % (self.dataset, num_files))

        results = []
        pool = multiprocessing.Pool(self.num_cpus)
        for i in range(num_files):
            results.append(pool.apply_async(self.store_preprocessed_data_per_file,
                                            (self._annotations[file_idxes[i][0]: file_idxes[i][1]],
                                             '%s-%i-%i' % (tag, i, file_idxes[i][1] - file_idxes[i][0]), store_dir,)))
        pool.close()
        pool.join()
        pool.terminate()

        for result in results:
            tmp = result.get()
            if tmp is not None:
                print(tmp)
        print('[data.%sDataset] multi-processing ends, %fs' % (self.dataset, time.time() - time_begin))

    def get_batch_samples_training(self, num_files):
        assert self.subset in ['training']
        m = len(self.train_files)
        file_idx = np.random.randint(0, m, num_files)
        data = []
        for i in file_idx:
            file = self.train_files[i]
            with open(file, 'rb') as f:
                data += pickle.load(f)
        print('File {} ({:4} samples) is loaded for training.'.format(file_idx, len(data)))
        return data

    def get_samples_testing(self):
        assert self.subset in ['training']
        for i, file in enumerate(self.test_files):
            print('File %s (%i of %i) is loaded for testing.' % (file, i, len(self.test_files)))
            with open(file, 'rb') as f:
                data = pickle.load(f)
            yield data

