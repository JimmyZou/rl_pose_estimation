import time
import pickle
import multiprocessing
from sklearn.decomposition import PCA
import numpy as np


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

    def load_annotation(self):
        raise NotImplementedError

    @property
    def annotations(self):
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
        raise NotImplemented

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

        return (filename, xyz_pose, depth_img, pose_bbx,
                coeff, normalized_rotate_pose, normalized_rotate_points, rotated_bbx)

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

        results = []
        pool = multiprocessing.Pool(self.num_cpus)
        for i in range(num_files):
            results.append(pool.apply_async(self.store_preprocessed_data_per_file,
                                            (self._annotations[file_idxes[i][0]: file_idxes[i][1]],
                                             '%s-%i-%i' % (tag, i, self.num_imgs_per_file), store_dir,)))
        pool.close()
        pool.join()
        pool.terminate()

        for result in results:
            tmp = result.get()
            if tmp is not None:
                print(tmp)
        print('[data.%sDataset] multi-processing ends, %fs' % (self.dataset, time.time() - time_begin))


