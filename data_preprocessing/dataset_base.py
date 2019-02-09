import numpy as np
from collections import namedtuple
from datetime import datetime
import time, os, sys, pickle, cv2, glob, multiprocessing


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

    def preprocess(self):
        """
        preprocess an namedtuple example to targeted format
        """
        raise NotImplemented

    def store_preprocessed_data_per_file(self, annotations, stored_file_idx, store_dir):
        """
        preprocess 'self.num_imgs_per_file' images and save in one file
        """
        stored_data = []
        for label in annotations:
            # TODO: preprocess a sample, namedtuple('sample', 'filename, xyz_pose, depth_img, bbox, cropped_points')
            stored_data.append(self.convert_to_example(label))
        with open(store_dir + str(stored_file_idx) + '.pkl', 'wb') as f:
            pickle.dump(stored_data, f)
            print('[data.%sDataset] File %s is saved.' % (self.dataset, store_dir + str(stored_file_idx) + '.pkl'))

    def store_multi_processors(self, store_dir):
        print('[data.%sDataset] multi-processors start...' % self.dataset)
        time_begin = time.time()
        N = len(self._annotations)
        num_files = N // self.num_imgs_per_file + 1
        file_idxes = [(j * self.num_imgs_per_file, min((j + 1) * self.num_imgs_per_file, N)) for j in range(num_files)]

        results = []
        pool = multiprocessing.Pool(self.num_cpus)
        for i in range(num_files):
            results.append(pool.apply_async(self.store_preprocessed_data_per_file,
                                     (self._annotations[file_idxes[i][0]: file_idxes[i][1]], i, store_dir, )))
        pool.close()
        pool.join()
        print('[data.%sDataset] multi-processing ends, %fs' % (self.dataset, time.time() - time_begin))


