import numpy as np
from collections import namedtuple
from datetime import datetime
import time, os, sys, pickle, cv2, glob


class BaseDataset(object):
    """
    super class for several hand pose dataset, including NYU, MRSA15, ICVL.
    """
    def __init__(self, subset):
        """
        subset: e.g., train, validation, test
        """
        self.subset = subset
        # fx, fy: focal length, cx, cy: center of the camera, w, h: width and height of images
        self.camera_cfg = namedtuple('CameraConfig', 'fx, fy, cx, cy, w, h')
        # filename is the filename of an image, pose is its corresponding annotations
        self.annotation = namedtuple('annotation', 'filename, pose')

    def load_annotation(self):
        raise NotImplementedError

    @property
    def annotations(self):
        """
        return annotations
        """
        raise NotImplementedError

    def convert_to_example(self, label):
        """
        load the image corresponding to the label.filename
        """
        raise NotImplementedError

    def preprocessing(self, example):
        """
        preprocess an example (filename, depth_img, pose)
        """
        raise NotImplemented

    # def crop_from_xyz_pose(self, depth_img, pose, camera_cfg):

