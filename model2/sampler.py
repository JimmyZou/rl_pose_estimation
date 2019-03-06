import numpy as np
from collections import deque
import random
import pickle
import tensorflow as tf


class ReplayBuffer(object):
    def __init__(self, buffer_size=100000):
        self.num_experiences = 0
        self.buffer = deque(maxlen=buffer_size)
        self.save_path = None

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        # (obs_volume, obs_pose, acs, rs, next_obs_pose, gammas)
        batch = random.sample(self.buffer, batch_size)
        obs_volume, obs_pose, acs, rs, next_obs_pose, gammas = zip(*batch)
        # convert xyz pose to volume pose
        obs, obs_next = [], []
        for i in range(batch_size):
            # TODO: multi-processors
            # current pose volume
            volume = obs_volume[i]
            _, z_max, y_max, x_max = volume.shape  # NDHW
            pose_volume = np.zeros_like(volume, dtype=np.int8)
            next_pose_volume = np.zeros_like(volume, dtype=np.int8)

            xyz_pose = obs_pose[i]  # [1, num_joint, 3]
            next_xyz_pose = next_obs_pose[i]
            for j in range(xyz_pose.shape[1]):
                tmp = xyz_pose[0, j, :].astype(np.int32)
                # NDHW
                joint_coor = (0, min(tmp[2], z_max - 1), min(tmp[1], y_max - 1), min(tmp[0], x_max - 1))
                pose_volume[joint_coor] = 2

                tmp = next_xyz_pose[0, j, :].astype(np.int32)
                # NDHW
                joint_coor = (0, min(tmp[2], z_max - 1), min(tmp[1], y_max - 1), min(tmp[0], x_max - 1))
                next_pose_volume[joint_coor] = 2
            # NDHWC
            obs.append(np.stack([volume, pose_volume], axis=-1))
            obs_next.append(np.stack([volume, next_pose_volume], axis=-1))
        # concatenate to array
        tmp = [np.concatenate(item, axis=0) for item in [obs, acs, rs, obs_next, gammas]]
        return tmp

    def add(self, sample):
        # sample = [(obs_volume, obs_pose, acs, rs, next_obs_pose, gammas), ...]
        self.buffer += sample

    def count(self):
        return len(self.buffer)

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

    def save_as_file(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.buffer, f)
            print('[ReplayBuffer] File is saved at %s.' % self.save_path)

    def load_from_file(self):
        with open(self.save_path, 'rb') as f:
            samples = pickle.load(f)
            print('[ReplayBuffer] File is loaded from %s.' % self.save_path)
            self.add(samples)


class Sampler(object):
    def __init__(self, actor, critic, env, dataset, gamma=0.9):
        self.actor = actor
        self.critic = critic
        self.env = env
        self.dataset = dataset
        self.predefine_bbx = self.env.predefine_bbx
        self.gamma = gamma
        self.avg_r = 0
        self.n_rs = 0

    def collect_one_episode(self, example):
        obs_volume = []
        obs_pose = []
        acs = []
        rs = []
        gammas = []
        is_done = False
        self.env.reset(example)
        while not is_done:
            state, pose = self.env.get_obs()
            # transpose from WHDC to DHWC
            state = np.expand_dims(np.transpose(state, [2, 1, 0, 3]), axis=0)
            ac_flat = self.actor.get_action(state)
            r, is_done = self.env.step(ac_flat[0, :])
            obs_volume.append(state[..., 0])  # only volume of hand points are saved in obs_volume
            obs_pose.append(np.expand_dims(pose, axis=0))  # pose: [1, num_joint, 3]
            acs.append(ac_flat)
            rs.append(np.array([[r]]))
            gammas.append(np.array([[self.gamma]]))
            # average history rewards
            self.n_rs += 1
            self.avg_r = self.avg_r + (r - self.avg_r) / self.n_rs
        # next obs
        next_obs_pose = obs_pose[1:]
        next_obs_pose.append(obs_pose[-1])
        gammas[-1] = np.array([[0]])
        # save to buffer
        samples = list(zip(obs_volume, obs_pose, acs, rs, next_obs_pose, gammas))
        return samples, self.env.dis2targ[-1]

    def collect_multiple_samples(self, num_files=4, num_samples=1000):
        mul_samples = []
        mul_final_dis = []
        examples = random.sample(self.dataset.get_batch_samples_training(num_files), num_samples)
        for example in examples:
            samples, final_dis = self.collect_one_episode(example)
            mul_samples += samples
            mul_final_dis.append(final_dis)
            # try:
            #     samples, final_dis = self.collect_one_episode(example)
            #     mul_samples += samples
            #     mul_final_dis.append(final_dis)
            # except:
            #     print('[Warning] file %s' % example[0])
        print('avg_rewards({} samples): {:.4f}'.format(self.n_rs, self.avg_r))
        print('final avg distance: {:.4f} ({:.4f})'.format(np.mean(mul_final_dis), np.std(mul_final_dis)))
        return mul_samples, np.mean(mul_final_dis), self.avg_r

    def test_one_episode(self, example):
        # example: (filename, xyz_pose, depth_img, pose_bbx, cropped_points, coeff,
        #           normalized_rotate_pose, normalized_rotate_points, rotated_bbx, volume)
        is_done = False
        self.env.reset(example)
        while not is_done:
            state, pose = self.env.get_obs()
            # transpose from WHDC to DHWC
            state = np.expand_dims(np.transpose(state, [2, 1, 0, 3]), axis=0)
            ac_flat = self.actor.get_action(state)
            r, is_done = self.env.step(ac_flat[0, :])
        result = (self.env.dis2targ[-1], self.env.pose, example[0], example[1], example[2],
                  example[3], example[5], example[6], example[8])
        return result

    def test_multiple_samples(self):
        results = []
        for example in self.dataset.get_samples_testing():
            # list of tuples: (final_avg_distance, final_pose, filename, xyz_pose, depth_img, pose_bbx,
            #                  coeff, normalized_rotate_pose, rotated_bbx)
            results.append(self.test_one_episode(example))
        return results
