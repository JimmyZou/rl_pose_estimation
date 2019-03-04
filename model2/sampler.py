import numpy as np
from collections import deque
import random
import pickle


class ReplayBuffer(object):
    def __init__(self, buffer_size=100000):
        self.num_experiences = 0
        self.buffer = deque(maxlen=buffer_size)

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        # (obs_volume, obs_pose, acs, rs, next_obs_pose, gammas)
        batch = random.sample(self.buffer, batch_size)
        obs_volume, obs_pose, acs, rs, next_obs_pose, gammas = zip(*batch)
        # convert xyz pose to volume pose
        obs, obs_next = [], []
        for i in range(batch_size):
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

    def save_as_file(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)
            print('[ReplayBuffer] File is saved at %s.' % path)


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
        gammas[-1] = 0
        # save to buffer
        samples = list(zip(obs_volume, obs_pose, acs, rs, next_obs_pose, gammas))
        return samples

    def collect_multiple_samples(self, num_files=2):
        mul_samples = []
        examples = self.dataset.get_batch_samples_training(num_files)
        for example in examples[0:2]:
            try:
                mul_samples += self.collect_one_episode(example)
                print('avg_rewards({} samples):{:.4f}'.format(self.n_rs, self.avg_r))
            except:
                print('[Warning] file %s' % example[0])
        return mul_samples
