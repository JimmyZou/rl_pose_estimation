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
        samples = random.sample(self.buffer, batch_size)
        # concatenate to array
        tmp = [np.concatenate(item, axis=0) for item in zip(*samples)]
        return tmp

    def add(self, sample):
        # sample = [(state, action, reward, new_state, gamma), ...]
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

    def collect_one_episode_root(self, example):
        obs = []
        acs = []
        rs = []
        gammas = []
        is_done = False
        self.env.reset(example)
        while not is_done:
            state = self.env.get_obs()
            # transpose from WHDC to DHWC
            local_obs = np.expand_dims(np.transpose(state, [2, 1, 0, 3]), axis=0)
            ac = self.actor.get_action(local_obs)
            r = self.env.step(ac)
            obs.append(local_obs)
            acs.append(acs)
            rs.append(np.array([[r]]))
            gammas.append(np.array([[self.gamma]]))
            # average history rewards
            self.n_rs += 1
            self.avg_r = self.avg_r + (r - self.avg_r) / self.n_rs
        # next obs
        # TODO: save memory
        next_obs = obs[1:]
        next_obs.append(np.zeros_like(obs[0]))
        gammas[-1] = 0
        # save to buffer
        samples = list(zip(obs, acs, rs, next_obs, gammas))
        return samples

    def collect_multiple_samples_root(self, num_files=2):
        mul_samples = []
        examples = self.dataset.get_batch_samples_training(num_files)
        for example in examples:
            mul_samples += self.collect_one_episode_root(example)
            print('avg_rewards({} samples):{:.4f}'.format(self.n_rs, self.avg_r))
        return mul_samples
