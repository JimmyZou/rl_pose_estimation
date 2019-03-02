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
    def __init__(self, actor_root, critic_root, actor_chain, critic_chain,
                 env, dataset, root_obs_width=(40, 40, 20), chain_obs_width=(30, 30, 20), gamma=0.9):
        self.actor_root = actor_root
        self.critic_root = critic_root
        self.root_obs_width = root_obs_width
        self.actor_chain = actor_chain
        self.critic_chain = critic_chain
        self.chain_obs_width = chain_obs_width
        self.env = env
        self.dataset = dataset
        self.gamma = gamma
        self.avg_r = 0
        self.n_rs = 0

    def collect_one_episode_root(self, example):
        obs = []
        acs = []
        rs = []
        gammas = []
        self.env.reset(example)
        is_root = True
        while is_root:
            local_obs, _, _, is_root = self.env.get_obs(self.root_obs_width)
            # transpose and expand local_obs to NDHWC
            local_obs = np.expand_dims(np.expand_dims(np.transpose(local_obs, [2, 0, 1]), axis=0), axis=-1)
            ac = self.actor_root.get_action(local_obs)
            r = self.env.step(ac)
            obs.append(local_obs)
            acs.append(acs)
            rs.append(np.array([[r]]))
            gammas.append(np.array([[self.gamma]]))
            # average history rewards
            self.n_rs += 1
            self.avg_r = self.avg_r + (r - self.avg_r) / self.n_rs
        # next obs
        next_obs = obs[1:]
        # the terminal
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

    def collect_one_episode_chain(self, example):
        samples = []
        obs = []
        acs = []
        rs = []
        gammas = []
        self.env.reset(example)
        obs_width = self.root_obs_width
        all_done = False
        while not all_done:
            local_obs, all_done, chain_done, is_root = self.env.get_obs(obs_width)
            local_obs = np.expand_dims(np.expand_dims(np.transpose(local_obs, [2, 0, 1]), axis=0), axis=-1)
            if is_root:
                ac = self.actor_root.get_action(local_obs)
                r = self.env.step(ac)
                if chain_done:
                    obs_width = self.chain_obs_width
            else:
                ac = self.actor_chain.get_action(local_obs)
                r = self.env.step(ac)
                obs.append(local_obs)
                acs.append(acs)
                rs.append(np.array([[r]]))
                gammas.append(np.array([[self.gamma]]))
                # update average history reward
                self.n_rs += 1
                self.avg_r = self.avg_r + (r - self.avg_r) / self.n_rs
                if chain_done:
                    # next obs
                    next_obs = obs[1:]
                    # the terminal
                    next_obs.append(np.zeros_like(obs[0]))
                    gammas[-1] = 0
                    # save to buffer
                    samples = list(zip(obs, acs, rs, next_obs, gammas))
                    obs = []
                    acs = []
                    rs = []
                    gammas = []
        return samples

    def collect_multiple_samples_chain(self, num_files=1):
        mul_samples = []
        examples = self.dataset.get_batch_samples_training(num_files)
        for example in examples:
            mul_samples += self.collect_one_episode_chain(example)
            print('avg_rewards({} samples):{:.4f}'.format(self.n_rs, self.avg_r))
        return mul_samples


def in_test():
    # multi-processing
    pass


if __name__ == '__main__':
    in_test()
