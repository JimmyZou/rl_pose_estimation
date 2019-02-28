import numpy as np
from model1.ac_model import ActorRoot, CriticRoot
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
    def __init__(self, actor, critic, env, dataset, root_buffer, chain_buffer, gamma=0.9):
        self.actor = actor
        self.critic = critic
        self.env = env
        self.dataset = dataset
        self.gamma = gamma
        self.root_buffer = root_buffer
        self.chain_buffer = chain_buffer

    def collect_one_episode_root(self, example):
        obs = []
        acs = []
        rs = []
        gammas = []
        self.env.reset(example)
        is_root = True
        while is_root:
            local_obs, _, _, is_root = self.env.get_obs()
            local_obs = np.expand_dims(np.expand_dims(local_obs, axis=0), axis=-1)
            ac = self.actor.get_action(local_obs)
            r = self.env.step(ac)
            obs.append(local_obs)
            acs.append(acs)
            rs.append(np.array([[r]]))
            gammas.append(np.array([[self.gamma]]))
        # next obs
        next_obs = obs[1:]
        # the terminal
        next_obs.append(np.zeros_like(obs[0]))
        gammas[-1] = 0

        # save to buffer
        samples = list(zip(obs, acs, rs, next_obs, gammas))
        self.root_buffer.add(samples)

    def collect_multiple_samples_root(self, num_files=2):
        assert isinstance(self.actor, ActorRoot)
        assert isinstance(self.critic, CriticRoot)
        assert isinstance(self.root_buffer, ReplayBuffer)

        examples = self.dataset.get_batch_samples_training(num_files)
        for example in range(examples):
            self.collect_one_episode_root(example)

    def collect_one_episode_chain(self, example):
        raise NotImplemented

    def collect_multiple_samples_chain(self, num_files=2):
        raise NotImplemented



def in_test():
    # TODO: sampler, normalize reward
    pass


if __name__ == '__main__':
    in_test()
