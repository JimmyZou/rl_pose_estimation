import numpy as np
from collections import deque
import random
import pickle
import multiprocessing


class ReplayBuffer(object):
    def __init__(self, buffer_size=100000, save_path=None):
        self.num_experiences = 0
        self.buffer = deque(maxlen=buffer_size)
        self.save_path = save_path

    # @staticmethod
    # def xyzpose2volume(volume, xyz_pose, next_xyz_pose, i):
    #     pose_volume = np.zeros_like(volume, dtype=np.int8)
    #     next_pose_volume = np.zeros_like(volume, dtype=np.int8)
    #
    #     for j in range(xyz_pose.shape[1]):
    #         tmp = xyz_pose[0, j, :].astype(np.int32)
    #         joint_coor = (0, tmp[2], tmp[1], tmp[0])
    #         pose_volume[joint_coor] = 4
    #
    #         tmp = next_xyz_pose[0, j, :].astype(np.int32)
    #         joint_coor = (0, tmp[2], tmp[1], tmp[0])
    #         next_pose_volume[joint_coor] = 4
    #     obs = np.stack([volume, pose_volume], axis=-1)
    #     obs_next = np.stack([volume, next_pose_volume], axis=-1)
    #     return i, obs, obs_next
    #
    # def get_batch(self, batch_size, num_cpus=32):
    #     # Randomly sample batch_size examples
    #     # (obs_volume, obs_pose, acs, rs, next_obs_pose, gammas)
    #     batch = random.sample(self.buffer, batch_size)
    #     obs_volume, obs_pose, acs, rs, next_obs_pose, gammas = zip(*batch)
    #
    #     # convert xyz pose to volume pose
    #     pool = multiprocessing.Pool(num_cpus)
    #     results = []
    #     for i in range(batch_size):
    #         results.append(pool.apply_async(self.xyzpose2volume, (obs_volume[i], obs_pose[i], next_obs_pose[i], i,)))
    #     pool.close()
    #     pool.join()
    #     _, D, H, W = obs_volume[0].shape
    #     obs = np.zeros([batch_size, D, H, W, 2], dtype=np.int8)
    #     obs_next = np.zeros([batch_size, D, H, W, 2], dtype=np.int8)
    #     for result in results:
    #         tmp = result.get()  # i, obs, obs_next
    #         obs[tmp[0]: tmp[0]+1, ...] = tmp[1]
    #         obs_next[tmp[0]: tmp[0]+1, ...] = tmp[2]
    #
    #     # concatenate to array
    #     batch_data = [np.concatenate(item, axis=0) for item in [acs, rs, gammas]]
    #     batch_data += [obs, obs_next]
    #     # batch_data: [acs, rs, gammas, obs, obs_next]
    #     return batch_data

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        # (obs_volume, obs_pose, acs, rs, next_obs_pose, gammas)
        batch = random.sample(self.buffer, batch_size)
        obs_volume, obs_pose, acs, rs, next_obs_pose, gammas = zip(*batch)  # [N, W, H, D, 1]
        batch_data = [np.concatenate(item, axis=0) for item in [obs_volume, obs_pose, acs, rs, next_obs_pose, gammas]]
        obs = np.concatenate([batch_data[0], batch_data[1]], axis=4)
        obs_next = np.concatenate([batch_data[0], batch_data[4]], axis=4)
        # [acs, rs, gammas, obs, obs_next]
        return batch_data[2], batch_data[3], batch_data[5], obs, obs_next

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

"""
class SamplerOld(object):
    def __init__(self, actor, critic, env, dataset, gamma=0.9):
        self.actor = actor
        self.critic = critic
        self.env = env
        self.dataset = dataset
        self.predefined_bbx = self.env.predefined_bbx
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
"""


class Sampler(object):
    def __init__(self, actor, env, dataset, step_size, gamma=0.9):
        self.actor = actor
        self.env = env
        self.dataset = dataset
        self.predefined_bbx = self.env.predefined_bbx
        self.gamma = gamma
        self.step_size = step_size

    def collect_experiences_batch_examples(self, examples, sess, num_cpus):
        """
        collect experiences for a mini-batch of examples
        inputs
            examples: list of tuples,
            sess: for the prediction of init pose,
            num_cpus: multi-processing for batch examples
        """
        batch_size = len(examples)
        obs_volume = []
        obs_pose = []
        acs = []
        rs = []
        gammas = []
        max_error = []
        is_done = False
        self.env.reset(examples, sess)
        while not is_done:
            state, _ = self.env.get_obs()  # NWHDC, [N, num_joint, 3]
            ac_flat = self.actor.get_action(obs=state, step_size=self.step_size, dropout_prob=1.0)  # [N, ac_dims]
            r, is_done = self.env.step(ac_flat)  # [N, 1]

            obs_volume.append(state[..., 0:1])  # [N, W, H, D, 1]
            obs_pose.append(state[..., 1:2])  # [N, W, H, D, 1]
            acs.append(ac_flat)  # ac_flat: [N, ac_dims]
            rs.append(r)  # r: [N, 1]
            gammas.append(self.gamma * np.ones([batch_size, 1]))  # gamma: [N, 1]

        obs_volume = np.stack(obs_volume, axis=0)  # [max_iter, N, W, H, D]
        obs_pose = np.stack(obs_pose, axis=0)  # [max_iter, N, W, H, D]
        acs = np.stack(acs, axis=0)  # [max_iter, N, ac_dims]
        rs = np.stack(rs, axis=0)  # [max_iter, N, 1]
        gammas = np.stack(gammas, axis=0)  # [max_iter, N, 1]
        # print(obs_volume.shape, obs_pose.shape, acs.shape, rs.shape, gammas.shape)
        max_error.append(self.env.batch_max_error())

        results = []
        pool = multiprocessing.Pool(num_cpus)
        for i in range(batch_size):
            results.append(pool.apply_async(self.arrange_one_example,
                                            (obs_volume[:, i:i+1, ...], obs_pose[:, i:i+1, ...],
                                             acs[:, i:i+1, :], rs[:, i:i+1, :], gammas[:, i:i+1, :],)))
        pool.close()
        pool.join()
        samples = []
        for result in results:
            tmp = result.get()
            samples += tmp
        return samples, rs, max_error

    @staticmethod
    def arrange_one_example(volume, pose, ac, r, gamma):
        """
        volume: [max_iter, 1, W, H, D, 1]
        pose: [max_iter, 1,  W, H, D, 1]
        ac: [max_iter, 1, ac_dims]
        r: [max_iter, 1, 1]
        gamma: [max_iter, 1, 1]
        """
        volume = list(volume)
        pose = list(pose)
        ac = list(ac)
        r = list(r)
        gamma = list(gamma)
        # next obs
        next_pose = pose[1:]
        next_pose.append(pose[-1])
        gamma[-1] = np.array([[0]])

        samples = list(zip(volume, pose, ac, r, next_pose, gamma))
        return samples

    def collect_experiences(self, num_files, num_batch_samples, batch_size, sess, num_cpus):
        mul_experiences, rs_list, max_error_list = [], [], []
        examples = random.sample(self.dataset.get_batch_samples_training(num_files), num_batch_samples)
        for i in range(num_batch_samples//batch_size+1):
            batch_examples = examples[i*batch_size: min((i+1)*batch_size, num_batch_samples)]
            samples, rs, _max_error = self.collect_experiences_batch_examples(batch_examples, sess, num_cpus)
            mul_experiences += samples
            rs_list.append(rs[-1, :, 0])  # rs [max_iter, N, 1] to [N]
            max_error_list.append(_max_error)
        rs = np.hstack(rs_list)
        max_error = np.hstack(max_error_list)
        print('%i experiences were collected, %i examples' % (len(mul_experiences), num_batch_samples))
        print('average reward: {:.4f}, average max error: {:.4f}'.format(np.mean(rs), np.mean(max_error)))
        return mul_experiences, rs

    def aggregate_test_samples(self):
        examples = []
        for example in self.dataset.get_samples_testing():
            examples += example
        return examples

    def test_batch_samples(self, examples, batch_size, sess):
        N = len(examples)
        max_error, rs = [], []
        for i in range(N//batch_size+1):
            self.env.reset(examples[i*batch_size: min((i+1)*batch_size, N)], sess)
            is_done = False
            while not is_done:
                state, pose = self.env.get_obs()  # NWHDC
                ac_flat = self.actor.get_action(obs=state, step_size=self.step_size, dropout_prob=1.0)  # [N, ac_dims]
                r, is_done = self.env.step(ac_flat)  # [N, 1]
            max_error.append(self.env.batch_max_error())
            rs.append(r[:, 0])
        max_error = np.hstack(max_error)
        # print(max_error.shape)
        rs = np.hstack(rs)
        return max_error, rs

