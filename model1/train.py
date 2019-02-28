from model1.ac_model import Actor, Critic
from data_preprocessing.nyu_dataset import NYUDataset
from model1.environment import HandEnv
from model1.sampler import ReplayBuffer, Sampler
import os
import tensorflow as tf


def train_root():
    actor_root = Actor(scope='actor_root', tau=0.001, obs_dims=(40, 40, 20),
                       cnn_layer=(8, 16, 32, 64, 128), fc_layer=(128, 32))
    critic_root = Critic(scope='critic_root', tau=0.001, obs_dims=(40, 40, 20),
                         cnn_layer=(8, 16, 32, 64, 128), fc_layer=(64, 20, 20, 10))

    dataset = NYUDataset(subset='training', root_dir='../../../hand_pose_data/nyu/')
    env = HandEnv(dataset, 'training', iter_per_joint=(5, 3), reward_beta=0.1)
    root_buffer = ReplayBuffer(buffer_size=1000)
    sampler = Sampler(actor_root, critic_root, None, None, env, dataset, root_buffer, None)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        actor_root.load_sess(sess)
        sess.run(actor_root.update_target_ops)
        critic_root.load_sess(sess)
        sess.run(critic_root.update_target_ops)

        sampler.collect_multiple_samples_root(num_files=2)


def train_chain():
    actor_root = Actor(scope='actor_root', tau=0.001)
    critic_root = Critic(scope='critic_root', tau=0.001)
    actor_chain = Actor(scope='actor_root', tau=0.001)
    critic_chain = Critic(scope='critic_root', tau=0.001)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'


