import sys
sys.path.append('..')
from unmaintained.ac_model import Actor, Critic
from data_preprocessing.nyu_dataset import NYUDataset
from data_preprocessing.icvl_dataset import ICVLDataset
from data_preprocessing.mrsa_dataset import MRSADataset
from unmaintained.environment import HandEnv
from unmaintained.sampler import Sampler
import os
import tensorflow as tf
import utils
import numpy as np


def test(config):
    if config['dataset'] == 'nyu':
        dataset = NYUDataset(subset='testing', root_dir='/hand_pose_data/nyu/')
    elif config['dataset'] == 'icvl':
        dataset = ICVLDataset(subset='testing', root_dir='/hand_pose_data/icvl/')
    elif config['dataset'] == 'mrsa15':
        dataset = MRSADataset(subset='testing', test_fold=config['mrsa_test_fold'],
                              root_dir='../../../hand_pose_data/mrsa15/')
    else:
        raise ValueError('Dataset name %s error...' % config['dataset'])

    actor_root = Actor(scope='actor_root',
                       tau=config['tau'],
                       lr=config['learning_rate'],
                       obs_dims=config['root_obs_dims'],
                       cnn_layer=config['root_actor_cnn_layers'],
                       fc_layer=config['root_actor_fc_layers'])
    critic_root = Critic(scope='critic_root',
                         tau=config['tau'],
                         lr=config['learning_rate'],
                         obs_dims=config['root_obs_dims'],
                         cnn_layer=config['root_critic_cnn_layers'],
                         fc_layer=config['root_critic_fc_layers'])
    actor_chain = Actor(scope='actor_chain',
                        tau=config['tau'],
                        lr=config['learning_rate'],
                        obs_dims=config['chain_obs_dims'],
                        cnn_layer=config['chain_actor_cnn_layers'],
                        fc_layer=config['chain_actor_fc_layers'])
    critic_chain = Critic(scope='critic_chain',
                          tau=config['tau'],
                          lr=config['learning_rate'],
                          obs_dims=config['chain_obs_dims'],
                          cnn_layer=config['chain_critic_cnn_layers'],
                          fc_layer=config['chain_critic_fc_layers'])
    env = HandEnv(dataset=config['dataset'],
                  subset='training',
                  iter_per_joint=config['iter_per_joint'],
                  reward_beta=config['beta'])
    sampler = Sampler(actor_root, critic_root, actor_chain, critic_chain, env, dataset)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        # load actor root model
        root_dir = config['saved_model_path'] + '/' + config['dataset'] + '/'
        saved_actor_dir = root_dir + config['actor_model_name'] + '_root.pkl'
        if os.path.exists(saved_actor_dir):
            utils.loadFromFlat(actor_root.get_trainable_variables(), saved_actor_dir)
            print("Actor parameter loaded from %s" % saved_actor_dir)
        else:
            ValueError('Dir %s errors...' % saved_actor_dir)
        actor_root.load_sess(sess)

        # load critic root model
        saved_critic_dir = root_dir + config['critic_model_name'] + '_root.pkl'
        if os.path.exists(saved_critic_dir):
            utils.loadFromFlat(critic_root.get_trainable_variables(), saved_critic_dir)
            print("Critic parameter loaded from %s" % saved_critic_dir)
        else:
            ValueError('Dir %s errors...' % saved_critic_dir)
        critic_root.load_sess(sess)

        # load actor chain model
        saved_actor_dir = root_dir + config['actor_model_name'] + '_chain.pkl'
        if os.path.exists(saved_actor_dir):
            utils.loadFromFlat(actor_chain.get_trainable_variables(), saved_actor_dir)
            print("Actor_chain parameter loaded from %s" % saved_actor_dir)
        else:
            ValueError('Dir %s errors...' % saved_actor_dir)
        actor_chain.load_sess(sess)

        # load critic chain model
        saved_critic_dir = root_dir + config['critic_model_name'] + '_chain.pkl'
        if os.path.exists(saved_critic_dir):
            utils.loadFromFlat(critic_chain.get_trainable_variables(), saved_critic_dir)
            print("Critic_chain parameter loaded from %s" % saved_critic_dir)
        else:
            ValueError('Dir %s errors...' % saved_critic_dir)
        critic_chain.load_sess(sess)

        # test
        results = sampler.test_multiple_samples()



def get_config():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu_id', '-id', type=str, default='0')
    parser.add_argument('--saved_model_path', '-smp', type=str, default='../results/unmaintained/')
    parser.add_argument('--actor_model_name', '-amn', type=str, default='actor')
    parser.add_argument('--critic_model_name', '-cmn', type=str, default='critic')
    parser.add_argument('--batch_size', '-bs', type=int, default=128)
    parser.add_argument('--n_rounds', '-nr', type=int, default=1000)
    parser.add_argument('--update_iters', '-ui', type=int, default=1000)
    parser.add_argument('--n_iters', '-ni', type=int, default=10)
    parser.add_argument('--iter_per_joint', '-ipj', type=str, default='(5, 3)')
    parser.add_argument('--beta', '-beta', type=float, default=0.1)
    parser.add_argument('--dataset', '-data', type=str, default='icvl')
    parser.add_argument('--mrsa_test_fold', '-mtf', type=str, default='P9')
    parser.add_argument('--files_per_time', '-nfp', type=int, default=1)
    parser.add_argument('--buffer_size', '-buf', type=int, default=100000)

    # parameter of models
    parser.add_argument('--root_actor_cnn_layers', '-racl', type=str, default='(8, 16, 32, 64, 128)')
    parser.add_argument('--root_critic_cnn_layers', '-rccl', type=str, default='(8, 16, 32, 64, 128)')
    parser.add_argument('--root_actor_fc_layers', '-rafl', type=str, default='(256, 32)')
    parser.add_argument('--root_critic_fc_layers', '-rcfl', type=str, default='(64, 6, 32, 64)')
    parser.add_argument('--root_obs_dims', '-rod', type=str, default='(40, 40, 20)')
    parser.add_argument('--tau', '-tau', type=float, default=0.001)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5)

    parser.add_argument('--chain_actor_cnn_layers', '-racl', type=str, default='(8, 16, 32, 64, 128)')
    parser.add_argument('--chain_critic_cnn_layers', '-rccl', type=str, default='(8, 16, 32, 64, 128)')
    parser.add_argument('--chain_actor_fc_layers', '-rafl', type=str, default='(128, 32)')
    parser.add_argument('--chain_critic_fc_layers', '-rcfl', type=str, default='(64, 6, 32, 64)')
    parser.add_argument('--chain_obs_dims', '-rod', type=str, default='(30, 30, 20)')

    args = vars(parser.parse_args())
    utils.print_args(args)
    args = utils.str2int_tuple(args)
    return args


def main():
    config = get_config()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']


if __name__ == '__main__':
    main()
