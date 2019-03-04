import sys
sys.path.append('..')
from data_preprocessing.nyu_dataset import NYUDataset
from data_preprocessing.icvl_dataset import ICVLDataset
from data_preprocessing.mrsa_dataset import MRSADataset
from model2.environment import HandEnv
from model2.sampler import ReplayBuffer, Sampler
from model2.ac_model import Actor, Critic
import os
import tensorflow as tf
import utils
import numpy as np


def train(config):
    if config['dataset'] == 'nyu':
        dataset = NYUDataset(subset='training', root_dir='/home/data/nyu/')
        # (240, 180, 70), 6 * 14 = 84
        actor_cnn_layer = (4, 8, 16, 32, 64, 128)
        actor_fc_layer = (512, 512, 256)
        critic_cnn_layer = (4, 8, 16, 32, 64, 128)  # 768
        critic_fc_layer = (512, 84, 512, 128)
    elif config['dataset'] == 'icvl':
        dataset = ICVLDataset(subset='training', root_dir='/hand_pose_data/icvl/')
        # (140, 120, 60), 6 * 16 = 96
        actor_cnn_layer = (4, 8, 16, 32, 64)
        actor_fc_layer = (512, 512, 256)
        critic_cnn_layer = (4, 8, 16, 32, 64)  # 768
        critic_fc_layer = (512, 96, 512, 128)
    elif config['dataset'] == 'mrsa15':
        # (180, 120, 70), 6 * 21 = 126
        dataset = MRSADataset(subset='training', test_fold=config['mrsa_test_fold'],
                              root_dir='/hand_pose_data/mrsa15/')
        actor_cnn_layer = (4, 8, 16, 64, 128, 256)
        actor_fc_layer = (512, 512, 256)
        critic_cnn_layer = (4, 8, 16, 64, 128, 256)  # 512
        critic_fc_layer = (512, 126, 512, 128)
    else:
        raise ValueError('Dataset name %s error...' % config['dataset'])

    actor = Actor(scope='actor',
                  obs_dims=dataset.predefined_bbx,
                  ac_dim=6 * dataset.jnt_num,
                  cnn_layer=actor_cnn_layer,
                  fc_layer=actor_fc_layer,
                  tau=config['tau'],
                  lr=config['actor_lr'])
    critic = Critic(scope='critic',
                    obs_dims=dataset.predefined_bbx,
                    ac_dim=6 * dataset.jnt_num,
                    cnn_layer=critic_cnn_layer,
                    fc_layer=critic_fc_layer,
                    tau=config['tau'],
                    lr=config['critic_lr'])
    env = HandEnv(dataset=config['dataset'],
                  subset='training',
                  max_iters=config['max_iters'],
                  predefined_bbx=dataset.predefined_bbx)
    buffer = ReplayBuffer(buffer_size=config['buffer_size'])
    sampler = Sampler(actor, critic, env, dataset, config['gamma'])

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())

        root_dir = config['saved_model_path'] + '/' + config['dataset'] + '/'
        # actor model
        saved_actor_dir = root_dir + 'actor.pkl'
        if os.path.exists(saved_actor_dir):
            utils.loadFromFlat(actor.get_trainable_variables(), saved_actor_dir)
            print("Actor parameter loaded from %s" % saved_actor_dir)
        else:
            print("[Warning]: initialize the actor model")
        actor.load_sess(sess)
        sess.run(actor.update_target_ops)

        # critic model
        saved_critic_dir = root_dir + 'critic.pkl'
        if os.path.exists(saved_critic_dir):
            utils.loadFromFlat(critic.get_trainable_variables(), saved_critic_dir)
            print("Critic parameter loaded from %s" % saved_critic_dir)
        else:
            print("[Warning]: initialize critic root model")
        critic.load_sess(sess)
        sess.run(critic.update_target_ops)

        i = 0
        while i < config['n_rounds']:
            i += 1
            print('--------------------------------Round % i---------------------------------' % i)
            # sampling
            samples = sampler.collect_multiple_samples(config['files_per_time'])
            buffer.add(samples)

            # # training
            # for _ in range(config['n_iters']):
            #     actor_loss_list, q_loss_list = [], []
            #     for _ in range(config['update_iters']):
            #         # get a mini-batch of data
            #         state, action, reward, new_state, gamma = buffer.get_batch(config['batch_size'])
            #         # update actor
            #         q_gradient = critic.get_q_gradient(obs=state, ac=action)
            #         _, actor_loss = actor.train(q_gradient=q_gradient, obs=state)
            #         # update critic
            #         next_ac = actor.get_target_action(obs=new_state)
            #         _, q_loss = critic.train(obs=state, ac=action, next_obs=new_state,
            #                                  next_ac=next_ac, r=reward, gamma=gamma)
            #         actor_loss_list.append(actor_loss)
            #         q_loss_list.append(q_loss)
            #     # update target network
            #     sess.run(actor.update_target_ops)
            #     sess.run(critic.update_target_ops)
            #     print('Actor average loss: {:.4f}, Critic: {:.4f}'
            #           .format(np.mean(actor_loss_list), np.mean(q_loss_list)))
            #
            # utils.saveToFlat(actor.get_trainable_variables(), saved_actor_dir)
            # utils.saveToFlat(critic.get_trainable_variables(), saved_critic_dir)

            pass


def get_config():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu_id', '-id', type=str, default='0')
    parser.add_argument('--saved_model_path', '-smp', type=str, default='../results/model2/')
    parser.add_argument('--batch_size', '-bs', type=int, default=128)
    parser.add_argument('--n_rounds', '-nr', type=int, default=100)
    parser.add_argument('--update_iters', '-ui', type=int, default=1000)
    parser.add_argument('--n_iters', '-ni', type=int, default=10)
    parser.add_argument('--dataset', '-data', type=str, default='icvl')
    parser.add_argument('--mrsa_test_fold', '-mtf', type=str, default='P9')
    parser.add_argument('--files_per_time', '-nfp', type=int, default=1)
    parser.add_argument('--max_iters', '-mi', type=int, default=2)
    parser.add_argument('--buffer_size', '-buf', type=int, default=10000)
    parser.add_argument('--tau', '-tau', type=float, default=0.001)
    parser.add_argument('--actor_lr', '-alr', type=float, default=1e-5)
    parser.add_argument('--critic_lr', '-clr', type=float, default=1e-5)
    parser.add_argument('--gamma', '-gamma', type=float, default=0.9)
    args = vars(parser.parse_args())
    utils.print_args(args)
    return args


def main():
    config = get_config()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']
    train(config)


if __name__ == '__main__':
    main()
