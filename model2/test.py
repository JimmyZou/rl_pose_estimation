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
import pickle


def train(config):
    if config['dataset'] == 'nyu':
        dataset = NYUDataset(subset='testing', root_dir='/home/data/nyu/')
        # (240, 180, 70), 6 * 14 = 84
        actor_cnn_layer = (4, 8, 16, 32, 64, 128)
        actor_fc_layer = (512, 512, 256)
        critic_cnn_layer = (4, 8, 16, 32, 64, 128)  # 768
        critic_fc_layer = (512, 84, 512, 128)
    elif config['dataset'] == 'icvl':
        dataset = ICVLDataset(subset='testing', root_dir='/hand_pose_data/icvl/')
        # (140, 120, 60), 6 * 16 = 96
        actor_cnn_layer = (4, 8, 16, 32, 64)
        actor_fc_layer = (512, 512, 256)
        critic_cnn_layer = (4, 8, 16, 32, 64)  # 768
        critic_fc_layer = (512, 96, 512, 128)
    elif config['dataset'] == 'mrsa15':
        # (180, 120, 70), 6 * 21 = 126
        dataset = MRSADataset(subset='testing', test_fold=config['mrsa_test_fold'],
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
    sampler = Sampler(actor, critic, env, dataset, config['gamma'])

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())

        root_dir = config['saved_model_path'] + '/' + config['dataset'] + '/'
        # actor model
        save_actor_dir = root_dir + 'actor.pkl'
        if os.path.exists(save_actor_dir):
            utils.loadFromFlat(actor.get_trainable_variables(), save_actor_dir)
            print("Actor parameter loaded from %s" % save_actor_dir)
        else:
            ValueError('Actor parameter cannot be found at %s' % save_actor_dir)
        sess.run(actor.update_target_ops())
        actor.load_sess(sess)

        # critic model
        save_critic_dir = root_dir + 'critic.pkl'
        if os.path.exists(save_critic_dir):
            utils.loadFromFlat(critic.get_trainable_variables(), save_critic_dir)
            print("Critic parameter loaded from %s" % save_critic_dir)
        else:
            ValueError('Critic parameter cannot be found at %s' % save_critic_dir)
        sess.run(critic.update_target_ops)
        critic.load_sess(sess)

        # testing
        # list of tuples: (final_avg_distance, final_pose, filename, xyz_pose, depth_img, pose_bbx,
        #                  coeff, normalized_rotate_pose, rotated_bbx)
        results = sampler.test_multiple_samples()

        predictions = []
        for result in results:
            pred_pose = transfer_pose(result, dataset.predefined_bbx)
            # tuple: (filename, xyz_pose, pred_pose, depth_img)
            predictions.append((result[2], result[3], pred_pose, result[4]))
        with open(root_dir + 'predictions.pkl', 'wb') as f:
            pickle.dump(predictions, f)


def transfer_pose(example, predefined_bbx):
    pred_pose = example[1]
    rotated_bbx = example[8]
    x_min, x_max, y_min, y_max, z_min, z_max = rotated_bbx
    predefined_bbx = np.asarray([predefined_bbx])
    point1 = np.array([[x_min, y_min, z_min]])
    point2 = np.array([[x_max, y_max, z_max]])
    resize_ratio = predefined_bbx / (point2 - point1)
    rotated_pose = pred_pose / resize_ratio + point1

    coeff = example[6]
    x_min, x_max, y_min, y_max, z_min, z_max = example[5]
    xyz_pose = np.dot(rotated_pose, coeff.T) + np.array([[x_min, y_min, z_min]])
    return xyz_pose
