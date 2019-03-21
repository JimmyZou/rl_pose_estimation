import sys
sys.path.append('..')
from data_preprocessing.nyu_dataset import NYUDataset
from data_preprocessing.icvl_dataset import ICVLDataset
from data_preprocessing.mrsa_dataset import MRSADataset
from model2.environment import HandEnv
import os
import tensorflow as tf
import utils
import numpy as np
import time
import random


def pre_train_model(scope, obs_dims, cnn_layer, fc_layer, ac_dim):
    print('building model %s' % scope)
    with tf.variable_scope(scope):
        obs = tf.placeholder(shape=(None,) + obs_dims, dtype=tf.float32, name='state')

        last_out = tf.identity(obs)
        for idx, i in enumerate(cnn_layer):
            last_out = tf.contrib.layers.conv3d(inputs=last_out,
                                                num_outputs=i,
                                                kernel_size=5,
                                                activation_fn=tf.nn.elu,
                                                stride=1,
                                                padding='SAME',
                                                data_format='NDHWC',
                                                scope='3dcnn%i' % idx)
            last_out = tf.contrib.layers.max_pool3d(inputs=last_out,
                                                    kernel_size=[2, 2, 2],
                                                    stride=2,
                                                    padding='SAME',
                                                    data_format='NDHWC',
                                                    scope='maxpooling%i' % idx)
        fc_out = tf.contrib.layers.flatten(last_out, scope='flatten')
        for idx, i in enumerate(fc_layer):
            fc_out = tf.contrib.layers.fully_connected(inputs=fc_out,
                                                       num_outputs=i,
                                                       activation_fn=tf.nn.elu,
                                                       scope='fc%i' % idx)
        # the last layer
        ac = tf.contrib.layers.fully_connected(inputs=fc_out, num_outputs=ac_dim,
                                               activation_fn=None, scope='last_fc')
    return obs, ac


def get_pose_volume(xyz_pose, bbx):
    x_max, y_max, z_max = bbx
    # WHD
    pose_volume = np.zeros([x_max+1, y_max+1, z_max+1], dtype=np.int8)
    for j in range(xyz_pose.shape[0]):
        tmp = xyz_pose[j, :].astype(np.int32)
        joint_coor = (tmp[0], tmp[1], tmp[2])
        pose_volume[joint_coor] = 2

    # import matplotlib.pyplot as plt
    # points = np.asarray(np.where(pose_volume[0] > 0)).T[:, 0:2]
    # plt.figure()
    # plt.scatter(points[:, 0], points[:, 1], c='b', s=3)
    # plt.axis('off')
    # plt.show()

    return pose_volume


def collect_train_samples(env, dataset, home_pose_volume, num_files, num_samples):
    ac_dim = 4 * dataset.jnt_num - 1
    W, H, D = dataset.predefined_bbx[0] + 1, dataset.predefined_bbx[1] + 1, dataset.predefined_bbx[2] + 1
    train_samples = []
    train_label = []
    examples = random.sample(dataset.get_batch_samples_training(num_files), num_samples)
    # example (filename, xyz_pose, depth_img, pose_bbx, cropped_point,
    #          coeff, normalized_rotate_pose, normalized_rotate_points, rotated_bbx, volume)
    for example in examples:
        try:
            volume = np.stack([example[9], home_pose_volume], axis=3)[np.newaxis, :]  # NWHDC
            assert volume.shape == (1, W, H, D, 2)
            train_samples.append(np.transpose(volume, [0, 3, 2, 1, 4]))  # NDHWC
            label_lie_algebra = env.pose_to_lie_algebras(example[6])[np.newaxis, :]
            assert label_lie_algebra.shape == (1, ac_dim)
            train_label.append(label_lie_algebra)
        except:
            print(example[9].shape)
            print('[error] filename {} errors with shape'.format(example[0], example[9].shape))

    train_samples = np.concatenate(train_samples, axis=0)
    train_label = np.concatenate(train_label, axis=0)
    return train_samples, train_label


def pre_train(config):
    if config['dataset'] == 'nyu':
        dataset = NYUDataset(subset='training', root_dir='/home/data/nyu/', predefined_bbx=(63, 63, 31))
        # (160, 120, 70), 6 * 14 = 84
        cnn_layer = (8, 16, 32, 64, 128)  # 512
        fc_layer = (512, 512, 128)
    elif config['dataset'] == 'icvl':
        dataset = ICVLDataset(subset='training', root_dir='/hand_pose_data/icvl/', predefined_bbx=(63, 63, 31))
        # (140, 120, 60), 6 * 16 = 96
        cnn_layer = (8, 16, 32, 64, 128)  # 512
        fc_layer = (512, 512, 128)
    elif config['dataset'] == 'mrsa15':
        # (180, 120, 70), 6 * 21 = 126
        dataset = MRSADataset(subset='training', test_fold=config['mrsa_test_fold'],
                              root_dir='/hand_pose_data/mrsa15/', predefined_bbx=(63, 63, 31))
        cnn_layer = (8, 16, 32, 64, 128)  # 512
        fc_layer = (512, 512, 128)
    else:
        raise ValueError('Dataset name %s error...' % config['dataset'])
    ac_dim = 4 * dataset.jnt_num - 1
    obs_dims = (dataset.predefined_bbx[2] + 1, dataset.predefined_bbx[1] + 1, dataset.predefined_bbx[0] + 1, 2)
    env = HandEnv(dataset=config['dataset'],
                  subset='training',
                  max_iters=5,
                  predefined_bbx=dataset.predefined_bbx)
    home_pose_volume = get_pose_volume(env.home_pose, dataset.predefined_bbx)
    scope = 'pre_train'
    batch_size = config['batch_size']

    # define model and loss
    input, output = pre_train_model(scope, obs_dims, cnn_layer, fc_layer, ac_dim)
    label = tf.placeholder(shape=(None, ac_dim), dtype=tf.float32, name='action')
    tf_loss = tf.reduce_mean(tf.reduce_sum(0.5 * tf.square(label - output), axis=1))

    global_step = tf.Variable(0, trainable=False, name='step')
    lr = tf.train.exponential_decay(config['lr_start'], global_step,
                                    config['lr_decay_iters'], config['lr_decay_rate'])
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=tf_loss, global_step=global_step)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        root_dir = config['saved_model_path'] + config['dataset'] + '/'

        model_save_dir = root_dir + config['dataset'] + '_pretrain.pkl'
        if os.path.exists(model_save_dir):
            utils.loadFromFlat(trainable_vars, model_save_dir)
            print("Pre-train parameter loaded from %s" % model_save_dir)
        else:
            print("[Warning]: initialize the pre-train model")

        i = 0
        while i < config['n_rounds']:
            i += 1
            print('--------------------------------Round % i---------------------------------' % i)
            start_time = time.time()
            x_train, y_train = collect_train_samples(env, dataset, home_pose_volume,
                                                     config['files_per_time'],
                                                     config['samples_per_time'])
            N = x_train.shape[0]
            print('Collected samples {}'.format(x_train.shape[0]))
            loss_list = []
            for _ in range(config['train_iters']):
                batch_idx = np.random.randint(0, N, batch_size)
                _, batch_loss, step = sess.run([optimizer, tf_loss, global_step],
                                               feed_dict={input: x_train[batch_idx], label: y_train[batch_idx]})
                loss_list.append(batch_loss)
            end_time = time.time()
            print('Training loss: {:.4f}, time used: {:.2f}s, step: {:d}'
                  .format(np.mean(loss_list), end_time - start_time, step))
            utils.saveToFlat(trainable_vars, model_save_dir)


def get_config():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu_id', '-id', type=str, default='0')
    parser.add_argument('--saved_model_path', '-smp', type=str, default='../results/')
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--n_rounds', '-nr', type=int, default=1000)
    parser.add_argument('--train_iters', '-ni', type=int, default=100)
    parser.add_argument('--dataset', '-data', type=str, default='mrsa15')
    parser.add_argument('--mrsa_test_fold', '-mtf', type=str, default='P8')
    parser.add_argument('--files_per_time', '-fpt', type=int, default=10)
    parser.add_argument('--samples_per_time', '-spt', type=int, default=2000)
    parser.add_argument('--lr_start', '-lr', help='learning rate', type=float, default=0.00001)
    parser.add_argument('--lr_decay_rate', default=0.99)
    parser.add_argument('--lr_decay_iters', default=250)
    args = vars(parser.parse_args())
    utils.print_args(args)
    return args


def main():
    config = get_config()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']
    pre_train(config)


if __name__ == '__main__':
    main()