import sys
sys.path.append('..')
from data_preprocessing.nyu_dataset import NYUDataset
from data_preprocessing.icvl_dataset import ICVLDataset
from data_preprocessing.mrsa_dataset import MRSADataset
from model2.environment import HandEnv
from model2.ac_model import Pretrain
import os
import tensorflow as tf
import utils
import numpy as np
import time
import random
from tensorboardX import SummaryWriter
import multiprocessing


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


def collect_one_sample(example_volume, example_pose, home_pose_volume, W, H, D, env, ac_dim):
    volume = np.stack([example_volume, home_pose_volume], axis=3)[np.newaxis, :]  # NWHDC
    assert volume.shape == (1, W, H, D, 2)
    volume = np.transpose(volume, [0, 3, 2, 1, 4])  # NDHWC
    label_lie_algebra = env.pose_to_lie_algebras(example_pose)[np.newaxis, :]
    assert label_lie_algebra.shape == (1, ac_dim)
    return volume, label_lie_algebra


def collect_train_samples(env, dataset, home_pose_volume, num_files, num_samples, num_cpus):
    ac_dim = 4 * dataset.jnt_num - 1
    W, H, D = dataset.predefined_bbx[0] + 1, dataset.predefined_bbx[1] + 1, dataset.predefined_bbx[2] + 1
    train_samples = []
    train_label = []
    examples = random.sample(dataset.get_batch_samples_training(num_files), num_samples)
    # example (filename, xyz_pose, depth_img, pose_bbx, cropped_point,
    #          coeff, normalized_rotate_pose, normalized_rotate_points, rotated_bbx, volume)
    results = []
    pool = multiprocessing.Pool(num_cpus)
    for example in examples:
        results.append(pool.apply_async(collect_one_sample,
                                        (example[9], example[6], home_pose_volume, W, H, D, env, ac_dim,)))
    pool.close()
    pool.join()
    pool.terminate()
    for result in results:
        volume, label_lie_algebra = result.get()
        train_samples.append(volume)
        train_label.append(label_lie_algebra)
    train_samples = np.concatenate(train_samples, axis=0)
    train_label = np.concatenate(train_label, axis=0)
    return train_samples, train_label


def collect_test_samples(env, dataset, home_pose_volume, num_cpus):
    print('Loading test samples...')
    ac_dim = 4 * dataset.jnt_num - 1
    W, H, D = dataset.predefined_bbx[0] + 1, dataset.predefined_bbx[1] + 1, dataset.predefined_bbx[2] + 1
    test_samples = []
    test_label = []
    examples = []
    for data in dataset.get_samples_testing():
        examples += data

    results = []
    pool = multiprocessing.Pool(num_cpus)
    for example in examples:
        results.append(pool.apply_async(collect_one_sample,
                                        (example[9], example[6], home_pose_volume, W, H, D, env, ac_dim,)))
    pool.close()
    pool.join()
    pool.terminate()
    for result in results:
        volume, label_lie_algebra = result.get()
        test_samples.append(volume)
        test_label.append(label_lie_algebra)
    test_samples = np.concatenate(test_samples, axis=0)
    test_label = np.concatenate(test_label, axis=0)
    return test_samples, test_label


def pre_train(config):
    if config['dataset'] == 'nyu':
        dataset = NYUDataset(subset='training', root_dir='/hand_pose_data/nyu/', predefined_bbx=(63, 63, 31))
        ac_dim = 4 * dataset.jnt_num - 1
        # (160, 120, 70), 6 * 14 = 84
        cnn_layer = (8, 16, 32, 64, 128)  # 512
        fc_layer = (512, ac_dim, 512, 256)
    elif config['dataset'] == 'icvl':
        dataset = ICVLDataset(subset='training', root_dir='/hand_pose_data/icvl/', predefined_bbx=(63, 63, 31))
        ac_dim = 4 * dataset.jnt_num - 1
        # (140, 120, 60), 6 * 16 = 96
        cnn_layer = (8, 16, 32, 64, 128)  # 512
        fc_layer = (512, ac_dim, 512, 256)
    elif config['dataset'] == 'mrsa15':
        # (180, 120, 70), 6 * 21 = 126
        dataset = MRSADataset(subset='training', test_fold=config['mrsa_test_fold'],
                              root_dir='/hand_pose_data/mrsa15/', predefined_bbx=(63, 63, 31))
        ac_dim = 4 * dataset.jnt_num - 1
        cnn_layer = (8, 16, 32, 64, 128)  # 512
        fc_layer = (512, ac_dim, 512, 256)
    else:
        raise ValueError('Dataset name %s error...' % config['dataset'])
    obs_dims = (dataset.predefined_bbx[2] + 1, dataset.predefined_bbx[1] + 1, dataset.predefined_bbx[0] + 1, 2)
    env = HandEnv(dataset=config['dataset'],
                  subset='training',
                  max_iters=5,
                  predefined_bbx=dataset.predefined_bbx)
    home_pose_volume = get_pose_volume(env.home_pose, dataset.predefined_bbx)
    home_lie_algebra = env.pose_to_lie_algebras(env.home_pose)[np.newaxis, :]
    scope = 'pre_train'
    batch_size = config['batch_size']

    # define model and loss
    model = Pretrain(scope, obs_dims, cnn_layer, fc_layer, ac_dim)  # model.obs, model.ac, model.dropout_prob
    label = tf.placeholder(shape=(None, ac_dim), dtype=tf.float32, name='action')
    tf_mse = tf.reduce_sum(0.5 * tf.square(model.ac - label), axis=1)
    tf_loss = tf.reduce_mean(tf_mse)

    global_step = tf.Variable(0, trainable=False, name='step')
    lr = tf.train.exponential_decay(config['lr_start'], global_step,
                                    config['lr_decay_iters'], config['lr_decay_rate'])
    # optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=tf_loss, global_step=global_step)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss=tf_loss, global_step=global_step)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        root_dir = config['saved_model_path'] + config['dataset'] + '/'
        writer = SummaryWriter(log_dir=root_dir)

        model_save_dir = root_dir + config['dataset'] + '_pretrain.pkl'
        if os.path.exists(model_save_dir) and not config['new_training']:
            utils.loadFromFlat(model.get_trainable_variables(), model_save_dir)
            print("Pre-train parameter loaded from %s" % model_save_dir)
        else:
            print("[Warning]: initialize the pre-train model")

        x_test, y_test = collect_test_samples(env, dataset, home_pose_volume, config['num_cpus'])
        n_test = x_test.shape[0]
        print('test samples %i' % n_test)
        best_loss = 1000
        i = 0
        while i < config['n_rounds']:
            i += 1
            print('--------------------------------Round % i---------------------------------' % i)
            # test
            loss_list = []
            if i % config['test_gap'] == 1:
                start_time = time.time()
                for j in range(n_test // batch_size + 1):
                    idx1 = j * batch_size
                    idx2 = min((j + 1) * batch_size, n_test)
                    batch_loss = sess.run(tf_mse,
                                          feed_dict={model.obs: x_test[idx1: idx2],
                                                     label: y_test[idx1: idx2],
                                                     model.home_lie_algebra: home_lie_algebra,
                                                     model.dropout_prob: 1.0})
                    loss_list.append(batch_loss)
                test_loss = np.mean(np.hstack(loss_list))
                writer.add_scalar(config['dataset'] + '_test_loss', test_loss, i)
                end_time = time.time()
                print('>>> Testing loss: {:.4f}, best loss {:.4f}, time used: {:.2f}s'
                      .format(test_loss, best_loss, end_time - start_time))
                if best_loss > test_loss:
                    utils.saveToFlat(model.get_trainable_variables(), model_save_dir)
                    best_loss = test_loss.copy()
                    print('>>> Model saved... best loss {:.4f}'.format(best_loss))

            # train
            start_time = time.time()
            x_train, y_train = collect_train_samples(env, dataset, home_pose_volume,
                                                     config['files_per_time'],
                                                     config['samples_per_time'],
                                                     config['num_cpus'])
            print('Collected samples {}'.format(x_train.shape[0]))
            loss_list = []
            for _ in range(config['train_iters']):
                batch_idx = np.random.randint(0, x_train.shape[0], batch_size)
                _, batch_loss, step = sess.run([optimizer, tf_loss, global_step],
                                               feed_dict={model.obs: x_train[batch_idx],
                                                          label: y_train[batch_idx],
                                                          model.home_lie_algebra: home_lie_algebra,
                                                          model.dropout_prob: 0.5})
                loss_list.append(batch_loss)
            end_time = time.time()
            writer.add_scalar(config['dataset'] + '_train_loss', np.mean(loss_list), i)
            print('Training loss: {:.4f}, time used: {:.2f}s, step: {:d}'
                  .format(np.mean(loss_list), end_time - start_time, step))
        writer.close()


def get_config():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu_id', '-id', type=str, default='0')
    parser.add_argument('--saved_model_path', '-smp', type=str, default='../results/')
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--n_rounds', '-nr', type=int, default=800)
    parser.add_argument('--train_iters', '-ni', type=int, default=100)
    parser.add_argument('--dataset', '-data', type=str, default='mrsa15')
    parser.add_argument('--mrsa_test_fold', '-mtf', type=str, default='P8')
    parser.add_argument('--files_per_time', '-fpt', type=int, default=10)
    parser.add_argument('--samples_per_time', '-spt', type=int, default=2000)
    parser.add_argument('--lr_start', '-lr', help='learning rate', type=float, default=0.0001)
    parser.add_argument('--lr_decay_rate', default=0.99)
    parser.add_argument('--lr_decay_iters', default=500)
    parser.add_argument('--new_training', '-new', type=bool, default=1)
    parser.add_argument('--test_gap', '-tg', type=int, default=2)
    parser.add_argument('--num_cpus', '-cpus', type=int, default=20)
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
