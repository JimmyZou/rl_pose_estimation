import sys
sys.path.append('..')
from data_preprocessing.nyu_dataset import NYUDataset
from data_preprocessing.icvl_dataset import ICVLDataset
from data_preprocessing.mrsa_dataset import MRSADataset
from model.environment import HandEnv
from model.ac_model import Pretrain
import os
import tensorflow as tf
import utils
import numpy as np
import time
import random
from tensorboardX import SummaryWriter
import multiprocessing
import pickle
import glob


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


def collect_one_sample(idx, example_volume, example_pose, W, H, D, ac_dim, chain_idx):
    volume = example_volume[np.newaxis, :, :, :, np.newaxis]  # [1,W,H,D,C]
    assert volume.shape == (1, W, H, D, 1)
    volume = np.transpose(volume, [0, 3, 2, 1, 4])  # 1DHWC
    # label_lie_algebra = utils.pose_to_lie_algebras(example_pose, chain_idx)[np.newaxis, :]
    label = example_pose.reshape([-1])[np.newaxis, :]
    assert label.shape == (1, ac_dim)
    return volume, label, idx


def collect_train_samples(env, dataset, num_files, num_samples, num_cpus, ac_dim):
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
                                        (0, example[9], example[6], W, H, D, ac_dim, env.chains_idx)))
    pool.close()
    pool.join()
    pool.terminate()
    for result in results:
        volume, label, _ = result.get()
        train_samples.append(volume)
        train_label.append(label)
    train_samples = np.concatenate(train_samples, axis=0)
    train_label = np.concatenate(train_label, axis=0)
    return train_samples, train_label


def collect_test_samples(env, dataset, num_cpus, ac_dim, for_train=True):
    print('Loading test samples...')
    W, H, D = dataset.predefined_bbx[0] + 1, dataset.predefined_bbx[1] + 1, dataset.predefined_bbx[2] + 1
    test_samples = []
    test_label = []
    examples = []
    for data in dataset.get_samples_testing():
        examples += data

    results = []
    pool = multiprocessing.Pool(num_cpus)
    for idx, example in enumerate(examples):
        # example (filename, xyz_pose, depth_img, pose_bbx, cropped_point,
        #          coeff, normalized_rotate_pose, normalized_rotate_points, rotated_bbx, volume)
        results.append(pool.apply_async(collect_one_sample,
                                        (idx, example[9], example[6], W, H, D, ac_dim, env.chains_idx)))
    pool.close()
    pool.join()

    other_data = []
    for result in results:
        volume, label, i = result.get()
        test_samples.append(volume)
        test_label.append(label)
        if not for_train:
            # collect (filename, pose_bbx, coeff, rotated_bbx)
            tmp = examples[i]
            other_data.append((tmp[0], tmp[3], tmp[5], tmp[8]))
    test_samples = np.concatenate(test_samples, axis=0)
    test_label = np.concatenate(test_label, axis=0)
    return test_samples, test_label, other_data


def pre_train(config):
    if config['dataset'] == 'nyu':
        dataset = NYUDataset(subset='training', root_dir='/hand_pose_data/nyu/', predefined_bbx=(63, 63, 31))
        ac_dim = 3 * dataset.jnt_num
        weights = np.ones([1, dataset.jnt_num])
        weights[0, 13] = 2  # weight root joint error
        cnn_layer = (8, 16, 32, 64, 128)  # 512
        fc_layer = (512, 512, 256)
    elif config['dataset'] == 'icvl':
        dataset = ICVLDataset(subset='training', root_dir='/hand_pose_data/icvl/', predefined_bbx=(63, 63, 31))
        ac_dim = 3 * dataset.jnt_num
        weights = np.ones([1, dataset.jnt_num])
        weights[0, 0] = 2  # weight root joint error
        cnn_layer = (8, 16, 32, 64, 128)  # 512
        fc_layer = (512, 512, 256)
    elif config['dataset'] == 'mrsa15':
        # (180, 120, 70), 6 * 21 = 126
        dataset = MRSADataset(subset='training', test_fold=config['mrsa_test_fold'],
                              root_dir='/hand_pose_data/mrsa15/', predefined_bbx=(63, 63, 31))
        ac_dim = 3 * dataset.jnt_num
        weights = np.ones([1, dataset.jnt_num])
        weights[0, 0] = 2  # weight root joint error
        cnn_layer = (8, 16, 32, 64, 128)  # 512
        fc_layer = (512, 512, 256)
    else:
        raise ValueError('Dataset name %s error...' % config['dataset'])
    print('Loss Weights:', weights)
    obs_dims = (dataset.predefined_bbx[2] + 1, dataset.predefined_bbx[1] + 1, dataset.predefined_bbx[0] + 1, 1)
    env = HandEnv(dataset_name=config['dataset'],
                  subset='training',
                  max_iters=5,
                  predefined_bbx=dataset.predefined_bbx,
                  pretrained_model=None)
    scope = 'pre_train'
    batch_size = config['batch_size']

    # define model and loss
    model = Pretrain(scope, obs_dims, cnn_layer, fc_layer, ac_dim)  # model.obs, model.ac, model.dropout_prob
    tf_label = tf.placeholder(shape=(None, ac_dim), dtype=tf.float32, name='action')
    tf_weights = tf.placeholder(shape=(1, dataset.jnt_num), dtype=tf.float32, name='action')
    # average joint mse error
    tf_mse = tf.reduce_mean(tf_weights * tf.reduce_sum(
        tf.reshape(tf.square(model.ac - tf_label), [-1, int(ac_dim / 3), 3]), axis=2), axis=1)
    tf_loss = tf.reduce_mean(tf_mse)  # average over mini-batch
    tf_max_error = tf.sqrt(tf.reduce_max(tf.reduce_sum(
        tf.reshape(tf.square(model.ac - tf_label), [-1, int(ac_dim / 3), 3]), axis=2), axis=1))

    global_step = tf.Variable(0, trainable=False, name='step')
    lr = tf.train.exponential_decay(config['lr_start'], global_step,
                                    config['lr_decay_iters'], config['lr_decay_rate'])
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=tf_loss, global_step=global_step)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss=tf_loss, global_step=global_step)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        root_dir = config['saved_model_path'] + config['dataset'] + '/'
        writer = SummaryWriter(log_dir=root_dir)

        if config['dataset'] == 'mrsa15':
            model_save_dir = root_dir + config['dataset'] + '_' + config['mrsa_test_fold'] + '_pretrain.pkl'
        else:
            model_save_dir = root_dir + config['dataset'] + '_pretrain.pkl'
        if os.path.exists(model_save_dir) and not config['new_training']:
            utils.loadFromFlat(model.get_trainable_variables(), model_save_dir)
            print("Pre-train parameter loaded from %s" % model_save_dir)
        else:
            print("[Warning]: initialize the pre-train model")

        x_test, y_test, _ = collect_test_samples(env, dataset, config['num_cpus'], ac_dim)

        n_test = x_test.shape[0]
        print('test samples %i' % n_test)
        best_loss = 1000
        i = 0
        while i < config['n_rounds']:
            i += 1
            print('--------------------------------Round % i---------------------------------' % i)
            # test
            loss_list, max_error_list = [], []
            if i % config['test_gap'] == 1:
                start_time = time.time()
                for j in range(n_test // batch_size + 1):
                    idx1 = j * batch_size
                    idx2 = min((j + 1) * batch_size, n_test)
                    batch_loss, batch_max_error = sess.run([tf_mse, tf_max_error],
                                                           feed_dict={model.obs: x_test[idx1: idx2, ...],
                                                                      tf_label: y_test[idx1: idx2],
                                                                      model.dropout_prob: 1.0,
                                                                      tf_weights: weights})
                    loss_list.append(batch_loss)
                    max_error_list.append(batch_max_error)
                test_loss = np.mean(np.hstack(loss_list))
                max_error = np.hstack(max_error_list)
                writer.add_scalar(config['dataset'] + '_test_loss', test_loss, i)
                writer.add_histogram(config['dataset'] + '_max_error', max_error, i)
                end_time = time.time()
                print('>>> Testing loss: {:.4f}, best loss {:.4f}, mean_max_error {:.4f}， time used: {:.2f}s'
                      .format(test_loss, best_loss, np.mean(max_error), end_time - start_time))
                if best_loss > test_loss:
                    utils.saveToFlat(model.get_trainable_variables(), model_save_dir)
                    best_loss = test_loss.copy()
                    print('>>> Model saved... best loss {:.4f}'.format(best_loss))

            # train
            start_time = time.time()
            x_train, y_train = collect_train_samples(env, dataset,
                                                     config['files_per_time'],
                                                     config['samples_per_time'],
                                                     config['num_cpus'], ac_dim)
            print('Collected samples {}'.format(x_train.shape[0]))
            loss_list = []
            for _ in range(config['train_iters']):
                batch_idx = np.random.randint(0, x_train.shape[0], batch_size)
                _, batch_loss, step = sess.run([optimizer, tf_loss, global_step],
                                               feed_dict={model.obs: x_train[batch_idx, ...],
                                                          tf_label: y_train[batch_idx],
                                                          model.dropout_prob: 0.5,
                                                          tf_weights: weights})
                loss_list.append(batch_loss)
            end_time = time.time()
            writer.add_scalar(config['dataset'] + '_train_loss', np.mean(loss_list), i)
            print('Training loss: {:.4f}, time used: {:.2f}s, step: {:d}'
                  .format(np.mean(loss_list), end_time - start_time, step))
        writer.close()


"""
************************************************************************************************
     test on pre-train model and transfer the predicted pose to original coordinate system
************************************************************************************************
"""


def pre_test(config):
    if config['dataset'] == 'nyu':
        dataset = NYUDataset(subset='training', root_dir='/hand_pose_data/nyu/', predefined_bbx=(63, 63, 31))
        ac_dim = 3 * dataset.jnt_num
        # (160, 120, 70), 6 * 14 = 84
        cnn_layer = (8, 16, 32, 64, 128)  # 512
        fc_layer = (512, 512, 256)
    elif config['dataset'] == 'icvl':
        dataset = ICVLDataset(subset='training', root_dir='/hand_pose_data/icvl/', predefined_bbx=(63, 63, 31))
        ac_dim = 3 * dataset.jnt_num
        # (140, 120, 60), 6 * 16 = 96
        cnn_layer = (8, 16, 32, 64, 128)  # 512
        fc_layer = (512, 512, 256)
    elif config['dataset'] == 'mrsa15':
        # (180, 120, 70), 6 * 21 = 126
        dataset = MRSADataset(subset='training', test_fold=config['mrsa_test_fold'],
                              root_dir='/hand_pose_data/mrsa15/', predefined_bbx=(63, 63, 31))
        ac_dim = 3 * dataset.jnt_num
        cnn_layer = (8, 16, 32, 64, 128)  # 512
        fc_layer = (512, 512, 256)
    else:
        raise ValueError('Dataset name %s error...' % config['dataset'])
    obs_dims = (dataset.predefined_bbx[2] + 1, dataset.predefined_bbx[1] + 1, dataset.predefined_bbx[0] + 1, 1)
    env = HandEnv(dataset_name=config['dataset'],
                  subset='training',
                  max_iters=5,
                  predefined_bbx=dataset.predefined_bbx,
                  pretrained_model=None)
    scope = 'pre_train'
    batch_size = config['batch_size']
    tf.reset_default_graph()
    model = Pretrain(scope, obs_dims, cnn_layer, fc_layer, ac_dim)  # model.obs, model.ac, model.dropout_prob
    tf_label = tf.placeholder(shape=(None, ac_dim), dtype=tf.float32, name='action')
    tf_max_error = tf.sqrt(tf.reduce_max(tf.reduce_sum(
        tf.reshape(tf.square(model.ac - tf_label), [-1, int(ac_dim/3), 3]), axis=2), axis=1))

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        root_dir = config['saved_model_path'] + config['dataset'] + '/'

        if config['dataset'] == 'mrsa15':
            model_save_dir = root_dir + config['dataset'] + '_' + config['mrsa_test_fold'] + '_pretrain.pkl'
        else:
            model_save_dir = root_dir + config['dataset'] + '_pretrain.pkl'
        if os.path.exists(model_save_dir):
            utils.loadFromFlat(model.get_trainable_variables(), model_save_dir)
            print("Pre-train parameter loaded from %s" % model_save_dir)
        else:
            raise ValueError('Model not found from %s' % model_save_dir)

        x_test, y_test, other_data = collect_test_samples(env, dataset, config['num_cpus'], ac_dim, False)

        # # check function transfer_pretest_pose()
        # i = 0
        # filename, pred_pose = transfer_pretest_pose(y_test[i, :].reshape([int(ac_dim/3), 3]), other_data[i],
        #                                             dataset.jnt_num, env.chains_idx,
        #                                             dataset.predefined_bbx, dataset.camera_cfg, config['dataset'])
        # print(filename, '\n', pred_pose)

        n_test = x_test.shape[0]
        start_time = time.time()
        pred_list, max_error_list = [], []
        for j in range(n_test // batch_size + 1):
            idx1 = j * batch_size
            idx2 = min((j + 1) * batch_size, n_test)
            preds, max_error = sess.run([model.ac, tf_max_error],
                                        feed_dict={model.obs: x_test[idx1: idx2, ...],
                                                   model.dropout_prob: 1.0,
                                                   tf_label: y_test[idx1: idx2, ...]})
            pred_list.append(preds)
            max_error_list.append(max_error)
        pred_poses = np.concatenate(pred_list, axis=0).reshape([-1, int(ac_dim/3), 3])
        max_error_list = np.hstack(max_error_list)
        print('Prediction of initial pose is done...({} examples, mean_max_error {:.3f})'
              .format(pred_poses.shape[0], np.mean(max_error_list)))

        results = []
        pool = multiprocessing.Pool(config['num_cpus'])
        for i in range(pred_poses.shape[0]):
            results.append(pool.apply_async(transfer_pretest_pose,
                                            (pred_poses[i, :], other_data[i], dataset.jnt_num, env.chains_idx,
                                             dataset.predefined_bbx, dataset.camera_cfg, config['dataset'],)))
        pool.close()
        pool.join()
        preds = {}
        for result in results:
            _filename, pred_pose = result.get()
            if config['dataset'] == 'nyu':
                # depth_1_0000601.png, test/depth_1_0000601.png
                filename = 'test/' + _filename
            elif config['dataset'] == 'mrsa15':
                # /hand_pose_data/mrsa15/dataset/P8/I/000307_depth.bin, P8/I/000307_depth.bin
                filename = '/'.join(_filename.split('/')[-3:])
            elif config['dataset'] == 'icvl':
                # 'test_seq_1/image_0002.png'
                filename = _filename
            preds[filename] = pred_pose
        end_time = time.time()
        print('Pretest finished. ({:.2f}s)'.format(end_time - start_time))

        if config['dataset'] == 'mrsa15':
            # save predictions
            with open(root_dir + 'pre_test_results_%s.pkl' % config['mrsa_test_fold'], 'wb') as f:
                pickle.dump(preds, f)
                print('Predictions are saved as %s.' %
                      (root_dir + 'pre_test_results_%s.pkl' % config['mrsa_test_fold']))
        else:
            # save predictions
            with open(root_dir + 'pre_test_results.pkl', 'wb') as f:
                pickle.dump(preds, f)
                print('Predictions are saved as %s.' % root_dir + 'pre_test_results.pkl')
            # write text file
            write_text_file(preds, config['dataset'],
                            root_dir + config['dataset'] + '_pretrain_rl_pose_estimation.txt')


def write_text_file_mrsa15(results_dir='../../results/mrsa15/'):
    file_list = glob.glob(results_dir + 'pre_test_results*')
    preds = {}
    for file in file_list:
        with open(file, 'rb') as f:
            pred = pickle.load(f)
            preds.update(pred)
    write_text_file(preds, 'mrsa15', results_dir + 'mrsa15_pretrain_rl_pose_estimation.txt',
                    evaluation_dir='../evaluation/')


def write_text_file(preds, dataset_name, save_dir, evaluation_dir='../evaluation/'):
    if dataset_name == 'mrsa15':
        dir_file_test_list = evaluation_dir + 'groundtruth/msra/msra_test_list.txt'
    elif dataset_name == 'nyu':
        dir_file_test_list = evaluation_dir + 'groundtruth/nyu/nyu_test_list.txt'
    elif dataset_name == 'icvl':
        dir_file_test_list = evaluation_dir + 'groundtruth/icvl/icvl_test_list.txt'

    file_list = []
    with open(dir_file_test_list, 'r') as f:
        for line in f:
            filename = line.strip('\n')
            file_list.append(filename)

    with open(save_dir, 'w') as f:
        for filename in file_list:
            pose = preds[filename].reshape([-1])
            line = ' '.join(['{:.3f}'.format(coordinate) for coordinate in pose]) + '\n'
            f.write(line)
    print('text file is written as %s' % save_dir)


def transfer_pretest_pose(pred_pose, other_info, num_joints, chains_idx, predefined_bbx, camera_cfg, dataset_name):
    filename, pose_bbx, coeff, rotated_bbx = other_info
    # pred_pose, _ = utils.lie_algebras_to_pose(pred_algebra, num_joints, chains_idx)
    raw_pose = utils.transfer_pose(pred_pose, rotated_bbx, coeff, predefined_bbx, pose_bbx)
    if dataset_name == 'mrsa15':
        annotations = utils.xyz2uvd(raw_pose.reshape([-1]), camera_cfg)
    elif dataset_name == 'nyu':
        annotations = utils.xyz2uvd(raw_pose.reshape([-1]), camera_cfg)
    elif dataset_name == 'icvl':
        annotations = utils.xyz2uvd(raw_pose.reshape([-1]), camera_cfg)
    else:
        raise ValueError('Name of dataset errors %s.' % dataset_name)
    return filename, annotations


def get_config():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', '-mode', type=str, default='pretest')
    parser.add_argument('--gpu_id', '-id', type=str, default='0')
    parser.add_argument('--saved_model_path', '-smp', type=str, default='../../results/')
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--n_rounds', '-nr', type=int, default=2000)
    parser.add_argument('--train_iters', '-ni', type=int, default=100)
    parser.add_argument('--dataset', '-data', type=str, default='nyu')
    parser.add_argument('--mrsa_test_fold', '-mtf', type=str, default='P8')
    parser.add_argument('--files_per_time', '-fpt', type=int, default=10)
    parser.add_argument('--samples_per_time', '-spt', type=int, default=2000)
    parser.add_argument('--lr_start', '-lr', help='learning rate', type=float, default=0.0001)
    parser.add_argument('--lr_decay_rate', default=0.99)
    parser.add_argument('--lr_decay_iters', default=2000)
    parser.add_argument('--new_training', '-new', type=bool, default=1)
    parser.add_argument('--test_gap', '-tg', type=int, default=2)
    parser.add_argument('--num_cpus', '-cpus', type=int, default=32)
    args = vars(parser.parse_args())
    utils.print_args(args)
    return args


def main():
    config = get_config()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']
    if config['mode'] == 'pretrain':
        pre_train(config)
    elif config['mode'] == 'pretest':
        pre_test(config)
    elif config['mode'] == 'all':
        pre_train(config)
        pre_test(config)
    elif config['mode'] == 'write_mrsa15':
        write_text_file_mrsa15()
    else:
        raise ValueError('Args mode errors: %s' % config['mode'])


if __name__ == '__main__':
    main()

