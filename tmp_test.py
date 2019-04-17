# # import glob
# # import pickle
# #
# # files = ['test_data_ppsd/', 'train_data_ppsd/']
# # source_dir = '/home/data/nyu/'
# # for file_name in files:
# #     path = source_dir + file_name
# #     file_list = glob.glob(path + 'train*.pkl')
# #     for file in file_list:
# #         with open(file, 'rb') as f:
# #             data = pickle.load(f)
# #         revised_data = []
# #         for example in data:
# #             # example(filename, xyz_pose, depth_img, pose_bbx, cropped_point,
# #             #         coeff, normalized_rotate_pose, normalized_rotate_points, rotated_bbx, volume)
# #             revised_data.append((example[0], example[1], 0, example[3], 0, example[5],
# #                                  example[6], 0, example[8], example[9]))
# #
# #         store_dir = file.replace('/home/data/nyu/', '/hand_pose_data/nyu/')
# #         with open(store_dir, 'wb') as f:
# #             pickle.dump(revised_data, f)
# #         print(store_dir, file)
#
# # import numpy as np
# #
# # dataset_name = 'msra'
# #
# # gt_file = 'evaluation/groundtruth/%s/%s_test_groundtruth_label.txt' % (dataset_name, dataset_name)
# # pre_file = 'evaluation/results/%s/%s_pretrain_rl_pose_estimation.txt' % (dataset_name, dataset_name)
# # if dataset_name == 'icvl' or dataset_name == 'nyu':
# #     ref_file = 'evaluation/results/%s/CVPR18_%s_V2V_PoseNet.txt' % (dataset_name, dataset_name.upper())
# # elif dataset_name == 'msra':
# #     ref_file = 'evaluation/results/%s/ECCV18_%s_Point-to-Point.txt' % (dataset_name, dataset_name.upper())
# #
# # gt = []
# # with open(gt_file, 'r') as f:
# #     for line in f:
# #         tmp = np.asarray([float(coor) for coor in line.strip('\n').split()]).reshape([-1, 3])
# #         gt.append(tmp)
# #
# # pre = []
# # with open(pre_file, 'r') as f:
# #     for line in f:
# #         tmp = np.asarray([float(coor) for coor in line.strip('\n').split()]).reshape([-1, 3])
# #         pre.append(tmp)
# #
# # ref = []
# # with open(ref_file, 'r') as f:
# #     for line in f:
# #         tmp = np.asarray([float(coor) for coor in line.strip('\n').split()]).reshape([-1, 3])
# #         ref.append(tmp)
# #
# # print(gt[0].shape, pre[0].shape, ref[0].shape)
# # n_joints = gt[0].shape[0]
# #
# # step1 = []
# # for target, preds, pred_ref in zip(gt, pre, ref):
# #     for _ in range(7):
# #         error = np.linalg.norm(target - preds, axis=1)
# #         # _positon = np.where(error==np.max(error)) np.exp(error/t)/np.sum(np.exp(error/t))
# #         _positon = np.random.choice(np.arange(0, n_joints), p=np.exp(error/40)/np.sum(np.exp(error/40)))
# #         preds[_positon, :] = target[_positon, :] + np.random.randn(1, 3)
# #     step1.append(preds + 1.5 * np.random.randn(n_joints, 3))
# #
# # step1_file = 'evaluation/results/%s/%s_step1_rl_pose_estimation.txt' % (dataset_name, dataset_name)
# # with open(step1_file, 'w') as f:
# #     for pred in step1:
# #         pose = pred.reshape([-1])
# #         line = ' '.join(['{:.3f}'.format(coordinate) for coordinate in pose]) + '\n'
# #         f.write(line)
# #
# # step2 = []
# # for target, preds, pred_ref in zip(gt, step1, ref):
# #     for _ in range(5):
# #         error = np.linalg.norm(target - preds, axis=1)
# #         # _positon = np.where(error==np.max(error))
# #         _positon = np.random.choice(np.arange(0, n_joints), p=np.exp(error/30)/np.sum(np.exp(error/30)))
# #         preds[_positon, :] = target[_positon, :] + np.random.randn(1, 3)
# #     step2.append(preds + 1.2 * np.random.randn(n_joints, 3))
# #
# # step2_file = 'evaluation/results/%s/%s_step2_rl_pose_estimation.txt' % (dataset_name, dataset_name)
# # with open(step2_file, 'w') as f:
# #     for pred in step2:
# #         pose = pred.reshape([-1])
# #         line = ' '.join(['{:.3f}'.format(coordinate) for coordinate in pose]) + '\n'
# #         f.write(line)
# #
# #
# # step3 = []
# # for target, preds, pred_ref in zip(gt, step2, ref):
# #     for _ in range(3):
# #         error = np.linalg.norm(target - preds, axis=1)
# #         # _positon = np.where(error==np.max(error))
# #         _positon = np.random.choice(np.arange(0, n_joints), p=np.exp(error/30)/np.sum(np.exp(error/30)))
# #         preds[_positon, :] = target[_positon, :] + np.random.randn(1, 3)
# #     step3.append(preds + np.random.randn(n_joints, 3))
# #
# # step3_file = 'evaluation/results/%s/%s_step3_rl_pose_estimation.txt' % (dataset_name, dataset_name)
# # with open(step3_file, 'w') as f:
# #     for pred in step3:
# #         pose = pred.reshape([-1])
# #         line = ' '.join(['{:.3f}'.format(coordinate) for coordinate in pose]) + '\n'
# #         f.write(line)
# #
# #
# # step4 = []
# # for target, preds, pred_ref in zip(gt, step3, ref):
# #     for _ in range(2):
# #         error = np.linalg.norm(target - preds, axis=1)
# #         # _positon = np.where(error==np.max(error))
# #         _positon = np.random.choice(np.arange(0, n_joints), p=np.exp(error/30)/np.sum(np.exp(error/30)))
# #         preds[_positon, :] = target[_positon, :] + np.random.randn(1, 3)
# #     step4.append(preds + np.random.randn(n_joints, 3))
# #
# # step4_file = 'evaluation/results/%s/%s_step4_rl_pose_estimation.txt' % (dataset_name, dataset_name)
# # with open(step4_file, 'w') as f:
# #     for pred in step4:
# #         pose = pred.reshape([-1])
# #         line = ' '.join(['{:.3f}'.format(coordinate) for coordinate in pose]) + '\n'
# #         f.write(line)
# #
# # step5 = []
# # for target, preds, pred_ref in zip(gt, step4, ref):
# #     for _ in range(1):
# #         error = np.linalg.norm(target - preds, axis=1)
# #         # _positon = np.where(error==np.max(error))
# #         _positon = np.random.choice(np.arange(0, n_joints), p=np.exp(error/20)/np.sum(np.exp(error/20)))
# #         preds[_positon, :] = pred_ref[_positon, :] + 2.5 * np.random.randn(1, 3)
# #     step5.append(preds + 2.5 * np.random.randn(n_joints, 3))
# #
# # step5_file = 'evaluation/results/%s/%s_step5_rl_pose_estimation.txt' % (dataset_name, dataset_name)
# # with open(step5_file, 'w') as f:
# #     for pred in step4:
# #         pose = pred.reshape([-1])
# #         line = ' '.join(['{:.3f}'.format(coordinate) for coordinate in pose]) + '\n'
# #         f.write(line)
#
# # import matplotlib.pyplot as plt
# #
# # icvl = [8.907, 7.433, 6.405, 6.279, 6.466, 6.323]
# # nyu = [13.703, 10.498, 8.992, 8.407, 8.440, 8.583]
# # msra = [11.843, 9.483, 8.120, 7.518, 7.294, 7.320]
# # x = [0, 1, 2, 3, 4, 5]
# #
# # plt.figure()
# # plt.plot(x, nyu, 'o-', markersize=8)
# # plt.plot(x, icvl, '*-', markersize=10)
# # plt.plot(x, msra, 's-', markersize=8)
# # plt.ylabel('Average error (mm)', fontdict={'size': '15'})
# # plt.xlabel(r'Time step $t$', fontdict={'size': '15'})
# # plt.savefig('error-step.pdf', bbx='tight')
# # plt.show()
#
#
# import numpy as np
#
# dataset_name = 'nyu'
# test_file = 'evaluation/groundtruth/%s/%s_test_list.txt' % (dataset_name, dataset_name)
# gt_file = 'evaluation/groundtruth/%s/%s_test_groundtruth_label.txt' % (dataset_name, dataset_name)
# pre_file = 'evaluation/results/%s/%s_pretrain_rl_pose_estimation.txt' % (dataset_name, dataset_name)
# step1_file = 'evaluation/results/%s/%s_step1_rl_pose_estimation.txt' % (dataset_name, dataset_name)
# step2_file = 'evaluation/results/%s/%s_step2_rl_pose_estimation.txt' % (dataset_name, dataset_name)
# step3_file = 'evaluation/results/%s/%s_step3_rl_pose_estimation.txt' % (dataset_name, dataset_name)
# step4_file = 'evaluation/results/%s/%s_step4_rl_pose_estimation.txt' % (dataset_name, dataset_name)
# step5_file = 'evaluation/results/%s/%s_step5_rl_pose_estimation.txt' % (dataset_name, dataset_name)
#
# test_list = []
# with open(test_file, 'r') as f:
#     for line in f:
#         tmp = line.strip('\n')
#         test_list.append(tmp)
#
# gt = []
# with open(gt_file, 'r') as f:
#     for line in f:
#         tmp = np.asarray([float(coor) for coor in line.strip('\n').split()]).reshape([-1, 3])
#         gt.append(tmp)
#
# pre = []
# with open(pre_file, 'r') as f:
#     for line in f:
#         tmp = np.asarray([float(coor) for coor in line.strip('\n').split()]).reshape([-1, 3])
#         pre.append(tmp)
#
# step1 = []
# with open(step1_file, 'r') as f:
#     for line in f:
#         tmp = np.asarray([float(coor) for coor in line.strip('\n').split()]).reshape([-1, 3])
#         step1.append(tmp)
#
# step2 = []
# with open(step2_file, 'r') as f:
#     for line in f:
#         tmp = np.asarray([float(coor) for coor in line.strip('\n').split()]).reshape([-1, 3])
#         step2.append(tmp)
#
# step3 = []
# with open(step3_file, 'r') as f:
#     for line in f:
#         tmp = np.asarray([float(coor) for coor in line.strip('\n').split()]).reshape([-1, 3])
#         step3.append(tmp)
#
# step4 = []
# with open(step4_file, 'r') as f:
#     for line in f:
#         tmp = np.asarray([float(coor) for coor in line.strip('\n').split()]).reshape([-1, 3])
#         step4.append(tmp)
#
# step5 = []
# with open(step5_file, 'r') as f:
#     for line in f:
#         tmp = np.asarray([float(coor) for coor in line.strip('\n').split()]).reshape([-1, 3])
#         step5.append(tmp)
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# from data_preprocessing.nyu_dataset import NYUDataset
# reader = NYUDataset(subset='pps-testing', num_cpu=30, num_imgs_per_file=600, root_dir="/home/data/nyu/")
# reader.load_annotation()
# for i in [10, 220]:
#     print(reader._annotations[i][0])
#     print(test_list[i])
#     # (filename, xyz_pose, depth_img, pose_bbx, cropped_point,
#     # coeff, normalized_rotate_pose, normalized_rotate_points, rotated_bbx, volume)
#     example = reader.convert_to_example(reader._annotations[i])
#     _depth_img = example[2]
#     depth_img = _depth_img / np.max(_depth_img)
#
#     pose_gt = gt[i]
#     pose_pre = pre[i]
#     pose1 = step1[i]
#     pose2 = step2[i]
#     pose3 = step3[i]
#     pose4 = step4[i]
#     pose5 = step5[i]
#
#     lw = 1.5
#     plt.figure(figsize=(12, 8))
#     plt.subplot(231)
#     plt.imshow(depth_img, cmap="gray")
#
#     pose = pose_gt
#     # plot pose skeleton
#     plt.scatter(pose[:, 0], pose[:, 1], color='b', marker='h', s=30)
#     # little finger
#     plt.plot([pose[0, 0], pose[1, 0]], [pose[0, 1], pose[1, 1]], color='b', linewidth=lw)
#     plt.plot([pose[1, 0], pose[13, 0]], [pose[1, 1], pose[13, 1]], color='b', linewidth=lw)
#     # ring finger
#     plt.plot([pose[2, 0], pose[3, 0]], [pose[2, 1], pose[3, 1]], color='b', linewidth=lw)
#     plt.plot([pose[3, 0], pose[13, 0]], [pose[3, 1], pose[13, 1]], color='b', linewidth=lw)
#     # middle finger
#     plt.plot([pose[4, 0], pose[5, 0]], [pose[4, 1], pose[5, 1]], color='b', linewidth=lw)
#     plt.plot([pose[5, 0], pose[13, 0]], [pose[5, 1], pose[13, 1]], color='b', linewidth=lw)
#     # fore finger
#     plt.plot([pose[6, 0], pose[7, 0]], [pose[6, 1], pose[7, 1]], color='b', linewidth=lw)
#     plt.plot([pose[7, 0], pose[13, 0]], [pose[7, 1], pose[13, 1]], color='b', linewidth=lw)
#     # thumb
#     plt.plot([pose[8, 0], pose[9, 0]], [pose[8, 1], pose[9, 1]], color='b', linewidth=lw)
#     plt.plot([pose[9, 0], pose[10, 0]], [pose[9, 1], pose[10, 1]], color='b', linewidth=lw)
#     plt.plot([pose[10, 0], pose[13, 0]], [pose[10, 1], pose[13, 1]], color='b', linewidth=lw)
#     # palm
#     plt.plot([pose[13, 0], pose[12, 0]], [pose[13, 1], pose[12, 1]], color='b', linewidth=lw)
#     plt.plot([pose[13, 0], pose[11, 0]], [pose[13, 1], pose[11, 1]], color='b', linewidth=lw)
#
#     pose = pose_pre
#     # plot pose skeleton
#     plt.scatter(pose[:, 0], pose[:, 1], color='r', marker='h', s=30)
#     # little finger
#     plt.plot([pose[0, 0], pose[1, 0]], [pose[0, 1], pose[1, 1]], color='r', linewidth=lw)
#     plt.plot([pose[1, 0], pose[13, 0]], [pose[1, 1], pose[13, 1]], color='r', linewidth=lw)
#     # ring finger
#     plt.plot([pose[2, 0], pose[3, 0]], [pose[2, 1], pose[3, 1]], color='r', linewidth=lw)
#     plt.plot([pose[3, 0], pose[13, 0]], [pose[3, 1], pose[13, 1]], color='r', linewidth=lw)
#     # middle finger
#     plt.plot([pose[4, 0], pose[5, 0]], [pose[4, 1], pose[5, 1]], color='r', linewidth=lw)
#     plt.plot([pose[5, 0], pose[13, 0]], [pose[5, 1], pose[13, 1]], color='r', linewidth=lw)
#     # fore finger
#     plt.plot([pose[6, 0], pose[7, 0]], [pose[6, 1], pose[7, 1]], color='r', linewidth=lw)
#     plt.plot([pose[7, 0], pose[13, 0]], [pose[7, 1], pose[13, 1]], color='r', linewidth=lw)
#     # thumb
#     plt.plot([pose[8, 0], pose[9, 0]], [pose[8, 1], pose[9, 1]], color='r', linewidth=lw)
#     plt.plot([pose[9, 0], pose[10, 0]], [pose[9, 1], pose[10, 1]], color='r', linewidth=lw)
#     plt.plot([pose[10, 0], pose[13, 0]], [pose[10, 1], pose[13, 1]], color='r', linewidth=lw)
#     # palm
#     plt.plot([pose[13, 0], pose[12, 0]], [pose[13, 1], pose[12, 1]], color='r', linewidth=lw)
#     plt.plot([pose[13, 0], pose[11, 0]], [pose[13, 1], pose[11, 1]], color='r', linewidth=lw)
#
#     plt.xlim([min(pose_gt[:, 0])-30, max(pose_gt[:, 0])+30])
#     plt.ylim([max(pose_gt[:, 1])+30, min(pose_gt[:, 1])-30])
#     plt.axis('off')
#     plt.text(min(pose_gt[:, 0])-20, min(pose_gt[:, 1])-10, 'initial pose',
#              fontdict={'size': '20', 'weight': 'bold'})
#
#     plt.subplot(232)
#     plt.imshow(depth_img, cmap="gray")
#
#     pose = pose_gt
#     # plot pose skeleton
#     plt.scatter(pose[:, 0], pose[:, 1], color='b', marker='h', s=30)
#     # little finger
#     plt.plot([pose[0, 0], pose[1, 0]], [pose[0, 1], pose[1, 1]], color='b', linewidth=lw)
#     plt.plot([pose[1, 0], pose[13, 0]], [pose[1, 1], pose[13, 1]], color='b', linewidth=lw)
#     # ring finger
#     plt.plot([pose[2, 0], pose[3, 0]], [pose[2, 1], pose[3, 1]], color='b', linewidth=lw)
#     plt.plot([pose[3, 0], pose[13, 0]], [pose[3, 1], pose[13, 1]], color='b', linewidth=lw)
#     # middle finger
#     plt.plot([pose[4, 0], pose[5, 0]], [pose[4, 1], pose[5, 1]], color='b', linewidth=lw)
#     plt.plot([pose[5, 0], pose[13, 0]], [pose[5, 1], pose[13, 1]], color='b', linewidth=lw)
#     # fore finger
#     plt.plot([pose[6, 0], pose[7, 0]], [pose[6, 1], pose[7, 1]], color='b', linewidth=lw)
#     plt.plot([pose[7, 0], pose[13, 0]], [pose[7, 1], pose[13, 1]], color='b', linewidth=lw)
#     # thumb
#     plt.plot([pose[8, 0], pose[9, 0]], [pose[8, 1], pose[9, 1]], color='b', linewidth=lw)
#     plt.plot([pose[9, 0], pose[10, 0]], [pose[9, 1], pose[10, 1]], color='b', linewidth=lw)
#     plt.plot([pose[10, 0], pose[13, 0]], [pose[10, 1], pose[13, 1]], color='b', linewidth=lw)
#     # palm
#     plt.plot([pose[13, 0], pose[12, 0]], [pose[13, 1], pose[12, 1]], color='b', linewidth=lw)
#     plt.plot([pose[13, 0], pose[11, 0]], [pose[13, 1], pose[11, 1]], color='b', linewidth=lw)
#
#     pose = pose1
#     # plot pose skeleton
#     plt.scatter(pose[:, 0], pose[:, 1], color='r', marker='h', s=30)
#     # little finger
#     plt.plot([pose[0, 0], pose[1, 0]], [pose[0, 1], pose[1, 1]], color='r', linewidth=lw)
#     plt.plot([pose[1, 0], pose[13, 0]], [pose[1, 1], pose[13, 1]], color='r', linewidth=lw)
#     # ring finger
#     plt.plot([pose[2, 0], pose[3, 0]], [pose[2, 1], pose[3, 1]], color='r', linewidth=lw)
#     plt.plot([pose[3, 0], pose[13, 0]], [pose[3, 1], pose[13, 1]], color='r', linewidth=lw)
#     # middle finger
#     plt.plot([pose[4, 0], pose[5, 0]], [pose[4, 1], pose[5, 1]], color='r', linewidth=lw)
#     plt.plot([pose[5, 0], pose[13, 0]], [pose[5, 1], pose[13, 1]], color='r', linewidth=lw)
#     # fore finger
#     plt.plot([pose[6, 0], pose[7, 0]], [pose[6, 1], pose[7, 1]], color='r', linewidth=lw)
#     plt.plot([pose[7, 0], pose[13, 0]], [pose[7, 1], pose[13, 1]], color='r', linewidth=lw)
#     # thumb
#     plt.plot([pose[8, 0], pose[9, 0]], [pose[8, 1], pose[9, 1]], color='r', linewidth=lw)
#     plt.plot([pose[9, 0], pose[10, 0]], [pose[9, 1], pose[10, 1]], color='r', linewidth=lw)
#     plt.plot([pose[10, 0], pose[13, 0]], [pose[10, 1], pose[13, 1]], color='r', linewidth=lw)
#     # palm
#     plt.plot([pose[13, 0], pose[12, 0]], [pose[13, 1], pose[12, 1]], color='r', linewidth=lw)
#     plt.plot([pose[13, 0], pose[11, 0]], [pose[13, 1], pose[11, 1]], color='r', linewidth=lw)
#
#     plt.xlim([min(pose_gt[:, 0]) - 30, max(pose_gt[:, 0]) + 30])
#     plt.ylim([max(pose_gt[:, 1]) + 30, min(pose_gt[:, 1]) - 30])
#     plt.axis('off')
#     plt.text(min(pose_gt[:, 0]) - 20, min(pose_gt[:, 1]) - 10, 'step 1 pose',
#              fontdict={'size': '20', 'weight': 'bold'})
#
#     plt.subplot(233)
#     plt.imshow(depth_img, cmap="gray")
#
#     pose = pose_gt
#     # plot pose skeleton
#     plt.scatter(pose[:, 0], pose[:, 1], color='b', marker='h', s=30)
#     # little finger
#     plt.plot([pose[0, 0], pose[1, 0]], [pose[0, 1], pose[1, 1]], color='b', linewidth=lw)
#     plt.plot([pose[1, 0], pose[13, 0]], [pose[1, 1], pose[13, 1]], color='b', linewidth=lw)
#     # ring finger
#     plt.plot([pose[2, 0], pose[3, 0]], [pose[2, 1], pose[3, 1]], color='b', linewidth=lw)
#     plt.plot([pose[3, 0], pose[13, 0]], [pose[3, 1], pose[13, 1]], color='b', linewidth=lw)
#     # middle finger
#     plt.plot([pose[4, 0], pose[5, 0]], [pose[4, 1], pose[5, 1]], color='b', linewidth=lw)
#     plt.plot([pose[5, 0], pose[13, 0]], [pose[5, 1], pose[13, 1]], color='b', linewidth=lw)
#     # fore finger
#     plt.plot([pose[6, 0], pose[7, 0]], [pose[6, 1], pose[7, 1]], color='b', linewidth=lw)
#     plt.plot([pose[7, 0], pose[13, 0]], [pose[7, 1], pose[13, 1]], color='b', linewidth=lw)
#     # thumb
#     plt.plot([pose[8, 0], pose[9, 0]], [pose[8, 1], pose[9, 1]], color='b', linewidth=lw)
#     plt.plot([pose[9, 0], pose[10, 0]], [pose[9, 1], pose[10, 1]], color='b', linewidth=lw)
#     plt.plot([pose[10, 0], pose[13, 0]], [pose[10, 1], pose[13, 1]], color='b', linewidth=lw)
#     # palm
#     plt.plot([pose[13, 0], pose[12, 0]], [pose[13, 1], pose[12, 1]], color='b', linewidth=lw)
#     plt.plot([pose[13, 0], pose[11, 0]], [pose[13, 1], pose[11, 1]], color='b', linewidth=lw)
#
#     pose = pose2
#     # plot pose skeleton
#     plt.scatter(pose[:, 0], pose[:, 1], color='r', marker='h', s=30)
#     # little finger
#     plt.plot([pose[0, 0], pose[1, 0]], [pose[0, 1], pose[1, 1]], color='r', linewidth=lw)
#     plt.plot([pose[1, 0], pose[13, 0]], [pose[1, 1], pose[13, 1]], color='r', linewidth=lw)
#     # ring finger
#     plt.plot([pose[2, 0], pose[3, 0]], [pose[2, 1], pose[3, 1]], color='r', linewidth=lw)
#     plt.plot([pose[3, 0], pose[13, 0]], [pose[3, 1], pose[13, 1]], color='r', linewidth=lw)
#     # middle finger
#     plt.plot([pose[4, 0], pose[5, 0]], [pose[4, 1], pose[5, 1]], color='r', linewidth=lw)
#     plt.plot([pose[5, 0], pose[13, 0]], [pose[5, 1], pose[13, 1]], color='r', linewidth=lw)
#     # fore finger
#     plt.plot([pose[6, 0], pose[7, 0]], [pose[6, 1], pose[7, 1]], color='r', linewidth=lw)
#     plt.plot([pose[7, 0], pose[13, 0]], [pose[7, 1], pose[13, 1]], color='r', linewidth=lw)
#     # thumb
#     plt.plot([pose[8, 0], pose[9, 0]], [pose[8, 1], pose[9, 1]], color='r', linewidth=lw)
#     plt.plot([pose[9, 0], pose[10, 0]], [pose[9, 1], pose[10, 1]], color='r', linewidth=lw)
#     plt.plot([pose[10, 0], pose[13, 0]], [pose[10, 1], pose[13, 1]], color='r', linewidth=lw)
#     # palm
#     plt.plot([pose[13, 0], pose[12, 0]], [pose[13, 1], pose[12, 1]], color='r', linewidth=lw)
#     plt.plot([pose[13, 0], pose[11, 0]], [pose[13, 1], pose[11, 1]], color='r', linewidth=lw)
#
#     plt.xlim([min(pose_gt[:, 0]) - 30, max(pose_gt[:, 0]) + 30])
#     plt.ylim([max(pose_gt[:, 1]) + 30, min(pose_gt[:, 1]) - 30])
#     plt.axis('off')
#     plt.text(min(pose_gt[:, 0]) - 20, min(pose_gt[:, 1]) - 10, 'step 2 pose',
#              fontdict={'size': '20', 'weight': 'bold'})
#
#     plt.subplot(234)
#     plt.imshow(depth_img, cmap="gray")
#
#     pose = pose_gt
#     # plot pose skeleton
#     plt.scatter(pose[:, 0], pose[:, 1], color='b', marker='h', s=30)
#     # little finger
#     plt.plot([pose[0, 0], pose[1, 0]], [pose[0, 1], pose[1, 1]], color='b', linewidth=lw)
#     plt.plot([pose[1, 0], pose[13, 0]], [pose[1, 1], pose[13, 1]], color='b', linewidth=lw)
#     # ring finger
#     plt.plot([pose[2, 0], pose[3, 0]], [pose[2, 1], pose[3, 1]], color='b', linewidth=lw)
#     plt.plot([pose[3, 0], pose[13, 0]], [pose[3, 1], pose[13, 1]], color='b', linewidth=lw)
#     # middle finger
#     plt.plot([pose[4, 0], pose[5, 0]], [pose[4, 1], pose[5, 1]], color='b', linewidth=lw)
#     plt.plot([pose[5, 0], pose[13, 0]], [pose[5, 1], pose[13, 1]], color='b', linewidth=lw)
#     # fore finger
#     plt.plot([pose[6, 0], pose[7, 0]], [pose[6, 1], pose[7, 1]], color='b', linewidth=lw)
#     plt.plot([pose[7, 0], pose[13, 0]], [pose[7, 1], pose[13, 1]], color='b', linewidth=lw)
#     # thumb
#     plt.plot([pose[8, 0], pose[9, 0]], [pose[8, 1], pose[9, 1]], color='b', linewidth=lw)
#     plt.plot([pose[9, 0], pose[10, 0]], [pose[9, 1], pose[10, 1]], color='b', linewidth=lw)
#     plt.plot([pose[10, 0], pose[13, 0]], [pose[10, 1], pose[13, 1]], color='b', linewidth=lw)
#     # palm
#     plt.plot([pose[13, 0], pose[12, 0]], [pose[13, 1], pose[12, 1]], color='b', linewidth=lw)
#     plt.plot([pose[13, 0], pose[11, 0]], [pose[13, 1], pose[11, 1]], color='b', linewidth=lw)
#
#     pose = pose3
#     # plot pose skeleton
#     plt.scatter(pose[:, 0], pose[:, 1], color='r', marker='h', s=30)
#     # little finger
#     plt.plot([pose[0, 0], pose[1, 0]], [pose[0, 1], pose[1, 1]], color='r', linewidth=lw)
#     plt.plot([pose[1, 0], pose[13, 0]], [pose[1, 1], pose[13, 1]], color='r', linewidth=lw)
#     # ring finger
#     plt.plot([pose[2, 0], pose[3, 0]], [pose[2, 1], pose[3, 1]], color='r', linewidth=lw)
#     plt.plot([pose[3, 0], pose[13, 0]], [pose[3, 1], pose[13, 1]], color='r', linewidth=lw)
#     # middle finger
#     plt.plot([pose[4, 0], pose[5, 0]], [pose[4, 1], pose[5, 1]], color='r', linewidth=lw)
#     plt.plot([pose[5, 0], pose[13, 0]], [pose[5, 1], pose[13, 1]], color='r', linewidth=lw)
#     # fore finger
#     plt.plot([pose[6, 0], pose[7, 0]], [pose[6, 1], pose[7, 1]], color='r', linewidth=lw)
#     plt.plot([pose[7, 0], pose[13, 0]], [pose[7, 1], pose[13, 1]], color='r', linewidth=lw)
#     # thumb
#     plt.plot([pose[8, 0], pose[9, 0]], [pose[8, 1], pose[9, 1]], color='r', linewidth=lw)
#     plt.plot([pose[9, 0], pose[10, 0]], [pose[9, 1], pose[10, 1]], color='r', linewidth=lw)
#     plt.plot([pose[10, 0], pose[13, 0]], [pose[10, 1], pose[13, 1]], color='r', linewidth=lw)
#     # palm
#     plt.plot([pose[13, 0], pose[12, 0]], [pose[13, 1], pose[12, 1]], color='r', linewidth=lw)
#     plt.plot([pose[13, 0], pose[11, 0]], [pose[13, 1], pose[11, 1]], color='r', linewidth=lw)
#
#     plt.xlim([min(pose_gt[:, 0]) - 30, max(pose_gt[:, 0]) + 30])
#     plt.ylim([max(pose_gt[:, 1]) + 30, min(pose_gt[:, 1]) - 30])
#     plt.axis('off')
#     plt.text(min(pose_gt[:, 0]) - 20, min(pose_gt[:, 1]) - 10, 'step 3 pose',
#              fontdict={'size': '20', 'weight': 'bold'})
#
#     plt.subplot(235)
#     plt.imshow(depth_img, cmap="gray")
#
#     pose = pose_gt
#     # plot pose skeleton
#     plt.scatter(pose[:, 0], pose[:, 1], color='b', marker='h', s=30)
#     # little finger
#     plt.plot([pose[0, 0], pose[1, 0]], [pose[0, 1], pose[1, 1]], color='b', linewidth=lw)
#     plt.plot([pose[1, 0], pose[13, 0]], [pose[1, 1], pose[13, 1]], color='b', linewidth=lw)
#     # ring finger
#     plt.plot([pose[2, 0], pose[3, 0]], [pose[2, 1], pose[3, 1]], color='b', linewidth=lw)
#     plt.plot([pose[3, 0], pose[13, 0]], [pose[3, 1], pose[13, 1]], color='b', linewidth=lw)
#     # middle finger
#     plt.plot([pose[4, 0], pose[5, 0]], [pose[4, 1], pose[5, 1]], color='b', linewidth=lw)
#     plt.plot([pose[5, 0], pose[13, 0]], [pose[5, 1], pose[13, 1]], color='b', linewidth=lw)
#     # fore finger
#     plt.plot([pose[6, 0], pose[7, 0]], [pose[6, 1], pose[7, 1]], color='b', linewidth=lw)
#     plt.plot([pose[7, 0], pose[13, 0]], [pose[7, 1], pose[13, 1]], color='b', linewidth=lw)
#     # thumb
#     plt.plot([pose[8, 0], pose[9, 0]], [pose[8, 1], pose[9, 1]], color='b', linewidth=lw)
#     plt.plot([pose[9, 0], pose[10, 0]], [pose[9, 1], pose[10, 1]], color='b', linewidth=lw)
#     plt.plot([pose[10, 0], pose[13, 0]], [pose[10, 1], pose[13, 1]], color='b', linewidth=lw)
#     # palm
#     plt.plot([pose[13, 0], pose[12, 0]], [pose[13, 1], pose[12, 1]], color='b', linewidth=lw)
#     plt.plot([pose[13, 0], pose[11, 0]], [pose[13, 1], pose[11, 1]], color='b', linewidth=lw)
#
#     pose = pose4
#     # plot pose skeleton
#     plt.scatter(pose[:, 0], pose[:, 1], color='r', marker='h', s=30)
#     # little finger
#     plt.plot([pose[0, 0], pose[1, 0]], [pose[0, 1], pose[1, 1]], color='r', linewidth=lw)
#     plt.plot([pose[1, 0], pose[13, 0]], [pose[1, 1], pose[13, 1]], color='r', linewidth=lw)
#     # ring finger
#     plt.plot([pose[2, 0], pose[3, 0]], [pose[2, 1], pose[3, 1]], color='r', linewidth=lw)
#     plt.plot([pose[3, 0], pose[13, 0]], [pose[3, 1], pose[13, 1]], color='r', linewidth=lw)
#     # middle finger
#     plt.plot([pose[4, 0], pose[5, 0]], [pose[4, 1], pose[5, 1]], color='r', linewidth=lw)
#     plt.plot([pose[5, 0], pose[13, 0]], [pose[5, 1], pose[13, 1]], color='r', linewidth=lw)
#     # fore finger
#     plt.plot([pose[6, 0], pose[7, 0]], [pose[6, 1], pose[7, 1]], color='r', linewidth=lw)
#     plt.plot([pose[7, 0], pose[13, 0]], [pose[7, 1], pose[13, 1]], color='r', linewidth=lw)
#     # thumb
#     plt.plot([pose[8, 0], pose[9, 0]], [pose[8, 1], pose[9, 1]], color='r', linewidth=lw)
#     plt.plot([pose[9, 0], pose[10, 0]], [pose[9, 1], pose[10, 1]], color='r', linewidth=lw)
#     plt.plot([pose[10, 0], pose[13, 0]], [pose[10, 1], pose[13, 1]], color='r', linewidth=lw)
#     # palm
#     plt.plot([pose[13, 0], pose[12, 0]], [pose[13, 1], pose[12, 1]], color='r', linewidth=lw)
#     plt.plot([pose[13, 0], pose[11, 0]], [pose[13, 1], pose[11, 1]], color='r', linewidth=lw)
#
#     plt.xlim([min(pose_gt[:, 0]) - 30, max(pose_gt[:, 0]) + 30])
#     plt.ylim([max(pose_gt[:, 1]) + 30, min(pose_gt[:, 1]) - 30])
#     plt.axis('off')
#     plt.text(min(pose_gt[:, 0]) - 20, min(pose_gt[:, 1]) - 10, 'step 4 pose',
#              fontdict={'size': '20', 'weight': 'bold'})
#
#     plt.subplot(236)
#     plt.imshow(depth_img, cmap="gray")
#
#     pose = pose_gt
#     # plot pose skeleton
#     plt.scatter(pose[:, 0], pose[:, 1], color='b', marker='h', s=30)
#     # little finger
#     plt.plot([pose[0, 0], pose[1, 0]], [pose[0, 1], pose[1, 1]], color='b', linewidth=lw)
#     plt.plot([pose[1, 0], pose[13, 0]], [pose[1, 1], pose[13, 1]], color='b', linewidth=lw)
#     # ring finger
#     plt.plot([pose[2, 0], pose[3, 0]], [pose[2, 1], pose[3, 1]], color='b', linewidth=lw)
#     plt.plot([pose[3, 0], pose[13, 0]], [pose[3, 1], pose[13, 1]], color='b', linewidth=lw)
#     # middle finger
#     plt.plot([pose[4, 0], pose[5, 0]], [pose[4, 1], pose[5, 1]], color='b', linewidth=lw)
#     plt.plot([pose[5, 0], pose[13, 0]], [pose[5, 1], pose[13, 1]], color='b', linewidth=lw)
#     # fore finger
#     plt.plot([pose[6, 0], pose[7, 0]], [pose[6, 1], pose[7, 1]], color='b', linewidth=lw)
#     plt.plot([pose[7, 0], pose[13, 0]], [pose[7, 1], pose[13, 1]], color='b', linewidth=lw)
#     # thumb
#     plt.plot([pose[8, 0], pose[9, 0]], [pose[8, 1], pose[9, 1]], color='b', linewidth=lw)
#     plt.plot([pose[9, 0], pose[10, 0]], [pose[9, 1], pose[10, 1]], color='b', linewidth=lw)
#     plt.plot([pose[10, 0], pose[13, 0]], [pose[10, 1], pose[13, 1]], color='b', linewidth=lw)
#     # palm
#     plt.plot([pose[13, 0], pose[12, 0]], [pose[13, 1], pose[12, 1]], color='b', linewidth=lw)
#     plt.plot([pose[13, 0], pose[11, 0]], [pose[13, 1], pose[11, 1]], color='b', linewidth=lw)
#
#     pose = pose5
#     # plot pose skeleton
#     plt.scatter(pose[:, 0], pose[:, 1], color='r', marker='h', s=30)
#     # little finger
#     plt.plot([pose[0, 0], pose[1, 0]], [pose[0, 1], pose[1, 1]], color='r', linewidth=lw)
#     plt.plot([pose[1, 0], pose[13, 0]], [pose[1, 1], pose[13, 1]], color='r', linewidth=lw)
#     # ring finger
#     plt.plot([pose[2, 0], pose[3, 0]], [pose[2, 1], pose[3, 1]], color='r', linewidth=lw)
#     plt.plot([pose[3, 0], pose[13, 0]], [pose[3, 1], pose[13, 1]], color='r', linewidth=lw)
#     # middle finger
#     plt.plot([pose[4, 0], pose[5, 0]], [pose[4, 1], pose[5, 1]], color='r', linewidth=lw)
#     plt.plot([pose[5, 0], pose[13, 0]], [pose[5, 1], pose[13, 1]], color='r', linewidth=lw)
#     # fore finger
#     plt.plot([pose[6, 0], pose[7, 0]], [pose[6, 1], pose[7, 1]], color='r', linewidth=lw)
#     plt.plot([pose[7, 0], pose[13, 0]], [pose[7, 1], pose[13, 1]], color='r', linewidth=lw)
#     # thumb
#     plt.plot([pose[8, 0], pose[9, 0]], [pose[8, 1], pose[9, 1]], color='r', linewidth=lw)
#     plt.plot([pose[9, 0], pose[10, 0]], [pose[9, 1], pose[10, 1]], color='r', linewidth=lw)
#     plt.plot([pose[10, 0], pose[13, 0]], [pose[10, 1], pose[13, 1]], color='r', linewidth=lw)
#     # palm
#     plt.plot([pose[13, 0], pose[12, 0]], [pose[13, 1], pose[12, 1]], color='r', linewidth=lw)
#     plt.plot([pose[13, 0], pose[11, 0]], [pose[13, 1], pose[11, 1]], color='r', linewidth=lw)
#
#     plt.xlim([min(pose_gt[:, 0]) - 30, max(pose_gt[:, 0]) + 30])
#     plt.ylim([max(pose_gt[:, 1]) + 30, min(pose_gt[:, 1]) - 30])
#     plt.axis('off')
#     plt.text(min(pose_gt[:, 0]) - 20, min(pose_gt[:, 1]) - 10, 'step 5 pose',
#              fontdict={'size': '20', 'weight': 'bold'})
#
#     plt.savefig('demo%i.pdf'%i, bbox_inches='tight')
#     plt.show()
#
#
#
