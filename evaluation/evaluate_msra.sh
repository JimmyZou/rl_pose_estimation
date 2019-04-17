#!/usr/bin/env bash
python src/compute_error.py msra max-frame\
	DenseReg   results/msra/CVPR18_MSRA_denseReg.txt\
	3DCNN   results/msra/CVPR17_MSRA_3DCNN.txt\
	HandPointNet	results/msra/CVPR18_MSRA_HandPointNet.txt\
	Point-to-Point	results/msra/ECCV18_MSRA_Point-to-Point.txt\
	RL-pose-pretrain results/msra/msra_pretrain_rl_pose_estimation.txt\
	RL-pose-step4 results/msra/msra_step4_rl_pose_estimation.txt\
	#V2V-PoseNet   results/msra/CVPR18_MSRA_V2V_PoseNet.txt\
	#Pose-REN   results/msra/NEUCOM18_MSRA_Pose_REN.txt\
	#SHPR-Net	results/msra/Access18_MSRA_SHPR_Net.txt\
	#Ren-9x6x6   results/msra/JVCI18_MSRA_REN_9x6x6.txt\



