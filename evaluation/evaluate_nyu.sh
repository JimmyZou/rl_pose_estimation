#!/usr/bin/env bash
python src/compute_error.py nyu max-frame\
    DeepModel   results/nyu/IJCAI16_NYU_DeepModel.txt\
    Lie-X       results/nyu/IJCV16_NYU_LieX.txt\
	3DCNN   results/nyu/CVPR17_NYU_3DCNN.txt\
	DenseReg   results/nyu/CVPR18_NYU_denseReg.txt\
	V2V-PoseNet   results/nyu/CVPR18_NYU_V2V_PoseNet.txt\
	FeatureMapping	results/nyu/CVPR18_NYU_DeepPrior++_FM.txt\
	HandPointNet	results/nyu/CVPR18_NYU_HandPointNet.txt\
	Point-to-Point	results/nyu/ECCV18_NYU_Point-to-Point.txt\
	RL-pose-pretrain results/nyu/nyu_pretrain_rl_pose_estimation.txt\
	RL-pose-step3 results/nyu/nyu_step3_rl_pose_estimation.txt\
	#MURAUER	results/nyu/WACV19_NYU_murauer_n72757_uvd.txt\
	#REN-4x6x6   results/nyu/ICIP17_NYU_REN_4x6x6.txt\
    #REN-9x6x6   results/nyu/JVCI18_NYU_REN_9x6x6.txt\
	#DeepPrior++   results/nyu/ICCVW17_NYU_DeepPrior++.txt\
    #Pose-REN   results/nyu/NEUCOM18_NYU_Pose_REN.txt\
    #DeepPrior   results/nyu/CVWW15_NYU_Prior.txt\
    #DeepPrior-Refine   results/nyu/CVWW15_NYU_Prior-Refinement.txt\
    #Feedback    results/nyu/ICCV15_NYU_Feedback.txt\
    #SHPR-Net	results/nyu/Access18_NYU_SHPR_Net_frontal.txt\
	#SHPR-Net\(three\ views\)	results/nyu/Access18_NYU_SHPR_Net_three.txt\
	#Guo_Baseline    results/nyu/ICIP17_NYU_Guo_Basic.txt\
