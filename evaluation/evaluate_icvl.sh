#!/usr/bin/env bash
python src/compute_error.py icvl max-frame\
    DeepModel   results/icvl/IJCAI16_ICVL_DeepModel.txt\
    LRF         results/icvl/CVPR14_LRF_Results.txt\
    DenseReg   results/icvl/CVPR18_ICVL_denseReg.txt\
	V2V-PoseNet   results/icvl/CVPR18_ICVL_V2V_PoseNet.txt\
	HandPointNet	results/icvl/CVPR18_ICVL_HandPointNet.txt\
	Point-to-Point	results/icvl/ECCV18_ICVL_Point-to-Point.txt\
	RL-pose-pretrain results/icvl/icvl_pretrain_rl_pose_estimation.txt\
	RL-pose-step3 results/icvl/icvl_step3_rl_pose_estimation.txt\
	#Guo_Baseline    results/icvl/ICIP17_ICVL_Guo_Basic.txt\
    #Ren-4x6x6   results/icvl/ICIP17_ICVL_REN_4x6x6.txt\
    #Ren-9x6x6   results/icvl/JVCI18_ICVL_REN_9x6x6.txt\
	#Pose-REN   results/icvl/NEUCOM18_ICVL_Pose_REN.txt\

	#SHPR-Net	results/icvl/Access18_ICVL_SHPR_Net.txt\
    #DeepPrior   results/icvl/CVWW15_ICVL_Prior.txt\
    #DeepPrior-Refine   results/nyu/CVWW15_ICVL_Prior-Refinement.txt\

