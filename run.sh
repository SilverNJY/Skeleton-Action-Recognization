#!/bin/bash
source /root/miniconda3/bin/activate /root/miniconda3/envs/lst
cd /opt/data/private/MMVRAC/LST


# my_preprocessed_uav | only joint info
date
CUDA_VISIBLE_DEVICES=0 python main_multipart_uav.py --config config/uav-cross-subject/lst_joint_v1.yaml --model model.ctrgcn.Model_lst_4part_uav --work-dir work_dir/uav/my/xsub_v1/lst_joint --device 0 --log-interval 5 --save-interval 1
date
CUDA_VISIBLE_DEVICES=0 python main_multipart_uav.py --config config/uav-cross-subject/lst_joint_v2.yaml --model model.ctrgcn.Model_lst_4part_uav --work-dir work_dir/uav/my/xsub_v2/lst_joint --device 0 --log-interval 5 --save-interval 1


# my_preprocessed_uav | joint+ motion info
date
CUDA_VISIBLE_DEVICES=0 python main_multipart_uav.py --config config/uav-cross-subject/lst_joint_vel_v1.yaml --model model.ctrgcn.Model_lst_4part_uav --work-dir work_dir/uav/my/xsub_v1/lst_joint_motion --device 0 --log-interval 5 --save-interval 1  
date
CUDA_VISIBLE_DEVICES=0 python main_multipart_uav.py --config config/uav-cross-subject/lst_joint_vel_v2.yaml --model model.ctrgcn.Model_lst_4part_uav --work-dir work_dir/uav/my/xsub_v2/lst_joint_motion --device 0 --log-interval 5 --save-interval 1  


# my_preprocessed_uav | only bone info
date
CUDA_VISIBLE_DEVICES=0 python main_multipart_uav.py --config config/uav-cross-subject/lst_bone_v1.yaml --model model.ctrgcn.Model_lst_4part_uav --work-dir work_dir/uav/my/xsub_v1/lst_bone --device 0 --log-interval 5 --save-interval 1
date
CUDA_VISIBLE_DEVICES=0 python main_multipart_uav.py --config config/uav-cross-subject/lst_bone_v2.yaml --model model.ctrgcn.Model_lst_4part_uav --work-dir work_dir/uav/my/xsub_v2/lst_bone --device 0 --log-interval 5 --save-interval 1


# my_preprocessed_uav | bone+motion info
date
CUDA_VISIBLE_DEVICES=0 python main_multipart_uav.py --config config/uav-cross-subject/lst_bone_vel_v1.yaml --model model.ctrgcn.Model_lst_4part_uav --work-dir work_dir/uav/my/xsub_v1/lst_bone_motion --device 0 --log-interval 5 --save-interval 1
date
CUDA_VISIBLE_DEVICES=0 python main_multipart_uav.py --config config/uav-cross-subject/lst_bone_vel_v2.yaml --model model.ctrgcn.Model_lst_4part_uav --work-dir work_dir/uav/my/xsub_v2/lst_bone_motion --device 0 --log-interval 5 --save-interval 1




# my_preprocessed_uav | only bone+motion_bone info
ensemble four modalities on UAV-Human cross subject v1
python ensemble.py --dataset uav/xsub_v1 --joint-dir work_dir/uav/my/xsub_v1/lst_joint --bone-dir work_dir/uav/my/xsub_v1/lst_bone --joint-motion-dir work_dir/uav/my/xsub_v1/lst_joint_motion --bone-motion-dir work_dir/uav/my/xsub_v1/lst_bone_motion
# ensemble four modalities on UAV-Human cross subject v2
python ensemble.py --dataset uav/xsub_v2 --joint-dir work_dir/uav/my/xsub_v2/lst_joint --bone-dir work_dir/uav/my/xsub_v2/lst_bone --joint-motion-dir work_dir/uav/my/xsub_v2/lst_joint_motion --bone-motion-dir work_dir/uav/my/xsub_v2/lst_bone_motion


