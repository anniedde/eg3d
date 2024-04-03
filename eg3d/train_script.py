import os

cmd_cafu = (
    'CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py'
    ' --outdir=training-runs/Margot_t0/'
    ' --data=/playpen-nas-ssd/awang/data/Margot/0/train/preprocessed'
    ' --cfg=ffhq'
    ' --batch-gpu=4'
    ' --gpus=4'
    ' --batch=16'
    ' --kimg=500'
    ' --snap=5'
    ' --resume=networks/ffhqrebalanced512-128.pkl'
    ' --gamma=20'
    ' --gen_pose_cond=False'
    ' --neural_rendering_resolution_final=128'
    ' --aug=ada'
    ' --freezed=22'
    ' --freezeg=100' # the layers BEFORE this are trainable
)

os.system(cmd_cafu)
