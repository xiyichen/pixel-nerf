#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cluster/scratch/xiychen/miniconda3/lib/
module load cuda/11.1.1
python train/train.py -n srn_car_exp -c conf/exp/srn.conf -D /cluster/scratch/xiychen/data/thuman_0.6smpl/thuman --gpu_id=0