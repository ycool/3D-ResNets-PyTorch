#! /bin/bash

set -x 
TOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd $TOP_DIR

# python
# pdb
# python -m pdb main.py --root_path datasets/hmdb51 \
time python main.py --root_path datasets/ucf101 \
       --video_path image \
       --annotation_path ucf101_total.json \
       --result_path results \
       --dataset ucf101 \
       --model resnet \
       --model_depth 50 \
       --n_classes 101 \
       --batch_size 128 \
       --n_threads 4 \
       --checkpoint 30
#       --tensorboard \
#       --n_epochs 100

# time python main.py --root_path datasets/hmdb51 \
#        --video_path image \
#        --annotation_path hmdb51_total.json \
#        --result_path results \
#        --dataset hmdb51 \
#        --model resnet \
#        --model_depth 50 \
#        --n_classes 51 \
#        --batch_size 128 \
#        --n_threads 4 \
#        --checkpoint 30


