#! /bin/bash

set -x 

python main.py \
       --root_path datasets/ucf101 \
       --video_path image \
       --annotation_path ucf101_validation.json \
       --result_path results \
       --dataset ucf101 \
       --resume_path results/save_180.pth \
       --model_depth 50 \
       --n_classes 101 \
       --n_threads 4 \
       --no_train \
       --no_val \
       --inference \
       --output_topk 1 \
       --inference_batch_size 1


