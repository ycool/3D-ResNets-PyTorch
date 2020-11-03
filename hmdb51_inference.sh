#! /bin/bash

set -x 

python main.py \
       --root_path datasets/hmdb51 \
       --video_path image \
       --annotation_path hmdb51_3.json \
       --result_path results \
       --dataset hmdb51 \
       --resume_path results/save_200.pth \
       --model_depth 50 \
       --n_classes 51 \
       --n_threads 4 \
       --no_train \
       --no_val \
       --inference \
       --output_topk 5 \
       --inference_batch_size 1


