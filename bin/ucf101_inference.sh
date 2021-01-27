#! /bin/bash

set -x 

# output results/val.json
python main.py \
       --root_path datasets/ucf101 \
       --video_path image \
       --annotation_path ucf101_test.json \
       --inference_subset test \
       --result_path results \
       --dataset ucf101 \
       --resume_path results/save_180.pth \
       --model_depth 50 \
       --n_classes 101 \
       --n_threads 4 \
       --no_train \
       --no_val \
       --inference \
       --output_topk 5 \
       --inference_batch_size 1

# input: results/val.json
python -m util_scripts.eval_accuracy \
       datasets/ucf101/ucf101_test.json \
       datasets/ucf101/results/test.json \
       --subset test \
       -k 1 \
       --ignore

python -m util_scripts.eval_accuracy \
       datasets/ucf101/ucf101_test.json \
       datasets/ucf101/results/test.json \
       --subset test \
       -k 2 \
       --ignore

python -m util_scripts.eval_accuracy \
       datasets/ucf101/ucf101_test.json \
       datasets/ucf101/results/val.json \
       --subset test \
       -k 3 \
       --ignore


# python main.py \
#        --root_path datasets/ucf101 \
#        --video_path image \
#        --annotation_path ucf101_validation.json \
#        --inference_subset val \
#        --result_path results \
#        --dataset ucf101 \
#        --resume_path results/save_180.pth \
#        --model_depth 50 \
#        --n_classes 101 \
#        --n_threads 4 \
#        --no_train \
#        --no_val \
#        --inference \
#        --output_topk 1 \
#        --inference_batch_size 1
       

