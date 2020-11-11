#! /bin/bash

set -x 

# python -m pdb generate_model_graph.py \
    python generate_model_graph.py \
       --root_path datasets/hmdb51 \
       --video_path image \
       --annotation_path hmdb51_total.json \
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
       --pretrain_path models/ucf101_180.pth \
       --n_pretrain_classes 51 \
       --inference_batch_size 1

#        --ft_begin_module fc  \
    
#       --pretrain_path models/ucf101_180.pth \
#       --ft_begin_module fc  \
    
#        --pretrain_path models/r2p1d50_K_200ep.pth \

    
