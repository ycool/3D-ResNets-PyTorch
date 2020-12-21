#! /bin/bash

set -x 

# python
# pdb
# python -m pdb main.py --root_path datasets/hmdb51 \
# baseline 
# time python main.py --root_path datasets/hmdb51 \
# time python -m pdb main.py --root_path datasets/hmdb51 \
# time python -m memory_profiler main.py --root_path datasets/hmdb51 \
 time python main.py --root_path datasets/hmdb51 \
       --video_path image \
       --annotation_path hmdb51_total.json \
       --result_path results \
       --dataset hmdb51 \
       --model resnext \
       --resnext_cardinality 32 \
       --model_depth 50 \
       --n_classes 51 \
       --batch_size 128 \
       --n_threads 4 \
       --checkpoint 50 \
       --tensorboard \
       --n_epochs 200

#       --model resnet \
#       --model_depth 50 \

# pretain-ucf101-ft_begin_module_all
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
#        --checkpoint 50 \
#        --tensorboard \
#        --n_epochs 200 \
#        --pretrain_path models/ucf101_180.pth \
#        --n_pretrain_classes 101 
#       --ft_begin_module fc  # empty for fine tuning all layers

#        --pretrain_path models/r2p1d50_K_200ep.pth \

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


