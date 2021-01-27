#! /bin/bash

set -x 

python -m util_scripts.generate_video_jpgs \
       datasets/ucf101 \
       datasets/ucf101/image \
       ucf101

