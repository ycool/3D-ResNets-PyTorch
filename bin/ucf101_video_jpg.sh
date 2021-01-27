#! /bin/bash

set -x 
TOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd $TOP_DIR

python -m util_scripts.generate_video_jpgs \
       datasets/ucf101 \
       datasets/ucf101/image \
       ucf101

