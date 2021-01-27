#! /bin/bash

set -x 

python -m util_scripts.generate_video_jpgs \
       datasets/hmdb51 \
       datasets/hmdb51/jpg \
       hmdb51

