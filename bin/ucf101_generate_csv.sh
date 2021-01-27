#! /bin/bash

set -x 

python -m util_scripts.hmdb51_csv \
       datasets/ucf101/image \
       datasets/ucf101/label

