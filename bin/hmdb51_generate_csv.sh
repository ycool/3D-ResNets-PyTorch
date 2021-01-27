#! /bin/bash

set -x 

python -m util_scripts.hmdb51_csv \
       datasets/hmdb51/image \
       datasets/hmdb51/label

