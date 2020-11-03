#! /bin/bash

set -x 


python -m util_scripts.eval_accuracy \
       datasets/hmdb51/json/hmdb51_3.json \
       datasets/hmdb51/results/val.json \
       --subset validation \
       -k 3 \
       --ignore
