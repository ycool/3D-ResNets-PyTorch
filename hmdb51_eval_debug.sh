#! /bin/bash

set -x 


python -m util_scripts.eval_debug \
       datasets/hmdb51/json/hmdb51_test.json \
       datasets/hmdb51/results/test.json \
       --subset test \
       -k 5
