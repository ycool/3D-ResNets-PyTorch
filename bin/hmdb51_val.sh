#! /bin/bash

set -x 
TOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd $TOP_DIR


python -m util_scripts.eval_accuracy \
       datasets/hmdb51/json/hmdb51_test.json \
       datasets/hmdb51/results/test.json \
       --subset test \
       -k 1 \
       --ignore

python -m util_scripts.eval_accuracy \
       datasets/hmdb51/json/hmdb51_test.json \
       datasets/hmdb51/results/test.json \
       --subset test \
       -k 2 \
       --ignore

python -m util_scripts.eval_accuracy \
       datasets/hmdb51/json/hmdb51_test.json \
       datasets/hmdb51/results/test.json \
       --subset test \
       -k 3 \
       --ignore
