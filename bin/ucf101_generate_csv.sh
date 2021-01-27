#! /bin/bash

set -x 
TOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd $TOP_DIR

python -m util_scripts.hmdb51_csv \
       datasets/ucf101/image \
       datasets/ucf101/label

