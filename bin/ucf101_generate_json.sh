#! /bin/bash

set -x 
TOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd $TOP_DIR

# to separate test / training / validation json file
# python -m util_scripts.hmdb51_json \
#        datasets/ucf101/label \
#        datasets/ucf101/image \
#        datasets/ucf101/json


# to generate total json file
time python -m util_scripts.hmdb51_json_total \
       datasets/ucf101/label \
       datasets/ucf101/image \
       datasets/ucf101/json
