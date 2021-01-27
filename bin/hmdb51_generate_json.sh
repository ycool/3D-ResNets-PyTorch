#! /bin/bash

set -x 

TOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd $TOP_DIR

# to separate test / training / validation json file
# python -m util_scripts.hmdb51_json \
#        datasets/hmdb51/label \
#        datasets/hmdb51/image \
#        datasets/hmdb51/json

# to generate total json file
python -m util_scripts.hmdb51_json_total \
       datasets/hmdb51/label \
       datasets/hmdb51/image \
       datasets/hmdb51/json
