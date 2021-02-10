#! /bin/bash

set -x 

TOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd $TOP_DIR

D=/apollo/3D-ResNets-PyTorch

# data conversion
$D/hmdb51_video_jpg.sh

# image files to generate csv files
$D/hmdb51_generate_csv.sh

# csv + image => json label file
$D/hmdb51_generate_json.sh

# train model from hmdb51 dataset
$D/hmdb51_train.sh

# inference test dataset
# output: datasets/hmdb51/results/test.json
#    #inference_subset#.json
$D/hmdb51_inference.sh

# calculate metrics
$D/hmdb51_val.sh

