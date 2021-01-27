#! /bin/bash

set -x 

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
$D/hmdb51_inference.sh

# calculate metrics
$D/hmdb51_val.sh

