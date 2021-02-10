#! /bin/bash

set -x 

export MLFLOW_TRACKING_URI="http://0.0.0.0:8000"

# conda env list
# conda activate mlflow

# mlflow train demo baseline
# mlflow run -e train . --experiment-name action

# mlflow action pipeline test
mlflow run -e train . --experiment-name action

# mlflow run -e inference . --experiment-name action

    # python generate_model_graph.py \
    #    --root_path datasets/hmdb51 \
    #    --video_path image \
    #    --annotation_path hmdb51_total.json \
    #    --result_path results \
    #    --dataset hmdb51 \
    #    --resume_path results/save_200.pth \
    #    --model_depth 50 \
    #    --n_classes 51 \
    #    --n_threads 4 \
    #    --no_train \
    #    --no_val \
    #    --inference \
    #    --output_topk 5 \
    #    --inference_batch_size 1


# to start mlflow server
# mlflow server --backend-store-uri postgresql://mlflow:mlflow@localhost/mlflow_db \
#    --default-artifact-root file:/home/hujiangtao/mlflow_db/mlruns -h 0.0.0.0 -p 8000
