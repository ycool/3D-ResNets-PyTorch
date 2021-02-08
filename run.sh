#! /bin/bash

set -x 

export MLFLOW_TRACKING_URI="http://0.0.0.0:8000"

# conda env list
# conda activate mlflow

# mlflow train demo baseline
# mlflow run -e train . --experiment-name action

# mlflow action pipeline test
mlflow run -e main . --experiment-name action



# to start mlflow server
# mlflow server --backend-store-uri postgresql://mlflow:mlflow@localhost/mlflow_db \
#    --default-artifact-root file:/home/hujiangtao/mlflow_db/mlruns -h 0.0.0.0 -p 8000
