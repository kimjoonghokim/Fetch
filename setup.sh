#!/bin/bash

# This script should install all the requirements as well as start up the policy, verifier, and embedding server
# The following steps are assuming you have 4 GPUs, 2 for the policy model and 1 for the embedding and 1 for the verifier model

pip install -r requirements.txt

# Load environment variables from server_config.env
set -o allexport
source ./server_config.env
set +o allexport

# Optional: print the model paths to confirm
echo "POLICY_MODEL_PATH=$POLICY_MODEL_PATH"
echo "VERIFIER_MODEL_PATH=$VERIFIER_MODEL_PATH"
echo "EMBEDDING_MODEL_PATH=$EMBEDDING_MODEL_PATH"

export NCCL_P2P_DISABLE=1 #disabling this helps with our NCCL issues on vLLM, feel free to comment this out if not necessary

CURRENT_DIR=$(pwd) # Save current directory

# Verifier model setup:
cd ./verifier || exit 1 # Change into the verifier directory
bash run.sh ./ 2 # Run verifier model on GPU 2
cd "$CURRENT_DIR" # Go back to original directory

# Embedding model setup:
cd ./cluster || exit 1 # Change into the cluster directory
bash run_app.sh ./ 3 # Run cluster model on GPU 3
cd "$CURRENT_DIR" # Go back to original directory

# Specify which GPUs to use for the policy server (GPU 0 and 1)
export CUDA_VISIBLE_DEVICES=0,1

# Run the vLLM API server using the policy model path from .env
python3 -m vllm.entrypoints.openai.api_server \
    --model "$POLICY_MODEL_PATH" \
    --port 8000 \
    --dtype float16 \
    --tensor-parallel-size 2 \
    --swap-space 8 \
    --max-model-len 4096
