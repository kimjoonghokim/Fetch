#!/bin/bash

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
