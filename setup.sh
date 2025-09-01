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
