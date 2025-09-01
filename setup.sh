#!/bin/bash

pip install -r requirements.txt
export NCCL_P2P_DISABLE=1 #disabling this helps with our NCCL issues on vLLM, feel free to comment this out if not necessary
