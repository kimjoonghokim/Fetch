pip install -r requirements.txt

python3 -m vllm.entrypoints.openai.api_server --model xmu-nlp/Llama-3-8b-gsm8k --port
8000 --dtype float16 --swap-space 8 --max-model-len 4096