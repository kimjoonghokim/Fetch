import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import torch
from transformers import AutoConfig, AutoTokenizer, LlamaForTokenClassification

load_dotenv(dotenv_path='../../server_config.env')

model_name_or_path = os.getenv("VERIFIER_MODEL_PATH")

if not model_name_or_path:
    raise ValueError("VERIFIER_MODEL_PATH environment variable not set")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
print("Tokenizer loaded successfully.")

value_model = LlamaForTokenClassification.from_pretrained(model_name_or_path, torch_dtype = torch.float16, device_map="auto")
value_model.eval()
print("Value model loaded successfully.")

app = FastAPI()

class InputText(BaseModel):
    texts: List[str]

class OutputPrediction(BaseModel):
    values: List[float]
    usage: dict

@app.post("/predict", response_model=OutputPrediction)
async def predict(input_text: InputText):
    max_seq_length = 1024
    inputs = tokenizer(input_text.texts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length)
    print("Length of tokenized sequences:", inputs["input_ids"].shape[1])
    
    # Calculate token usage
    prompt_tokens = sum(len(tokenizer.tokenize(text)) for text in input_text.texts)
    total_tokens = inputs["input_ids"].shape[1] * inputs["input_ids"].shape[0]  # batch_size * sequence_length
    completion_tokens = 0  # Verifier doesn't generate tokens, only processes input
    
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }
    
    inputs = {name: tensor.to(value_model.device) for name, tensor in inputs.items()}
    indices = torch.sum(inputs["attention_mask"], dim=-1) - 1
    with torch.no_grad():
        outputs = value_model(**inputs).logits.squeeze(-1)[torch.arange(len(indices)), indices].cpu().numpy().tolist()
    return {"values": outputs, "usage": usage}

