import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, LlamaForTokenClassification
import asyncio
from concurrent.futures import ThreadPoolExecutor

model_name_or_path = ["/path/to/model/1",
                      "/path/to/model/2"]

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path[0], use_fast=True)
print("Tokenizer loaded successfully.")

value_models = []
for i in range(len(model_name_or_path)):
    value_model = LlamaForTokenClassification.from_pretrained(model_name_or_path[i], torch_dtype = torch.float16)
    value_model.to(f"cuda:{i}")
    value_model.eval()
    value_models.append(value_model)
print("Value model loaded successfully.")

app = FastAPI()

# this script has not been tested for passing multiple texts as input
class InputText(BaseModel):
    texts: List[str]

class OutputPrediction(BaseModel):
    values: List[float]

@app.post("/predict", response_model=OutputPrediction)
async def predict(input_text: InputText):
    max_seq_length = 1024
    # Tokenize输入文本
    inputs = tokenizer(input_text.texts, 
                      return_tensors="pt",
                      padding=True,
                      truncation=True,
                      max_length=max_seq_length)
    print("Length of tokenized sequences:", inputs["input_ids"].shape[1])

    # 定义模型推理函数
    def _run_model(model, inputs):
        # 将输入数据移动到模型所在设备
        device_inputs = {k: v.to(model.device) for k, v in inputs.items()}
        indices = torch.sum(device_inputs["attention_mask"], dim=-1) - 1
        
        with torch.no_grad():
            # 执行模型推理
            logits = model(**device_inputs).logits.squeeze(-1)
            # 获取序列最后一个有效token的输出
            outputs = logits[torch.arange(len(indices)), indices]
            # 数值截断并转numpy
            return torch.clamp(outputs, min=-1, max=1).cpu().item()

    # 使用线程池并行执行
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=2) as executor:
        # 创建并提交任务
        futures = [
            loop.run_in_executor(executor, _run_model, model, inputs)
            for model in value_models
        ]
        # 等待所有任务完成
        outputs = await asyncio.gather(*futures)

    # 计算平均值并返回
    avg_outputs = np.mean(outputs, axis=0).tolist()
    if not isinstance(avg_outputs, list):
        avg_outputs = [avg_outputs]
    return {"values": avg_outputs}