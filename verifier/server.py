import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List

import torch
from transformers import AutoConfig, AutoTokenizer, LlamaForTokenClassification

model_name_or_path = "xmu-nlp/Llama-3-8b-gsm8k-value-A"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
print("Tokenizer loaded successfully.")

value_model = LlamaForTokenClassification.from_pretrained(model_name_or_path, torch_dtype = torch.float16, device_map="auto")
value_model.eval()
print("Value model loaded successfully.")

app = FastAPI(title="Value Model Server", description="Server for LLM Value Model Predictions")

class InputText(BaseModel):
    texts: List[str]

class OutputPrediction(BaseModel):
    values: List[float]

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Value Model Server</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .status { background: #d4edda; color: #155724; padding: 15px; border-radius: 5px; margin: 20px 0; }
            textarea { width: 100%; height: 150px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-family: monospace; }
            button { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background: #0056b3; }
            .result { background: #e7f3ff; padding: 15px; border-radius: 5px; margin-top: 20px; white-space: pre-wrap; font-family: monospace; }
            .error { background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Value Model Server</h1>
            <div class="status">
                ✅ Server is running successfully!<br>
                🤖 Model: xmu-nlp/Llama-3-8b-gsm8k-value-A<br>
                📡 Endpoint: POST /predict
            </div>
            
            <h3>Test the Model</h3>
            <p>Enter one or more texts (one per line) to get value predictions:</p>
            
            <form onsubmit="predict(event)">
                <textarea id="texts" placeholder="Enter text to analyze, e.g.:
The answer is 42.
This solution looks correct.
I'm not sure about this step."></textarea><br><br>
                <button type="submit">Get Predictions</button>
            </form>
            
            <div id="result"></div>
            
            <h3>API Usage</h3>
            <p>You can also use the API directly:</p>
            <pre style="background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;">
curl -X POST "http://localhost:8002/predict" \\
  -H "Content-Type: application/json" \\
  -d '{"texts": ["Your text here"]}'</pre>
        </div>

        <script>
        async function predict(event) {
            event.preventDefault();
            const resultDiv = document.getElementById('result');
            const textsInput = document.getElementById('texts');
            
            const texts = textsInput.value.split('\\n').filter(text => text.trim() !== '');
            
            if (texts.length === 0) {
                resultDiv.innerHTML = '<div class="error">Please enter at least one text!</div>';
                return;
            }
            
            resultDiv.innerHTML = '<div>Processing...</div>';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({texts: texts})
                });
                
                if (response.ok) {
                    const result = await response.json();
                    let output = 'Predictions:\\n\\n';
                    texts.forEach((text, i) => {
                        output += `Text ${i+1}: "${text}"\\n`;
                        output += `Value: ${result.values[i]}\\n\\n`;
                    });
                    resultDiv.innerHTML = '<div class="result">' + output + '</div>';
                } else {
                    const error = await response.text();
                    resultDiv.innerHTML = '<div class="error">Error: ' + error + '</div>';
                }
            } catch (error) {
                resultDiv.innerHTML = '<div class="error">Network error: ' + error.message + '</div>';
            }
        }
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "xmu-nlp/Llama-3-8b-gsm8k-value-A", "message": "Value model server is running"}

@app.post("/predict", response_model=OutputPrediction)
async def predict(input_text: InputText):
    max_seq_length = 1024
    inputs = tokenizer(input_text.texts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length)
    print("Length of tokenized sequences:", inputs["input_ids"].shape[1])
    inputs = {name: tensor.to(value_model.device) for name, tensor in inputs.items()}
    indices = torch.sum(inputs["attention_mask"], dim=-1) - 1
    with torch.no_grad():
        outputs = value_model(**inputs).logits.squeeze(-1)[torch.arange(len(indices)), indices].cpu().numpy().tolist()
    return {"values": outputs}
