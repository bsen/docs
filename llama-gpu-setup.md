# LLama 3.2 3B GPU Server Setup Guide

This guide details how to set up a Llama 3.2 3B model server on an AWS g4dn.xlarge instance with GPU support.

## Prerequisites

- AWS g4dn.xlarge instance (or similar with GPU)
- Ubuntu Server
- Access to Hugging Face with Llama 3 permissions
- At least 50GB storage

## Step 1: Initial Setup

```bash
# Update system
sudo apt update

# Install NVIDIA drivers
sudo apt install -y nvidia-utils-535 nvidia-driver-535

# Reboot the system
sudo reboot
```

## Step 2: Python Environment Setup

```bash
# Install Python packages
sudo apt install python3-pip python3-venv

# Create virtual environment
python3 -m venv llama_env
source llama_env/bin/activate

# Install required packages
pip install torch transformers accelerate
pip install fastapi uvicorn sse-starlette python-dotenv
```

## Step 3: Verify GPU Setup

```bash
# Check NVIDIA driver installation
nvidia-smi

# Verify CUDA availability in Python
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Step 4: Download Model

Create a file named `download_model.py`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv
import os

def download_model():
    print("\nStarting download process...")
    
    # Load environment variables
    load_dotenv()
    token = os.getenv('HF_TOKEN')
    
    if not token:
        raise ValueError("HF_TOKEN not found in .env file")
    
    MODEL_ID = "meta-llama/LLama-3.2-3B"
    
    try:
        print("\nStep 1: Downloading config file...")
        config_file = hf_hub_download(
            repo_id=MODEL_ID,
            filename="config.json",
            token=token,
            cache_dir="./model_cache"
        )
        print(f"Config file downloaded to: {config_file}")
        
        print("\nStep 2: Downloading tokenizer files...")
        tokenizer_file = hf_hub_download(
            repo_id=MODEL_ID,
            filename="tokenizer.json",
            token=token,
            cache_dir="./model_cache"
        )
        print(f"Tokenizer file downloaded to: {tokenizer_file}")
        
        print("\nStep 3: Now trying to load the model...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            token=token,
            cache_dir="./model_cache",
            trust_remote_code=True
        )
        print("Tokenizer loaded successfully!")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=token,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir="./model_cache",
            trust_remote_code=True
        )
        print("Model loaded successfully!")
        
        # Save both locally
        print("\nSaving model and tokenizer locally...")
        tokenizer.save_pretrained("./llama-3.2-3b-model")
        model.save_pretrained("./llama-3.2-3b-model")
        print("\nEverything completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(f"Error type: {type(e)}")

if __name__ == "__main__":
    download_model()
```

Create a `.env` file:
```
HF_TOKEN=your_huggingface_token_here
```

Run the download:
```bash
python download_model.py
```

## Step 5: Server Implementation

Create a file named `server.py`:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
from sse_starlette.sse import EventSourceResponse
import threading
import json

class GenerateRequest(BaseModel):
    message: str
    context: Optional[List[int]] = []
    system: Optional[str] = ""
    stream: bool = False

class LlamaModel:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this model")
            
        self.device = torch.device("cuda")
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        print("\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("./llama-3.2-3b-model")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Tokenizer loaded successfully!")
        
        print("\nLoading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "./llama-3.2-3b-model",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.model.eval()
        print("Model loaded successfully on GPU!\n")

    async def generate_text(self, prompt: str):
        print(f"\nStarting generation for: {prompt[:50]}...")
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=2048
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("Generation completed successfully")
            return response[len(prompt):].strip()
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            raise

    async def generate_stream(self, prompt: str):
        print(f"\nStarting streaming generation for: {prompt[:50]}...")
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=2048
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
            )

            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            for new_text in streamer:
                if new_text.strip():
                    yield {
                        "event": "message",
                        "data": json.dumps({"text": new_text, "done": False})
                    }
            
            yield {
                "event": "message",
                "data": json.dumps({"text": "", "done": True})
            }
            print("Streaming completed successfully")
            
        except Exception as e:
            print(f"Error during streaming: {str(e)}")
            raise

app = FastAPI()
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model = LlamaModel()

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    prompt = request.message
    if request.system:
        prompt = f"{request.system}\n\n{prompt}"

    if request.stream:
        return EventSourceResponse(model.generate_stream(prompt))
    else:
        response = await model.generate_text(prompt)
        return {"response": response}

@app.get("/api/version")
async def version():
    return {"version": "1.0.0"}

if __name__ == "__main__":
    print("\nStarting server on port 11434...")
    uvicorn.run(app, host="0.0.0.0", port=11434)
```

## Step 6: Configure AWS Security Group

Add inbound rules in your AWS security group:
- Custom TCP: Port 11434
- Source: Your IP or 0.0.0.0/0 (for all IPs)

## Step 7: Running the Server

```bash
# Start the server
python server.py
```

## Step 8: Testing

Test non-streaming:
```bash
curl -X POST http://YOUR_IP:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, what can you do?", "stream": false}'
```

Test streaming:
```bash
curl -N -X POST http://YOUR_IP:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, what can you do?", "stream": true}'
```

## API Endpoints

- `POST /api/generate`: Generate text from the model
  - Parameters:
    - `message`: Input text prompt
    - `system`: Optional system prompt
    - `stream`: Boolean for streaming response
- `GET /api/version`: Get server version

## Notes
- Make sure you have sufficient disk space (at least 50GB recommended)
- Monitor GPU memory usage with `nvidia-smi`
- The model runs in FP16 mode for better memory efficiency
