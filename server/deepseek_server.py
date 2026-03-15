"""
FastAPI server for DeepSeek-OCR-2 with OpenAI-compatible API.
Uses transformers directly (no vLLM) to avoid Triton/Blackwell issues.
Runs in the 'ocr-deepseek' conda environment on port 8003.
"""

import base64
import io
import os
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2"

app = FastAPI(title="DeepSeek-OCR-2 Server")

# Load model at startup
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model = model.eval()
print(f"{MODEL_NAME} loaded on {next(model.parameters()).device}")


# --- OpenAI-compatible API models ---

class ChatMessage(BaseModel):
    role: str
    content: list | str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    max_tokens: int = 4096
    temperature: float = 0.0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "deepseek"


# --- Endpoints ---

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "deepseek",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    # Extract text prompt and image from messages
    prompt_text = ""
    image = None

    for msg in request.messages:
        if isinstance(msg.content, str):
            prompt_text = msg.content
        elif isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        prompt_text = part["text"]
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            # base64 encoded image
                            header, b64data = url.split(",", 1)
                            image = Image.open(
                                io.BytesIO(base64.b64decode(b64data))
                            ).convert("RGB")

    if image is None:
        return {"error": "No image provided in the request"}

    # Save temp image for model.infer()
    tmp_path = f"/tmp/deepseek_ocr_{uuid.uuid4().hex}.png"
    image.save(tmp_path)

    result = model.infer(
        tokenizer,
        prompt=f"<image>\n{prompt_text}",
        image_file=tmp_path,
        output_path="/tmp/deepseek_ocr_out",
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=False,
        eval_mode=True,
    )

    os.unlink(tmp_path)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
