# OCR Server Setup

Serves three OCR vision-language models via OpenAI-compatible APIs on a GPU server.

## Requirements

- NVIDIA GPU with 20+ GB VRAM (or unified memory)
- CUDA 12.x drivers
- Conda (Miniconda or Anaconda)

## Setup

```bash
./setup_ocr.sh
```

This script:
1. Auto-detects your server IP and prompts for confirmation
2. Creates conda env `ocr` with vLLM + PyTorch CUDA
3. Creates conda env `ocr-deepseek` with transformers + FastAPI
4. Downloads all three models from Hugging Face (~10 GB total)

The confirmed IP is saved to `.server_ip` for reuse by start/stop scripts.

## Start / Stop

```bash
./start_ocr_services.sh    # Start all 3 services
./stop_ocr_services.sh     # Graceful shutdown
```

## Services

| Model | Port | Backend | Conda Env |
|-------|------|---------|-----------|
| PaddleOCR-VL (0.9B) | 8004 | vLLM | `ocr` |
| HunyuanOCR (1B) | 8002 | vLLM | `ocr` |
| DeepSeek-OCR-2 (3B) | 8003 | FastAPI + transformers | `ocr-deepseek` |

All expose `/v1/models` and `/v1/chat/completions` (OpenAI-compatible).

## Why Two Conda Environments?

DeepSeek-OCR-2 requires a separate environment because:
- Its model code imports `LlamaFlashAttention2`, removed in `transformers>=4.47` (needs `==4.46.3`)
- vLLM requires `transformers>=4.56`
- vLLM's Triton MoE kernels don't yet support Blackwell (sm_120) GPUs

The workaround: serve DeepSeek via `deepseek_server.py` (FastAPI + transformers) instead of vLLM.

## Logs

```bash
tail -f logs/paddleocr.log
tail -f logs/hunyuan.log
tail -f logs/deepseek.log
```

## GPU Memory

Three models share GPU memory. Current allocation:
- PaddleOCR-VL: `--gpu-memory-utilization 0.25`
- HunyuanOCR: `--gpu-memory-utilization 0.20`
- DeepSeek-OCR-2: Managed by transformers (no explicit limit)

Adjust values in `start_ocr_services.sh` if needed.

## Environment Variable

Set `OCR_SERVER_HOST` to override the server IP:
```bash
export OCR_SERVER_HOST=192.168.1.100
./setup_ocr.sh   # will default to this IP
```

## Security

These servers are designed for **private network / VPN use only**. They have:
- No authentication (any client with network access can submit requests)
- No request size limits
- No rate limiting

Do **not** expose these ports to the public internet. Run behind a VPN (e.g., Tailscale, WireGuard) or firewall.
