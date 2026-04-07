#!/bin/bash
# Start vLLM OCR model servers
# Models are served on ports 8002-8004 with OpenAI-compatible API

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PID_DIR="$SCRIPT_DIR/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ocr

echo "Starting OCR vLLM services..."

# PaddleOCR-VL on port 8004
if [ -f "$PID_DIR/paddleocr.pid" ] && kill -0 "$(cat "$PID_DIR/paddleocr.pid")" 2>/dev/null; then
    echo "PaddleOCR-VL already running (PID $(cat "$PID_DIR/paddleocr.pid"))"
else
    echo "Starting PaddleOCR-VL on port 8004..."
    nohup vllm serve PaddlePaddle/PaddleOCR-VL \
        --trust-remote-code \
        --port 8004 \
        --host 0.0.0.0 \
        --gpu-memory-utilization 0.25 \
        --enforce-eager \
        > "$LOG_DIR/paddleocr.log" 2>&1 &
    echo $! > "$PID_DIR/paddleocr.pid"
    echo "  PaddleOCR-VL started (PID $!)"
fi

# HunyuanOCR on port 8002
if [ -f "$PID_DIR/hunyuan.pid" ] && kill -0 "$(cat "$PID_DIR/hunyuan.pid")" 2>/dev/null; then
    echo "HunyuanOCR already running (PID $(cat "$PID_DIR/hunyuan.pid"))"
else
    echo "Starting HunyuanOCR on port 8002..."
    nohup vllm serve tencent/HunyuanOCR \
        --port 8002 \
        --host 0.0.0.0 \
        --gpu-memory-utilization 0.2 \
        --enforce-eager \
        > "$LOG_DIR/hunyuan.log" 2>&1 &
    echo $! > "$PID_DIR/hunyuan.pid"
    echo "  HunyuanOCR started (PID $!)"
fi

# DeepSeek-OCR-2 on port 8003
# Uses separate conda env (ocr-deepseek) with transformers + FastAPI
# because vLLM's Triton MoE kernels don't support Blackwell (sm_120).
if [ -f "$PID_DIR/deepseek.pid" ] && kill -0 "$(cat "$PID_DIR/deepseek.pid")" 2>/dev/null; then
    echo "DeepSeek-OCR-2 already running (PID $(cat "$PID_DIR/deepseek.pid"))"
else
    echo "Starting DeepSeek-OCR-2 on port 8003 (FastAPI/transformers)..."
    conda activate ocr-deepseek
    nohup python "$SCRIPT_DIR/deepseek_server.py" \
        > "$LOG_DIR/deepseek.log" 2>&1 &
    echo $! > "$PID_DIR/deepseek.pid"
    echo "  DeepSeek-OCR-2 started (PID $!)"
    conda activate ocr
fi

echo ""
echo "All services starting. Check logs with:"
echo "  tail -f $LOG_DIR/paddleocr.log"
echo "  tail -f $LOG_DIR/hunyuan.log"
echo "  tail -f $LOG_DIR/deepseek.log"
echo ""
# Read saved server IP
SERVER_IP="YOUR_SERVER_IP"
if [ -f "$SCRIPT_DIR/.server_ip" ]; then
    SERVER_IP=$(cat "$SCRIPT_DIR/.server_ip")
fi

echo "Services will be available at:"
echo "  ┌────────────────┬──────────────────────────┐"
echo "  │ PaddleOCR-VL   │ http://$SERVER_IP:8004 │"
echo "  ├────────────────┼──────────────────────────┤"
echo "  │ HunyuanOCR     │ http://$SERVER_IP:8002 │"
echo "  ├────────────────┼──────────────────────────┤"
echo "  │ DeepSeek-OCR-2 │ http://$SERVER_IP:8003 │"
echo "  └────────────────┴──────────────────────────┘"
