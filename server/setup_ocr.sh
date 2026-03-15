#!/bin/bash
# Setup script for OCR vLLM services on DGX Spark
# Creates conda environments, installs dependencies, and downloads models

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAILSCALE_IP="100.67.76.96"

echo "========================================="
echo " OCR vLLM Services - Setup"
echo "========================================="
echo ""

# Make sure conda is available
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda is not installed or not in PATH."
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi

eval "$(conda shell.bash hook)"

# ─── Step 1: Conda environment 'ocr' (vLLM for PaddleOCR + HunyuanOCR) ──────

echo "[1/5] Setting up conda environment 'ocr'..."

if conda env list | grep -qw "^ocr "; then
    echo "  Conda environment 'ocr' already exists."
else
    echo "  Creating conda environment 'ocr' with Python 3.12..."
    conda create -n ocr python=3.12 -y
    echo "  Conda environment 'ocr' created."
fi

conda activate ocr
echo "  Activated environment: ocr (Python $(python --version 2>&1 | awk '{print $2}'))"
echo ""

# ─── Step 2: Install vllm + PyTorch CUDA ─────────────────────────────────────

echo "[2/5] Installing vllm and PyTorch with CUDA..."

if python -c "import vllm" 2>/dev/null; then
    VLLM_VER=$(python -c "import vllm; print(vllm.__version__)")
    echo "  vllm already installed (version $VLLM_VER)."
else
    echo "  Installing vllm via pip (this may take a few minutes)..."
    pip install vllm
    VLLM_VER=$(python -c "import vllm; print(vllm.__version__)")
    echo "  vllm $VLLM_VER installed."
fi

# Ensure PyTorch has CUDA support (vllm may pull CPU-only torch)
TORCH_CUDA=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$TORCH_CUDA" != "True" ]; then
    echo "  PyTorch missing CUDA support, reinstalling with CUDA 12.8..."
    pip install --force-reinstall --no-deps torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128
fi
echo "  PyTorch: $(python -c "import torch; print(torch.__version__)")"
echo "  CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")"
echo ""

# ─── Step 3: Conda environment 'ocr-deepseek' (transformers + FastAPI) ───────

echo "[3/5] Setting up conda environment 'ocr-deepseek'..."
echo "  (DeepSeek-OCR-2 uses MoE which requires Triton kernels not yet"
echo "   supported on Blackwell/sm_120. Uses transformers+FastAPI instead.)"

if conda env list | grep -qw "^ocr-deepseek "; then
    echo "  Conda environment 'ocr-deepseek' already exists."
else
    echo "  Creating conda environment 'ocr-deepseek' with Python 3.11..."
    conda create -n ocr-deepseek python=3.11 -y
    echo "  Conda environment 'ocr-deepseek' created."
fi

conda activate ocr-deepseek

if python -c "import transformers" 2>/dev/null; then
    echo "  Dependencies already installed."
else
    echo "  Installing PyTorch CUDA, transformers, and FastAPI..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install "transformers==4.46.3" "tokenizers==0.20.3" einops addict matplotlib \
        accelerate fastapi uvicorn python-multipart
fi
echo "  PyTorch: $(python -c "import torch; print(torch.__version__)")"
echo "  transformers: $(python -c "import transformers; print(transformers.__version__)")"
echo ""

# ─── Step 4: Download models ─────────────────────────────────────────────────

echo "[4/5] Downloading OCR models from Hugging Face..."
echo "  (Models are cached in ~/.cache/huggingface/)"
echo ""

download_model() {
    local model="$1"
    local label="$2"
    echo "  Downloading $label ($model)..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('$model')
print('    Done.')
"
}

download_model "PaddlePaddle/PaddleOCR-VL" "PaddleOCR-VL (~2 GB)"
download_model "tencent/HunyuanOCR" "HunyuanOCR (~2 GB)"
download_model "deepseek-ai/DeepSeek-OCR-2" "DeepSeek-OCR-2 (~6 GB)"

echo ""

# ─── Step 5: Summary ─────────────────────────────────────────────────────────

echo "[5/5] Verifying setup..."
echo ""
echo "========================================="
echo " Setup complete!"
echo "========================================="
echo ""
echo "Conda environments:"
echo "  ocr          - vLLM serving PaddleOCR-VL and HunyuanOCR"
echo "  ocr-deepseek - FastAPI/transformers serving DeepSeek-OCR-2"
echo ""
echo "To start all OCR services:"
echo "  $SCRIPT_DIR/start_ocr_services.sh"
echo ""
echo "To stop all OCR services:"
echo "  $SCRIPT_DIR/stop_ocr_services.sh"
echo ""
echo "Remote clients can connect via Tailscale:"
echo "  PaddleOCR-VL:   http://$TAILSCALE_IP:8004"
echo "  HunyuanOCR:     http://$TAILSCALE_IP:8002"
echo "  DeepSeek-OCR-2: http://$TAILSCALE_IP:8003"
echo ""
echo "Example from remote machine:"
echo "  python ocr_processor.py --engine paddleocr --server http://$TAILSCALE_IP:8004"
echo "  python ocr_processor.py --engine hunyuan   --server http://$TAILSCALE_IP:8002"
echo "  python ocr_processor.py --engine deepseek  --server http://$TAILSCALE_IP:8003"
