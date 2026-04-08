# pdf-ocr

Make scanned PDFs searchable using open-weight OCR vision-language models.

Processes PDF files from an input directory, runs OCR to extract text, and produces searchable PDFs with an invisible text layer (Ctrl+F works) in the output directory.

## Supported OCR Engines

| Engine | Model | Params | Notes |
|--------|-------|--------|-------|
| `paddleocr` | PaddleOCR-VL | 0.9B | 109 languages, native PDF support |
| `hunyuan` | HunyuanOCR | 1B | 100+ languages |
| `deepseek` | DeepSeek-OCR-2 | 3B | Document-level VLM |
| `trocr` | Microsoft TrOCR | 334M | Line-level OCR, CPU friendly |

All engines can run **locally** (models auto-download on first use) or on a **remote GPU server** via OpenAI-compatible API (vLLM).

## Quick Start

### 1. Install Dependencies

```bash
# Create conda environment
conda create -n ocr python=3.12 -y
conda activate ocr

# Install base dependencies
pip install PyMuPDF Pillow requests numpy transformers sentencepiece
```

### 2. Add Your PDFs

```bash
# Place scanned PDFs in the input directory
mkdir -p input output
cp /path/to/your/scanned_notes.pdf input/
```

### 3. Run OCR

**Option A: Using a remote GPU server** (recommended for large files)

```bash
# Point to your GPU server running vLLM
python ocr_processor.py --engine paddleocr --server http://your-gpu-server:8004

# Output: output/scanned_notes_paddleocr.pdf (searchable!)
```

**Option B: Using TrOCR locally on CPU** (no GPU needed, smallest model)

```bash
# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Run TrOCR — model downloads automatically on first run (~1.3 GB)
python ocr_processor.py --engine trocr

# Output: output/scanned_notes_trocr.pdf
```

**Option C: Run all engines for comparison**

```bash
# Requires OCR_SERVER_HOST env var or server/.server_ip file
export OCR_SERVER_HOST=192.168.1.100
./run_all.sh

# Output: output/scanned_notes_paddleocr.pdf
#         output/scanned_notes_hunyuan.pdf
#         output/scanned_notes_deepseek.pdf
#         output/scanned_notes_trocr.pdf
```

### 4. Reprocess Specific Pages

If the OCR quality is poor on certain pages, reprocess just those pages:

```bash
# Reprocess page 55 only (updates existing output PDF)
python ocr_processor.py --engine paddleocr --server http://your-gpu-server:8004 --pages 55

# Reprocess a range of pages
python ocr_processor.py --engine paddleocr --server http://your-gpu-server:8004 --pages 10-20,55
```

## CLI Reference

```
python ocr_processor.py [OPTIONS]

Options:
  --engine     OCR engine: paddleocr, hunyuan, deepseek, trocr, all
               (default: paddleocr)
  --server     vLLM server URL for remote OCR (e.g., http://gpu-server:8004)
               If omitted, runs the engine locally (downloads model on first use)
  --input      Input directory containing PDF files (default: ./input)
  --output     Output directory for searchable PDFs (default: ./output)
  --dpi        Rendering DPI — higher = better OCR but more memory (default: 300)
  --pages      Pages to process: '55' or '10-20,55'
               If output exists, updates only those pages
  -v           Enable verbose/debug logging
```

## Model Auto-Download

When running locally (without `--server`), models are downloaded automatically from Hugging Face on first use and cached in `~/.cache/huggingface/`. Download sizes:

| Engine | Model Size | First-Run Download |
|--------|-----------|-------------------|
| trocr | 334M params | ~1.3 GB |
| paddleocr | 0.9B params | ~2 GB (requires paddlepaddle) |
| hunyuan | 1B params | ~2 GB (requires vllm + GPU) |
| deepseek | 3B params | ~6 GB (requires GPU) |

For local use of engines other than TrOCR, install engine-specific dependencies:

```bash
# PaddleOCR (GPU)
pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install -U "paddleocr[doc-parser]"

# PaddleOCR (CPU fallback)
pip install paddlepaddle==3.2.1
pip install -U "paddleocr[doc-parser]"

# TrOCR (GPU)
pip install torch torchvision

# TrOCR (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Server Setup (GPU Machine)

For remote OCR processing, set up vLLM servers on a GPU machine. See [server/README.md](server/README.md) for details.

```bash
cd server
./setup_ocr.sh                # One-time: creates conda envs, installs deps, downloads models
./start_ocr_services.sh       # Start all 3 OCR servers
./stop_ocr_services.sh        # Stop all servers
```

Services listen on:
- PaddleOCR-VL: port 8004
- HunyuanOCR: port 8002
- DeepSeek-OCR-2: port 8003

## Architecture

```
Client Machine                          GPU Server
──────────────                          ──────────
                     HTTP/JSON
ocr_processor.py  ──────────────>  vLLM (PaddleOCR-VL)     :8004
  --server URL    ──────────────>  vLLM (HunyuanOCR)       :8002
                  ──────────────>  FastAPI (DeepSeek-OCR-2) :8003

input/*.pdf  -->  [render page]  -->  [send to API]  -->  [get text]
                                                            |
output/*.pdf <--  [invisible text layer added]  <-----------+
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Project Structure

```
pdf-ocr/
├── ocr_processor.py       # Client CLI (main entry point)
├── pyproject.toml          # Project config (ruff, pytest, mypy)
├── requirements.txt        # Client dependencies
├── run_all.sh              # Batch wrapper (runs all engines)
├── run_trocr.sh            # TrOCR-only wrapper (CPU)
├── tests/                  # Unit tests (pytest)
└── server/                 # GPU server deployment
    ├── README.md            # Server setup guide + security notes
    ├── setup_ocr.sh         # One-time setup (conda envs, models)
    ├── start_ocr_services.sh
    ├── stop_ocr_services.sh
    └── deepseek_server.py   # FastAPI wrapper for DeepSeek-OCR-2
```
