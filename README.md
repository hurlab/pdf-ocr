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

All engines can run **locally** or on a **remote GPU server** via OpenAI-compatible API (vLLM).

## Quick Start

### Client Setup

```bash
# Create conda environment
conda create -n ocr python=3.12 -y
conda activate ocr
pip install PyMuPDF Pillow requests
```

### Usage

```bash
# Process PDFs using a remote vLLM server
python ocr_processor.py --engine paddleocr --server http://your-gpu-server:8004

# Process with all engines for comparison
# (outputs: document_paddleocr.pdf, document_hunyuan.pdf, document_deepseek.pdf)
./run_all.sh

# Reprocess specific pages (updates existing output)
python ocr_processor.py --engine paddleocr --server http://your-gpu-server:8004 --pages 55

# Process a page range
python ocr_processor.py --engine hunyuan --server http://your-gpu-server:8002 --pages 10-20,55
```

### CLI Options

```
--engine     OCR engine: paddleocr, hunyuan, deepseek, trocr, all
--server     vLLM server URL (e.g., http://gpu-server:8004)
--input      Input directory (default: ./input)
--output     Output directory (default: ./output)
--dpi        Rendering DPI (default: 300)
--pages      Pages to process: '55' or '10-20,55' (updates existing output)
```

## Server Setup (GPU Machine)

See [server/README.md](server/README.md) for setting up vLLM OCR servers on a GPU machine (tested on NVIDIA GPU server with 128GB unified memory).

```bash
cd server
./setup_ocr.sh      # One-time: creates envs, installs deps, downloads models
./start_ocr_services.sh   # Start all 3 OCR servers
./stop_ocr_services.sh    # Stop all servers
```

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

## Project Structure

```
pdf-ocr/
├── ocr_processor.py       # Client CLI
├── run_all.sh             # Batch wrapper (runs all engines)
├── requirements.txt       # Client dependencies
└── server/                # GPU server deployment
    ├── setup_ocr.sh       # One-time setup (conda envs, models)
    ├── start_ocr_services.sh
    ├── stop_ocr_services.sh
    └── deepseek_server.py # FastAPI wrapper for DeepSeek-OCR-2
```
