#!/bin/bash
# Run TrOCR (local CPU) on all input PDFs.
# Output: output/<filename>_trocr.pdf

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate ocr

echo "--- [trocr] local CPU (microsoft/trocr-base-handwritten) ---"
python ocr_processor.py \
  --engine trocr \
  --input input \
  --output output \
  --dpi 200

echo ""
echo "Done. Output:"
ls -lh output/*trocr*.pdf 2>/dev/null
