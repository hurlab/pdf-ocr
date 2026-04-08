#!/bin/bash
set -e
# Run OCR on all input PDFs using all remote vLLM engines.
# Output: output/<filename>_paddleocr.pdf, output/<filename>_hunyuan.pdf, etc.

SERVER_URL="http://${OCR_SERVER_HOST:-YOUR_SERVER_IP}"

if [[ "$SERVER_URL" == *"YOUR_SERVER_IP"* ]]; then
  # Try reading from server/.server_ip
  if [ -f "server/.server_ip" ]; then
    SERVER_URL="http://$(cat server/.server_ip)"
  else
    echo "ERROR: Set OCR_SERVER_HOST or create server/.server_ip"
    echo "  export OCR_SERVER_HOST=192.168.1.100"
    exit 1
  fi
fi
DPI=300
INPUT="input"
OUTPUT="output"

ENGINES=(
  "paddleocr ${SERVER_URL}:8004"
  "hunyuan   ${SERVER_URL}:8002"
  "deepseek  ${SERVER_URL}:8003"
)

# Activate conda environment
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate ocr
mkdir -p "$OUTPUT"

echo "=== PDF OCR Comparison ==="
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo "DPI:    $DPI"
echo ""

# Remote engines via vLLM API
for entry in "${ENGINES[@]}"; do
  engine=$(echo "$entry" | awk '{print $1}')
  server=$(echo "$entry" | awk '{print $2}')

  echo "--- [$engine] via $server ---"
  python ocr_processor.py \
    --engine "$engine" \
    --server "$server" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --dpi "$DPI"
  echo ""
done

# Local TrOCR (CPU, line-level OCR)
echo "--- [trocr] local CPU ---"
python ocr_processor.py \
  --engine trocr \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --dpi "$DPI"
echo ""

echo "=== Done ==="
echo "Output files:"
ls -lh "$OUTPUT"/*.pdf 2>/dev/null
