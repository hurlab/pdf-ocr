#!/bin/bash
# Run OCR on all input PDFs using all remote vLLM engines on DGX Spark.
# Output: output/APUSH_paddleocr.pdf, output/APUSH_hunyuan.pdf, etc.

DGX="http://100.67.76.96"
DPI=300
INPUT="input"
OUTPUT="output"

ENGINES=(
  "paddleocr ${DGX}:8004"
  "hunyuan   ${DGX}:8002"
  "deepseek  ${DGX}:8003"
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

echo "=== Done ==="
echo "Output files:"
ls -lh "$OUTPUT"/*.pdf 2>/dev/null
