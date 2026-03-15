# PROJECT_LOG.md

## Session 2026-03-15 10:41 CDT

- **Coding CLI used:** Claude Code CLI (claude-opus-4-6)
- **Phase(s) worked on:** Full initial setup — scripts, environments, dependencies, model serving

### Concrete changes implemented
- Created `start_ocr_services.sh` — launches PaddleOCR-VL (vLLM, port 8004), HunyuanOCR (vLLM, port 8002), DeepSeek-OCR-2 (FastAPI, port 8003)
- Created `stop_ocr_services.sh` — graceful shutdown with PID tracking
- Created `setup_ocr.sh` — 5-step setup: two conda envs (`ocr` for vLLM, `ocr-deepseek` for transformers+FastAPI), dependency installation, model downloads
- Created `deepseek_server.py` — FastAPI server wrapping DeepSeek-OCR-2 via transformers with OpenAI-compatible `/v1/models` and `/v1/chat/completions` endpoints

### Files/modules/functions touched
- `setup_ocr.sh` (created, then rewritten to handle two-env architecture)
- `start_ocr_services.sh` (created, iterated through port/memory/backend changes)
- `stop_ocr_services.sh` (created)
- `deepseek_server.py` (created)

### Key technical decisions and rationale
1. **Port 8001→8004 for PaddleOCR:** Port 8001 already in use by unknown process. Moved to 8004.
2. **--enforce-eager on all vLLM models:** CUDA graph capture fails on Blackwell sm_120.
3. **Separate conda env for DeepSeek (ocr-deepseek):** Two conflicts: (a) DeepSeek-OCR-2's custom model code imports `LlamaFlashAttention2` which was removed in transformers>=4.47, requiring transformers==4.46.3; (b) vLLM requires transformers>=4.56.
4. **FastAPI+transformers instead of vLLM for DeepSeek:** Triton 3.6.0 MoE kernels fail with `nvrtc: error: invalid value for --gpu-architecture` on sm_120. Pure PyTorch MoE in transformers works fine. Evaluated env var workarounds (VLLM_DISABLED_KERNELS, VLLM_MOE_KERNEL_BACKEND) but they don't fix prebuilt wheels.
5. **GPU memory utilization splits:** PaddleOCR 0.25, HunyuanOCR 0.2. Initial 0.15 for PaddleOCR was too low (no KV cache space).
6. **PyTorch CUDA force-reinstall:** vLLM pip install pulls cpu-only torch. Must `pip install --force-reinstall --no-deps torch --index-url .../cu128` after vLLM.

### Problems encountered and resolutions
| Problem | Resolution |
|---|---|
| `libcudart.so.12` not found | System has CUDA 13, vLLM built for CUDA 12. Fixed by installing PyTorch with cu128 index. |
| PyTorch installed as CPU-only | vLLM's deps override CUDA torch. Fixed with `--force-reinstall --no-deps`. |
| PaddleOCR OOM at gpu-memory-utilization 0.9 | HunyuanOCR already using GPU memory. Reduced to 0.25. |
| PaddleOCR OOM at gpu-memory-utilization 0.15 | Too low for KV cache after model load. Increased to 0.25. |
| DeepSeek Triton nvrtc error | Blackwell sm_120 unsupported. Pivoted to FastAPI+transformers. |
| DeepSeek `LlamaFlashAttention2` import error | Requires transformers==4.46.3. Created separate conda env. |
| Port 8001 already in use | Moved PaddleOCR to port 8004. |

### Items explicitly completed
- All three OCR model services operational and responding to API requests
- Setup, start, and stop scripts finalized
- Project memory files created (project_ocr_services.md, user_profile.md, feedback_blackwell_compat.md)

### Verification performed
- `curl localhost:8004/v1/models` — PaddleOCR-VL responds with model JSON
- `curl localhost:8002/v1/models` — HunyuanOCR responds with model JSON
- `curl localhost:8003/v1/models` — DeepSeek-OCR-2 responds with model JSON
- No image inference test performed yet
