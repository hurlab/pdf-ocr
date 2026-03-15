# PROJECT_HANDOFF.md

## 1. Project Overview

OCR model serving infrastructure on NVIDIA DGX Spark (GB10 Blackwell, 128GB unified memory). Three OCR vision-language models are served via OpenAI-compatible APIs, accessible remotely over Tailscale (IP: 100.67.76.96) by other machines running `ocr_processor.py`.

- **Last updated:** 2026-03-15 10:41 CDT
- **Last coding CLI used:** Claude Code CLI (claude-opus-4-6)

## 2. Current State

| Component | Status | Notes |
|---|---|---|
| PaddleOCR-VL (port 8004) | **Completed in Session 2026-03-15 10:41 CDT** | vLLM, conda env `ocr`, --enforce-eager --gpu-memory-utilization 0.25 |
| HunyuanOCR (port 8002) | **Completed in Session 2026-03-15 10:41 CDT** | vLLM, conda env `ocr`, --enforce-eager --gpu-memory-utilization 0.2 |
| DeepSeek-OCR-2 (port 8003) | **Completed in Session 2026-03-15 10:41 CDT** | FastAPI+transformers, conda env `ocr-deepseek` (separate due to Triton MoE/Blackwell incompatibility) |
| `setup_ocr.sh` | **Completed in Session 2026-03-15 10:41 CDT** | 5-step setup: two conda envs, vLLM, PyTorch CUDA fix, transformers+FastAPI, model downloads |
| `start_ocr_services.sh` | **Completed in Session 2026-03-15 10:41 CDT** | Starts all 3 services with PID tracking |
| `stop_ocr_services.sh` | **Completed in Session 2026-03-15 10:41 CDT** | Graceful stop with 15s timeout + force kill |
| `deepseek_server.py` | **Completed in Session 2026-03-15 10:41 CDT** | FastAPI wrapper exposing /v1/models and /v1/chat/completions |

## 3. Execution Plan Status

| Phase | Status | Last Updated | Notes |
|---|---|---|---|
| 1. Create start/stop scripts | Completed | 2026-03-15 10:41 CDT | |
| 2. Create setup script | Completed | 2026-03-15 10:41 CDT | Expanded to handle two conda envs |
| 3. Install vLLM + PyTorch CUDA | Completed | 2026-03-15 10:41 CDT | Required force-reinstall with cu128 index |
| 4. Launch PaddleOCR-VL | Completed | 2026-03-15 10:41 CDT | Port changed from 8001→8004 (conflict) |
| 5. Launch HunyuanOCR | Completed | 2026-03-15 10:41 CDT | First model to work |
| 6. Launch DeepSeek-OCR-2 | Completed | 2026-03-15 10:41 CDT | Pivoted from vLLM to FastAPI+transformers |
| 7. Verify all 3 endpoints | Completed | 2026-03-15 10:41 CDT | All respond to /v1/models |

## 4. Outstanding Work

| Item | Status | Last Updated | Reference |
|---|---|---|---|
| End-to-end OCR inference test | Not started | 2026-03-15 10:41 CDT | Session 2026-03-15 — services respond to /v1/models but no image inference test done yet |
| DeepSeek /v1/chat/completions testing | Not started | 2026-03-15 10:41 CDT | Session 2026-03-15 — FastAPI endpoint written but not tested with actual image input |
| Remote machine integration test | Not started | 2026-03-15 10:41 CDT | Session 2026-03-15 — needs testing from the other computer via Tailscale |
| Autostart on boot | Not started | 2026-03-15 10:41 CDT | Session 2026-03-15 — currently manual via start_ocr_services.sh |

## 5. Risks, Open Questions, and Assumptions

| Item | Status | Opened | Notes |
|---|---|---|---|
| Blackwell sm_120 Triton support | Open | 2026-03-15 | Triton 3.6.0 does not support sm_120/sm_121a. When future Triton adds support, DeepSeek can move to vLLM. |
| PyTorch capability warning | Open | 2026-03-15 | PyTorch 2.10.0+cu128 warns about sm_12.1 exceeding max supported 12.0, but works in practice. |
| Port 8001 conflict | Mitigated | 2026-03-15 | Unknown process on port 8001. PaddleOCR moved to 8004. Root cause not investigated. |
| GPU memory sharing | Open | 2026-03-15 | Three models share 120GB GPU. Current allocation: 0.25 (Paddle) + 0.2 (Hunyuan) + DeepSeek (uncontrolled via transformers). May need tuning under load. |
| DeepSeek API compatibility | Open | 2026-03-15 | deepseek_server.py mimics OpenAI API but may not match all fields ocr_processor.py expects. Needs integration testing. |

## 6. Verification Status

| Item | Method | Result | Date |
|---|---|---|---|
| PaddleOCR-VL /v1/models | `curl localhost:8004/v1/models` | Returns model JSON | 2026-03-15 10:41 CDT |
| HunyuanOCR /v1/models | `curl localhost:8002/v1/models` | Returns model JSON | 2026-03-15 10:41 CDT |
| DeepSeek-OCR-2 /v1/models | `curl localhost:8003/v1/models` | Returns model JSON | 2026-03-15 10:41 CDT |
| Image inference (any model) | Not yet verified | — | Needs actual image OCR test |
| Remote Tailscale access | Not yet verified | — | Needs test from another machine |

## 7. Restart Instructions

**Starting point:** All three services are currently running and responding to `/v1/models`. The infrastructure is complete.

**Recommended next actions:**
1. Test actual image OCR inference on each model (send a base64 image to `/v1/chat/completions`)
2. Test remote access from the other machine via `ocr_processor.py --server http://100.67.76.96:<port>`
3. Verify `deepseek_server.py`'s `/v1/chat/completions` endpoint handles image payloads correctly — the `model.infer()` API may differ from what the client expects
4. Optionally investigate what's on port 8001
5. Optionally set up systemd services for autostart on boot

**Key files:**
- `/home/juhur/PROJECTS/OCR/setup_ocr.sh` — one-time setup
- `/home/juhur/PROJECTS/OCR/start_ocr_services.sh` — start all services
- `/home/juhur/PROJECTS/OCR/stop_ocr_services.sh` — stop all services
- `/home/juhur/PROJECTS/OCR/deepseek_server.py` — DeepSeek FastAPI server

- **Last updated:** 2026-03-15 10:41 CDT
