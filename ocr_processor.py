#!/usr/bin/env python3
"""
PDF OCR Processor

Makes scanned PDFs searchable by adding an invisible text layer
using OCR results from multiple vision-language models.

Supported engines:
  paddleocr  - PaddleOCR-VL (0.9B params, native PDF support, CPU/GPU)
  hunyuan    - HunyuanOCR (1B params, GPU required via vLLM)
  trocr      - Microsoft TrOCR (334M params, line-level OCR, CPU/GPU)
  deepseek   - DeepSeek-OCR-2 (3B params, GPU required)

Remote mode (--server):
  Any engine can run on a remote vLLM server via OpenAI-compatible API.
  No local GPU or model dependencies needed.

Usage:
    python ocr_processor.py --engine paddleocr
    python ocr_processor.py --engine paddleocr --server http://gpu-server:8001
    python ocr_processor.py --engine all --input ./input --output ./output
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import fitz
    import numpy as np
    import PIL.Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INVISIBLE_TEXT_FONTSIZE = 3
INVISIBLE_TEXT_FONTSIZE_FALLBACK = 1
INVISIBLE_TEXT_FONTNAME = "helv"
INVISIBLE_TEXT_RENDER_MODE = 3  # invisible: no fill, no stroke
TEXT_LINE_MIN_HEIGHT = 15  # minimum pixel height for TrOCR line detection
BINARY_THRESHOLD = 180  # grayscale to binary threshold for line detection
DEFAULT_DPI = 300
DEFAULT_MAX_TOKENS = 16384
TROCR_MAX_NEW_TOKENS = 256


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Base Engine
# ---------------------------------------------------------------------------


class BaseOCREngine(ABC):
    """Abstract base class for OCR engines."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        pass

    @abstractmethod
    def ocr_image(self, image: "PIL.Image.Image") -> str:
        """Process a PIL Image and return extracted text."""
        pass

    def ocr_pdf(self, pdf_path: str) -> list[str] | None:
        """Process entire PDF directly, returning per-page text.

        Returns None if the engine doesn't support direct PDF processing,
        in which case the caller should fall back to page-by-page image OCR.
        """
        return None


# ---------------------------------------------------------------------------
# PaddleOCR-VL Engine
# ---------------------------------------------------------------------------


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR engine with automatic GPU/CPU selection.

    GPU: Uses PaddleOCR-VL (0.9B VLM, native PDF support, 109 languages).
    CPU: Falls back to PaddleOCR PP-OCRv4 (classic pipeline, fast).
    """

    def __init__(self):
        try:
            import paddle
        except ImportError:
            logger.error(
                "PaddleOCR not installed. Install with:\n"
                "  pip install paddlepaddle-gpu==3.2.1  # or paddlepaddle for CPU\n"
                '  pip install -U "paddleocr[doc-parser]"'
            )
            raise

        self._use_vl = paddle.device.is_compiled_with_cuda()
        self._pipeline = None
        self._classic_ocr = None

        if self._use_vl:
            from paddleocr import PaddleOCRVL

            logger.info("Initializing PaddleOCR-VL pipeline (GPU)...")
            self._pipeline = PaddleOCRVL()
        else:
            from paddleocr import PaddleOCR

            logger.info("No GPU detected. Using PaddleOCR PP-OCRv5 (CPU)...")
            self._classic_ocr = PaddleOCR(lang="en")

    @property
    def engine_name(self) -> str:
        return "PaddleOCR-VL" if self._use_vl else "PaddleOCR (PP-OCRv4)"

    def ocr_image(self, image: "PIL.Image.Image") -> str:
        import numpy as np

        img_array = np.array(image.convert("RGB"))

        if self._use_vl:
            return self._ocr_image_vl(img_array)
        return self._ocr_image_classic(img_array)

    def ocr_pdf(self, pdf_path: str) -> list[str] | None:
        """Process PDF directly (VL mode only)."""
        if not self._use_vl:
            return None  # Classic mode uses page-by-page fallback

        logger.info("  Using PaddleOCR-VL native PDF processing...")
        try:
            output = self._pipeline.predict(pdf_path)
            page_texts = []
            for res in output:
                text = self._extract_text_vl(res)
                page_texts.append(text if text else "")
            return page_texts
        except Exception as e:
            logger.warning(f"  Native PDF processing failed: {e}")
            logger.info("  Falling back to page-by-page image OCR...")
            return None

    def _ocr_image_vl(self, img_array: "np.ndarray") -> str:
        output = self._pipeline.predict(img_array)
        texts = []
        for res in output:
            text = self._extract_text_vl(res)
            if text:
                texts.append(text)
        return "\n".join(texts)

    def _ocr_image_classic(self, img_array: "np.ndarray") -> str:
        result = self._classic_ocr.predict(img_array)
        if not result:
            return ""
        lines = []
        for res in result:
            # PaddleOCR v3.4+ returns dict-like OCRResult objects
            if isinstance(res, dict) or hasattr(res, "keys"):
                rec_texts = res.get("rec_texts", [])
                lines.extend(rec_texts)
            elif isinstance(res, (list, tuple)):
                for item in res:
                    if item and len(item) >= 2:
                        text = item[1][0] if isinstance(item[1], (list, tuple)) else str(item[1])
                        lines.append(text)
        return "\n".join(lines)

    def _extract_text_vl(self, res: object) -> str:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                res.save_to_markdown(save_path=tmpdir)
                md_files = sorted(Path(tmpdir).rglob("*.md"))
                parts = []
                for md_file in md_files:
                    content = md_file.read_text(encoding="utf-8").strip()
                    if content:
                        parts.append(content)
                if parts:
                    return "\n".join(parts)
        except Exception as e:
            logger.debug(f"save_to_markdown failed: {e}")

        if hasattr(res, "rec_texts"):
            return "\n".join(res.rec_texts)
        if hasattr(res, "text"):
            return str(res.text)
        return ""


# ---------------------------------------------------------------------------
# HunyuanOCR Engine
# ---------------------------------------------------------------------------


class HunyuanOCREngine(BaseOCREngine):
    """HunyuanOCR engine (1B params, 100+ languages, requires GPU + vLLM)."""

    def __init__(self):
        try:
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams
        except ImportError:
            logger.error(
                "HunyuanOCR dependencies not installed. Install with:\n"
                '  pip install "vllm>=0.12.0"\n'
                "  pip install transformers Pillow"
            )
            raise

        model_path = "tencent/HunyuanOCR"
        logger.info("Loading HunyuanOCR model (this may take a while)...")
        self.llm = LLM(model=model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.sampling_params = SamplingParams(temperature=0, max_tokens=DEFAULT_MAX_TOKENS)

    @property
    def engine_name(self) -> str:
        return "HunyuanOCR"

    def ocr_image(self, image: "PIL.Image.Image") -> str:
        image = image.convert("RGB")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f, format="PNG")
            tmp_path = f.name

        try:
            prompt_text = (
                "提取文档图片中正文的所有信息用markdown格式表示，"
                "其中页眉、页脚部分忽略，表格用html格式表达，"
                "文档中公式用latex格式表示。"
            )
            messages = [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": tmp_path},
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = {"prompt": prompt, "multi_modal_data": {"image": [image]}}
            output = self.llm.generate([inputs], self.sampling_params)[0]
            text = output.outputs[0].text
            return self._clean_text(text)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove repeated lines (known model artifact)."""
        lines = text.strip().split("\n")
        cleaned = []
        for line in lines:
            if not cleaned or line != cleaned[-1]:
                cleaned.append(line)
        return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# TrOCR Engine (Microsoft)
# ---------------------------------------------------------------------------


class TrOCREngine(BaseOCREngine):
    """Microsoft TrOCR engine (334M params, line-level OCR, CPU/GPU).

    TrOCR is a line-level recognition model. This engine splits each page
    into text lines using projection profiles, then recognizes each line.
    Best for printed text documents.
    """

    def __init__(self, model_variant: str = "microsoft/trocr-base-handwritten"):
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError:
            logger.error(
                "TrOCR dependencies not installed. Install with:\n"
                "  pip install transformers torch torchvision sentencepiece"
            )
            raise

        import torch

        logger.info(f"Loading TrOCR model: {model_variant}...")
        self.processor = TrOCRProcessor.from_pretrained(model_variant)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_variant,
            low_cpu_mem_usage=False,
            _fast_init=False,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.model.eval()
        logger.info(f"  TrOCR loaded on {self.device}")

    @property
    def engine_name(self) -> str:
        return "TrOCR"

    def ocr_image(self, image: "PIL.Image.Image") -> str:
        image = image.convert("RGB")
        lines = self._detect_text_lines(image)

        if not lines:
            # No lines detected; try the whole image as a single line
            lines = [image]

        recognized = []
        for line_img in lines:
            text = self._recognize_line(line_img)
            if text.strip():
                recognized.append(text.strip())

        return "\n".join(recognized)

    def _recognize_line(self, line_image: "PIL.Image.Image") -> str:
        """Recognize text from a single line image."""
        import torch

        pixel_values = self.processor(images=line_image, return_tensors="pt").pixel_values.to(
            self.device
        )

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_new_tokens=TROCR_MAX_NEW_TOKENS)

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text

    @staticmethod
    def _detect_text_lines(
        image: "PIL.Image.Image", min_height: int = TEXT_LINE_MIN_HEIGHT
    ) -> list["PIL.Image.Image"]:
        """Split a page image into text line crops using horizontal projection."""
        import numpy as np

        gray = image.convert("L")
        arr = np.array(gray)
        # Binarize: dark pixels on light background
        binary = (arr < BINARY_THRESHOLD).astype(np.uint8)

        # Horizontal projection profile
        projection = binary.sum(axis=1)
        threshold = max(1, projection.max() * 0.01)

        # Find line boundaries
        in_text = False
        lines = []
        start = 0
        for i, val in enumerate(projection):
            if val > threshold and not in_text:
                start = i
                in_text = True
            elif val <= threshold and in_text:
                height = i - start
                if height >= min_height:
                    lines.append((max(0, start - 3), min(arr.shape[0], i + 3)))
                in_text = False

        if in_text and arr.shape[0] - start >= min_height:
            lines.append((max(0, start - 3), arr.shape[0]))

        return [image.crop((0, y0, image.width, y1)) for y0, y1 in lines]


# ---------------------------------------------------------------------------
# DeepSeek-OCR Engine
# ---------------------------------------------------------------------------


class DeepSeekOCREngine(BaseOCREngine):
    """DeepSeek-OCR-2 engine (3B params, document-level VLM, requires GPU).

    Uses the Transformers backend with the model's built-in infer() method.
    Falls back to a basic generate() approach if infer() is unavailable.
    """

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-OCR-2"):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            logger.error(
                "DeepSeek-OCR dependencies not installed. Install with:\n"
                "  pip install transformers==4.46.3 torch torchvision\n"
                "  pip install flash-attn --no-build-isolation  # optional, for GPU\n"
                "NOTE: DeepSeek-OCR-2 requires transformers<=4.46.x. "
                "Use a separate venv if other engines need newer transformers."
            )
            raise

        logger.info(f"Loading DeepSeek-OCR model: {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Try flash_attention_2 first (GPU), fall back to eager (CPU)
        attn_impl = "eager"
        if torch.cuda.is_available():
            try:
                import flash_attn  # noqa: F401

                attn_impl = "flash_attention_2"
            except ImportError:
                logger.info("  flash-attn not installed, using eager attention")

        self.model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation=attn_impl,
            trust_remote_code=True,
            use_safetensors=True,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.eval().to(self.device)
        if self.device == "cuda":
            self.model = self.model.to(torch.bfloat16)

        logger.info(f"  DeepSeek-OCR loaded on {self.device} (attn: {attn_impl})")

    @property
    def engine_name(self) -> str:
        return "DeepSeek-OCR-2"

    def ocr_image(self, image: "PIL.Image.Image") -> str:
        image = image.convert("RGB")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f, format="PNG")
            tmp_path = f.name

        try:
            prompt = "<image>\n<|grounding|>Convert the document to markdown."

            # Try the model's built-in infer() method first
            if hasattr(self.model, "infer"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    res = self.model.infer(
                        self.tokenizer,
                        prompt=prompt,
                        image_file=tmp_path,
                        output_path=tmpdir,
                        base_size=1024,
                        image_size=768,
                        crop_mode=True,
                        save_results=False,
                    )
                    if isinstance(res, str):
                        return res
                    if isinstance(res, (list, tuple)) and res:
                        return str(res[0])
                    return str(res) if res else ""

            # Fallback: manual tokenization and generation
            return self._generate_fallback(image, prompt)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _generate_fallback(self, image: "PIL.Image.Image", prompt: str) -> str:
        """Fallback generation using standard transformers generate()."""
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=8192, do_sample=False)

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Remove the prompt from the output
        if prompt in text:
            text = text[text.index(prompt) + len(prompt) :]
        return text.strip()


# ---------------------------------------------------------------------------
# vLLM API Engine (remote server)
# ---------------------------------------------------------------------------

# Default OCR prompts per engine (used when calling remote vLLM servers)
ENGINE_PROMPTS = {
    "paddleocr": "OCR this document image. Extract all text content in reading order.",
    "hunyuan": (
        "提取文档图片中正文的所有信息用markdown格式表示，"
        "其中页眉、页脚部分忽略，表格用html格式表达，"
        "文档中公式用latex格式表示。"
    ),
    "deepseek": "<image>\n<|grounding|>Convert the document to markdown.",
    "trocr": "OCR this document image. Extract all text content in reading order.",
}


class VLLMApiEngine(BaseOCREngine):
    """OCR engine that calls a remote vLLM server via OpenAI-compatible API.

    Works with any VLM served by vLLM (PaddleOCR-VL, HunyuanOCR, DeepSeek-OCR, etc.).
    No local GPU or model dependencies needed - just sends images over HTTP.
    """

    def __init__(self, server_url: str, engine_key: str = ""):
        import requests

        self._requests = requests
        self.server_url = server_url.rstrip("/")
        self._engine_key = engine_key
        self._prompt = ENGINE_PROMPTS.get(engine_key, ENGINE_PROMPTS["paddleocr"])

        # Auto-detect model name from the server
        logger.info(f"Connecting to vLLM server: {self.server_url}")
        try:
            resp = requests.get(f"{self.server_url}/v1/models", timeout=10)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            if models:
                self._model_name = models[0]["id"]
                logger.info(f"  Model detected: {self._model_name}")
            else:
                raise RuntimeError("No models found on server")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to vLLM server at {self.server_url}: {e}")

    @property
    def engine_name(self) -> str:
        return f"vLLM API ({self._model_name})"

    def ocr_image(self, image: "PIL.Image.Image") -> str:
        import base64

        image = image.convert("RGB")

        # Encode image to base64 PNG
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        payload = {
            "model": self._model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": self._prompt},
                    ],
                }
            ],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": 0,
        }

        resp = self._requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Engine Factory
# ---------------------------------------------------------------------------

ENGINE_REGISTRY = {
    "paddleocr": PaddleOCREngine,
    "hunyuan": HunyuanOCREngine,
    "trocr": TrOCREngine,
    "deepseek": DeepSeekOCREngine,
}


def create_engine(engine_name: str) -> BaseOCREngine | None:
    """Create an engine instance. Returns None if initialization fails."""
    if engine_name not in ENGINE_REGISTRY:
        logger.error(
            f"Unknown engine: {engine_name}. Available: {', '.join(ENGINE_REGISTRY.keys())}"
        )
        return None
    try:
        return ENGINE_REGISTRY[engine_name]()
    except Exception as e:
        logger.warning(f"Engine '{engine_name}' failed to initialize: {e}")
        return None


# ---------------------------------------------------------------------------
# PDF Processing
# ---------------------------------------------------------------------------


def parse_pages(pages_str: str, total: int) -> set[int]:
    """Parse page spec like '5,10-15,55' into a set of 0-based page indices."""
    result = set()
    for part in pages_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                start = max(1, int(a))
                end = min(total, int(b))
            except ValueError:
                logger.warning(f"  Ignoring invalid page range: '{part}'")
                continue
            result.update(range(start - 1, end))
        else:
            try:
                idx = int(part) - 1
            except ValueError:
                logger.warning(f"  Ignoring invalid page spec: '{part}'")
                continue
            if 0 <= idx < total:
                result.add(idx)
    return result


def _add_invisible_text(page: "fitz.Page", text: str) -> None:
    """Add invisible text layer to a PDF page for searchability."""
    import fitz

    if not text.strip():
        return

    text_rect = page.rect + fitz.Rect(2, 2, -2, -2)
    rc = page.insert_textbox(
        text_rect,
        text,
        fontsize=INVISIBLE_TEXT_FONTSIZE,
        fontname=INVISIBLE_TEXT_FONTNAME,
        render_mode=INVISIBLE_TEXT_RENDER_MODE,
        overlay=True,
    )
    # If text didn't fit, retry with smaller font
    if rc < 0:
        page.clean_contents()
        page.insert_textbox(
            text_rect,
            text,
            fontsize=INVISIBLE_TEXT_FONTSIZE_FALLBACK,
            fontname=INVISIBLE_TEXT_FONTNAME,
            render_mode=INVISIBLE_TEXT_RENDER_MODE,
            overlay=True,
        )


def process_pdf(
    input_path: Path,
    output_path: Path,
    engine: BaseOCREngine,
    dpi: int = DEFAULT_DPI,
    pages: set[int] | None = None,
) -> bool:
    """Process a single PDF: add invisible OCR text layer to make it searchable.

    Args:
        pages: If set, only process these 0-based page indices.
               If output_path exists, it is used as the base document
               and only the specified pages are re-OCR'd (update mode).
    """
    import fitz
    from PIL import Image

    logger.info(f"Processing: {input_path.name}")

    try:
        input_doc = fitz.open(str(input_path))
    except Exception as e:
        logger.error(f"Failed to open {input_path}: {e}")
        return False

    total_pages = len(input_doc)

    # Update mode: if output exists and specific pages requested,
    # work on the existing output and only redo those pages.
    update_mode = pages is not None and output_path.exists()
    if update_mode:
        doc = fitz.open(str(output_path))
        page_label = ",".join(str(p + 1) for p in sorted(pages))
        logger.info(f"  Update mode: re-processing page(s) {page_label} in {output_path.name}")
    else:
        doc = input_doc
        input_doc = None  # same object

    if pages is not None:
        logger.info(f"  Pages to process: {len(pages)} of {total_pages}")
    else:
        logger.info(f"  Total pages: {total_pages}")

    # Page-by-page image OCR (always use image path for targeted pages)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    # Source for rendering is always the original input
    render_doc = fitz.open(str(input_path)) if update_mode else doc

    for page_idx in range(total_pages):
        page_num = page_idx + 1

        # Skip pages not in the target set
        if pages is not None and page_idx not in pages:
            continue

        # In update mode, replace the page from the input first
        if update_mode:
            doc.delete_page(page_idx)
            doc.insert_pdf(
                fitz.open(str(input_path)), from_page=page_idx, to_page=page_idx, start_at=page_idx
            )

        page = doc[page_idx]

        logger.info(f"  [{page_num}/{total_pages}] Rendering at {dpi} DPI...")
        render_page = render_doc[page_idx]
        pix = render_page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data)).convert("RGB")

        logger.info(f"  [{page_num}/{total_pages}] Running OCR ({engine.engine_name})...")
        try:
            text = engine.ocr_image(image)
        except Exception as e:
            logger.warning(f"  [{page_num}/{total_pages}] OCR failed: {e}")
            text = ""

        del image, pix, img_data

        if text.strip():
            _add_invisible_text(page, text)
            logger.info(f"  [{page_num}/{total_pages}] Text layer added ({len(text)} chars)")
        else:
            logger.info(f"  [{page_num}/{total_pages}] No text detected, skipping.")

    if update_mode:
        render_doc.close()

    # Save searchable PDF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path), garbage=4, deflate=True)
    doc.close()
    if input_doc is not None:
        input_doc.close()

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Saved: {output_path} ({file_size_mb:.1f} MB)")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make scanned PDFs searchable using open-weight OCR models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Engines:
  paddleocr  PaddleOCR-VL  (0.9B)  CPU/GPU  native PDF support
  hunyuan    HunyuanOCR    (1B)    GPU      vLLM required
  trocr      Microsoft TrOCR (334M) CPU/GPU  line-level OCR
  deepseek   DeepSeek-OCR-2 (3B)   GPU      document-level VLM
  all        Run ALL engines, output files named {stem}_{engine}.pdf

Remote mode (--server):
  Point to a vLLM server - no local GPU needed.
  Start server: vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code

Examples:
  %(prog)s --engine paddleocr
  %(prog)s --engine paddleocr --server http://gpu-server:8001
  %(prog)s --engine all                    # compare all engines
  %(prog)s --pages 55 --server http://gpu-server:8001  # reprocess page 55
  %(prog)s --pages 10-20,55  --server http://gpu-server:8001
""",
    )
    parser.add_argument(
        "--engine",
        choices=[*ENGINE_REGISTRY.keys(), "all"],
        default="paddleocr",
        help="OCR engine to use, or 'all' to run every engine (default: paddleocr)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("input"),
        help="Input directory containing PDF files (default: ./input)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory for searchable PDFs (default: ./output)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="DPI for page rendering (higher = better OCR, more memory) (default: 300)",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Pages to process: '55' or '10-20,55'. If output exists, updates only those pages.",
    )
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        help="vLLM server URL for remote OCR (e.g., http://gpu-server:8001)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Validate --server + --engine all combination
    if args.server and args.engine == "all":
        logger.error(
            "--server cannot be used with --engine all.\n"
            "Run separately per engine:\n"
            f"  {sys.argv[0]} --engine paddleocr --server http://gpu-server:8001\n"
            f"  {sys.argv[0]} --engine hunyuan   --server http://gpu-server:8002"
        )
        sys.exit(1)

    if not args.input.is_dir():
        logger.error(f"Input directory not found: {args.input}")
        sys.exit(1)

    pdf_files = sorted(args.input.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {args.input}")
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDF file(s) in {args.input}")

    # Determine which engines to run
    if args.engine == "all":
        engine_keys = list(ENGINE_REGISTRY.keys())
    else:
        engine_keys = [args.engine]

    mode = f"server={args.server}" if args.server else "local"
    logger.info(
        f"Engine(s): {', '.join(engine_keys)} | Mode: {mode} | "
        f"DPI: {args.dpi} | Output: {args.output}"
    )

    total_runs = 0
    total_success = 0

    for engine_key in engine_keys:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Initializing engine: {engine_key}")
        logger.info(f"{'=' * 60}")

        if args.server:
            # Remote mode: use vLLM API
            try:
                engine = VLLMApiEngine(args.server, engine_key=engine_key)
            except Exception as e:
                logger.warning(f"Failed to connect to server: {e}")
                continue
        else:
            # Local mode: load model locally
            engine = create_engine(engine_key)
            if engine is None:
                logger.warning(f"Skipping engine '{engine_key}' (failed to initialize).")
                continue

        for pdf_file in pdf_files:
            total_runs += 1
            # Always include engine name in output filename
            output_file = args.output / f"{pdf_file.stem}_{engine_key}.pdf"

            # Parse --pages (needs total page count from the PDF)
            target_pages = None
            if args.pages:
                import fitz as _fitz

                _tmp = _fitz.open(str(pdf_file))
                target_pages = parse_pages(args.pages, len(_tmp))
                _tmp.close()

            if process_pdf(pdf_file, output_file, engine, dpi=args.dpi, pages=target_pages):
                total_success += 1

        # Free engine memory before loading the next one
        del engine

    logger.info(f"\nDone: {total_success}/{total_runs} runs completed successfully.")
    if total_success == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
