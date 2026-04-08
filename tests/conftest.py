from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_engine():
    """Create a mock OCR engine that returns predefined text."""
    engine = MagicMock()
    engine.engine_name = "mock"
    engine.ocr_image.return_value = "Hello World"
    engine.ocr_pdf.return_value = None
    return engine


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a minimal single-page PDF for testing."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Test content")
    pdf_path = tmp_path / "test.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path
