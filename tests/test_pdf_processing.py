from pathlib import Path

from ocr_processor import _add_invisible_text, process_pdf


class TestAddInvisibleText:
    def test_adds_text_to_page(self, sample_pdf):
        import fitz

        doc = fitz.open(str(sample_pdf))
        page = doc[0]
        _add_invisible_text(page, "invisible search text")
        # Extract text to verify it was added
        text = page.get_text()
        assert "invisible search text" in text
        doc.close()

    def test_empty_text_skipped(self, sample_pdf):
        import fitz

        doc = fitz.open(str(sample_pdf))
        page = doc[0]
        original_text = page.get_text()
        _add_invisible_text(page, "   ")
        assert page.get_text() == original_text
        doc.close()


class TestProcessPdf:
    def test_creates_output_file(self, sample_pdf, mock_engine, tmp_path):
        output = tmp_path / "output.pdf"
        result = process_pdf(sample_pdf, output, mock_engine, dpi=72)
        assert result is True
        assert output.exists()

    def test_output_is_valid_pdf(self, sample_pdf, mock_engine, tmp_path):
        import fitz

        output = tmp_path / "output.pdf"
        process_pdf(sample_pdf, output, mock_engine, dpi=72)
        doc = fitz.open(str(output))
        assert len(doc) > 0
        doc.close()

    def test_ocr_text_in_output(self, sample_pdf, mock_engine, tmp_path):
        import fitz

        mock_engine.ocr_image.return_value = "recognized text here"
        output = tmp_path / "output.pdf"
        process_pdf(sample_pdf, output, mock_engine, dpi=72)
        doc = fitz.open(str(output))
        text = doc[0].get_text()
        assert "recognized text here" in text
        doc.close()

    def test_specific_pages_only(self, sample_pdf, mock_engine, tmp_path):
        output = tmp_path / "output.pdf"
        # Process only page 1 (0-based: {0})
        result = process_pdf(sample_pdf, output, mock_engine, dpi=72, pages={0})
        assert result is True
        assert mock_engine.ocr_image.call_count == 1

    def test_nonexistent_input_returns_false(self, mock_engine, tmp_path):
        result = process_pdf(
            Path("/nonexistent/file.pdf"),
            tmp_path / "out.pdf",
            mock_engine,
        )
        assert result is False
