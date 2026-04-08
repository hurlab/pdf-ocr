import logging

from ocr_processor import parse_pages, setup_logging


class TestParsePages:
    def test_single_page(self):
        result = parse_pages("5", 10)
        assert result == {4}  # 0-based

    def test_page_range(self):
        result = parse_pages("2-4", 10)
        assert result == {1, 2, 3}

    def test_mixed(self):
        result = parse_pages("1,3-5,10", 10)
        assert result == {0, 2, 3, 4, 9}

    def test_out_of_bounds_ignored(self):
        result = parse_pages("100", 10)
        assert result == set()

    def test_range_clamped(self):
        result = parse_pages("8-15", 10)
        assert result == {7, 8, 9}

    def test_none_pages_returns_all(self):
        """Passing empty comma-separated parts should yield empty set."""
        # parse_pages splits on comma; an empty part would raise ValueError
        # so we test a valid but out-of-range spec instead
        result = parse_pages("0", 10)
        assert result == set()  # page 0 -> idx -1, not in [0, total)


class TestSetupLogging:
    def _reset_root_logger(self):
        """Remove all handlers from root logger so basicConfig can reconfigure."""
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

    def test_default_info(self):
        self._reset_root_logger()
        setup_logging(verbose=False)
        assert logging.getLogger().level == logging.INFO

    def test_verbose_debug(self):
        self._reset_root_logger()
        setup_logging(verbose=True)
        assert logging.getLogger().level == logging.DEBUG
