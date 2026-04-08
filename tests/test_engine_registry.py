from unittest.mock import MagicMock, patch

from ocr_processor import ENGINE_REGISTRY, create_engine


class TestEngineRegistry:
    def test_all_engines_registered(self):
        expected = {"paddleocr", "hunyuan", "trocr", "deepseek"}
        assert set(ENGINE_REGISTRY.keys()) == expected

    def test_registry_values_are_classes(self):
        for name, cls in ENGINE_REGISTRY.items():
            assert callable(cls), f"{name} is not callable"


class TestCreateEngine:
    def test_unknown_engine_returns_none(self):
        result = create_engine("nonexistent")
        assert result is None

    def test_failed_init_returns_none(self):
        with patch.dict(ENGINE_REGISTRY, {"broken": MagicMock(side_effect=ImportError("test"))}):
            result = create_engine("broken")
            assert result is None
