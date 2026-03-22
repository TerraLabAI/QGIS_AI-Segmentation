from src.shared.activation_manager import (
    is_activated, activate_plugin, get_user_email, set_user_email,
    detect_sibling_activation, auto_activate_from_sibling,
    get_newsletter_url,
)


class FakeSettings:
    """In-memory QSettings mock."""
    def __init__(self):
        self._data = {}

    def value(self, key, default="", type=None):
        val = self._data.get(key, default)
        if type is bool:
            return bool(val)
        return val

    def setValue(self, key, value):
        self._data[key] = value


class TestActivationManager:
    def test_not_activated_by_default(self):
        s = FakeSettings()
        assert is_activated("ai-segmentation", s) is False

    def test_activate_with_valid_code(self):
        s = FakeSettings()
        ok, msg = activate_plugin("ai-segmentation", "fromage", s)
        assert ok is True
        assert is_activated("ai-segmentation", s) is True

    def test_activate_case_insensitive(self):
        s = FakeSettings()
        ok, _ = activate_plugin("ai-segmentation", "BAGUETTE", s)
        assert ok is True

    def test_activate_with_invalid_code(self):
        s = FakeSettings()
        ok, msg = activate_plugin("ai-segmentation", "wrong", s)
        assert ok is False
        assert is_activated("ai-segmentation", s) is False

    def test_detect_sibling_from_canvas(self):
        s = FakeSettings()
        activate_plugin("ai-canvas", "fromage", s)
        sibling = detect_sibling_activation("ai-segmentation", s)
        assert sibling == "ai-canvas"

    def test_detect_no_sibling(self):
        s = FakeSettings()
        assert detect_sibling_activation("ai-segmentation", s) is None

    def test_auto_activate_from_sibling(self):
        s = FakeSettings()
        activate_plugin("ai-canvas", "fromage", s)
        result = auto_activate_from_sibling("ai-segmentation", s)
        assert result is True
        assert is_activated("ai-segmentation", s) is True

    def test_get_user_email_from_sibling(self):
        s = FakeSettings()
        set_user_email("ai-canvas", "user@test.com", s)
        email = get_user_email("ai-segmentation", s)
        assert email == "user@test.com"

    def test_newsletter_url(self):
        url = get_newsletter_url("ai-segmentation")
        assert "plugin=ai-segmentation" in url
        assert "terra-lab.ai" in url
