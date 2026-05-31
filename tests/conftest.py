"""Global test configuration for home-assistant-llm."""

import sys
import unittest.mock

def pytest_configure(config):
    """Mock turbojpeg before test collection begins.

    Home Assistant 2026.5.0 imports turbojpeg unconditionally in
    homeassistant/components/camera/img_util.py, which causes import
    errors in test environments where the C library is not available.
    This hook runs during pytest's initialization, before collection.
    """

    sys.modules["turbojpeg"] = unittest.mock.MagicMock()
