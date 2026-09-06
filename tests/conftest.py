"""Global test configuration for home-assistant-llm."""

import sys
import unittest.mock
import warnings

def pytest_configure(config):
    """Install test environment shims before test collection begins.

    This hook runs during pytest's initialization, before collection, so it can
    patch ``sys.modules`` ahead of anything that imports the affected modules.
    """
    _install_probatio_as_voluptuous()
    _mock_turbojpeg()
    _mock_gazetteer_matcher()


def _install_probatio_as_voluptuous():
    """Alias ``voluptuous`` to probatio for Home Assistant >= 2026.9.

    Home Assistant >= 2026.9 replaced its use of ``voluptuous`` with the
    drop-in ``probatio`` package, and dropped ``voluptuous_openapi`` in favour
    of ``probatio.to_openapi``. The integration reflects this with a
    try/except import that only falls back to ``probatio.to_openapi`` when
    ``voluptuous_openapi`` cannot be imported.

    ``probatio.to_openapi`` only converts schemas built by probatio itself, so
    once that path is taken every ``import voluptuous`` in the test process
    (the integration, the tests, and dependencies) must resolve to the probatio
    shim. ``probatio.compat.install_as_voluptuous()`` registers the shim under
    the ``voluptuous`` name. We only do this when ``voluptuous_openapi`` is
    absent, mirroring the integration's own branching, so an older Home
    Assistant install that still imports ``voluptuous_openapi`` keeps its real
    ``voluptuous`` untouched and continues to pass.
    """
    try:
        import voluptuous_openapi  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        # Home Assistant < 2026.9 still uses voluptuous_openapi for conversion.
        return

    try:
        from probatio.compat import install_as_voluptuous
    except ModuleNotFoundError:  # pragma: no cover - defensive
        return

    # In case ``voluptuous`` was already imported by a plugin, suppress the
    # informational "shadowing" RuntimeWarning; the alias still applies.
    warnings.filterwarnings(
        "ignore",
        message="install_as_voluptuous is shadowing an already-imported voluptuous",
        category=RuntimeWarning,
    )
    install_as_voluptuous()


def _mock_turbojpeg():
    """Mock turbojpeg before test collection begins.

    Home Assistant 2026.5.0 imports turbojpeg unconditionally in
    homeassistant/components/camera/img_util.py, which causes import
    errors in test environments where the C library is not available.
    This hook runs during pytest's initialization, before collection.
    """

    sys.modules["turbojpeg"] = unittest.mock.MagicMock()


def _mock_gazetteer_matcher():
    """Mock gazetteer_matcher before test collection begins.

    Home Assistant 2026.9 imports the native ``gazetteer_matcher`` package
    unconditionally in homeassistant/components/conversation/default_agent.py,
    which causes import errors in test environments where the C extension is
    not installed. The project's tests do not exercise conversation gazetteer
    matching, so a mock is sufficient to let Home Assistant import.
    """
    try:
        import gazetteer_matcher  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["gazetteer_matcher"] = unittest.mock.MagicMock()
