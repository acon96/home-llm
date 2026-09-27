"""Tests for the llama-cpp-python wheel sourcing and validation helpers in utils.

Covers the switch to upstream (abetlen/llama-cpp-python) release wheels:
  - libc (musl/glibc) detection and wheel-suffix selection
  - upstream wheel URL construction and the libc fallback loop
  - local wheel drop-in support
  - the available-versions listing from the GitHub releases API
  - import validation (including surfacing the real traceback)
"""

import multiprocessing
import os
import sys

import pytest

from custom_components.llama_conversation import utils
from custom_components.llama_conversation.const import (
    EMBEDDED_LLAMA_CPP_PYTHON_VERSION,
    LLAMA_CPP_PYTHON_WHEEL_REPO,
)
from custom_components.llama_conversation.utils import (
    LlamaCppPythonInstallError,
    _load_extension,
    get_available_llama_cpp_versions,
    get_libc,
    get_upstream_wheel_suffixes,
    install_llama_cpp_python,
    validate_llama_cpp_python_installation,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_get_libc(monkeypatch):
    cases = [
        (("musl", "1.2.5"), "musl"),
        (("glibc", "2.36"), "glibc"),
        (("GLIBC", "2.36"), "glibc"),
        (("", ""), ""),
        (("weird", "1"), ""),
    ]
    for libc_value, expected in cases:
        monkeypatch.setattr(utils.platform, "libc_ver", lambda: libc_value)
        assert get_libc() == expected

    def raise_libc_ver():
        raise ValueError("unrecognized configuration name")

    monkeypatch.setattr(utils.platform, "libc_ver", raise_libc_ver)
    assert get_libc() == ""


def test_get_upstream_wheel_suffixes(monkeypatch):
    # musl (HAOS / HA container), arm
    monkeypatch.setattr(utils.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(utils.platform, "libc_ver", lambda: ("musl", "1"))
    assert get_upstream_wheel_suffixes() == ["py3-none-musllinux_1_2_aarch64"]

    # glibc, x86_64
    monkeypatch.setattr(utils.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(utils.platform, "libc_ver", lambda: ("glibc", "2.36"))
    assert get_upstream_wheel_suffixes() == [
        "py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64"
    ]

    # unknown libc: both candidates, musl first (HA is musl-based)
    monkeypatch.setattr(utils.platform, "machine", lambda: "amd64")
    monkeypatch.setattr(utils.platform, "libc_ver", lambda: ("", ""))
    assert get_upstream_wheel_suffixes() == [
        "py3-none-musllinux_1_2_x86_64",
        "py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64",
    ]


def _mock_install(monkeypatch, results):
    """Replace _install_package_with_stderr with a recorder.

    ``results`` is a list of (success, error) tuples consumed one per call.
    Returns the list of captured (package, kwargs) pairs.
    """
    calls = []
    state = {"i": 0}

    def fake_install(package, *, upgrade=True, target=None, constraints=None, timeout=None, reinstall=False):
        calls.append((package, {"upgrade": upgrade, "target": target,
                                "constraints": constraints, "timeout": timeout,
                                "reinstall": reinstall}))
        result = results[min(state["i"], len(results) - 1)]
        state["i"] += 1
        return result

    monkeypatch.setattr(utils, "is_installed", lambda _name: False)
    monkeypatch.setattr(utils, "_install_package_with_stderr", fake_install)
    return calls


def test_install_llama_cpp_python_uses_upstream_musl_wheel(monkeypatch):
    monkeypatch.setattr(utils.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(utils.platform, "libc_ver", lambda: ("musl", "1"))
    calls = _mock_install(monkeypatch, [(True, None)])

    assert install_llama_cpp_python("/tmp/fake-config") is True
    assert len(calls) == 1
    expected_url = (
        f"https://github.com/{LLAMA_CPP_PYTHON_WHEEL_REPO}/releases/download/v{EMBEDDED_LLAMA_CPP_PYTHON_VERSION}"
        f"/llama_cpp_python-{EMBEDDED_LLAMA_CPP_PYTHON_VERSION}-py3-none-musllinux_1_2_aarch64.whl"
    )
    assert calls[0][0] == expected_url
    assert calls[0][1]["reinstall"] is False


def test_install_llama_cpp_python_uses_upstream_glibc_wheel(monkeypatch):
    monkeypatch.setattr(utils.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(utils.platform, "libc_ver", lambda: ("glibc", "2.36"))
    calls = _mock_install(monkeypatch, [(True, None)])

    assert install_llama_cpp_python("/tmp/fake-config") is True
    expected_url = (
        f"https://github.com/{LLAMA_CPP_PYTHON_WHEEL_REPO}/releases/download/v{EMBEDDED_LLAMA_CPP_PYTHON_VERSION}"
        f"/llama_cpp_python-{EMBEDDED_LLAMA_CPP_PYTHON_VERSION}-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
    )
    assert calls[0][0] == expected_url


def test_install_llama_cpp_python_falls_back_to_glibc_wheel_on_unknown_libc(monkeypatch):
    monkeypatch.setattr(utils.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(utils.platform, "libc_ver", lambda: ("", ""))
    calls = _mock_install(monkeypatch, [(False, "musl wheel not supported"), (True, None)])

    assert install_llama_cpp_python("/tmp/fake-config") is True
    assert len(calls) == 2
    assert "musllinux_1_2_x86_64" in calls[0][0]
    assert "manylinux2014_x86_64" in calls[1][0]
    # the second candidate must force a reinstall so uv replaces the first one
    assert calls[0][1]["reinstall"] is False
    assert calls[1][1]["reinstall"] is True


def test_install_llama_cpp_python_specific_version(monkeypatch):
    monkeypatch.setattr(utils.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(utils.platform, "libc_ver", lambda: ("musl", "1"))
    calls = _mock_install(monkeypatch, [(True, None)])

    assert install_llama_cpp_python("/tmp/fake-config", True, "0.3.33") is True
    expected_url = (
        f"https://github.com/{LLAMA_CPP_PYTHON_WHEEL_REPO}/releases/download/v0.3.33"
        "/llama_cpp_python-0.3.33-py3-none-musllinux_1_2_aarch64.whl"
    )
    assert calls[0][0] == expected_url
    assert calls[0][1]["reinstall"] is True


def test_install_llama_cpp_python_specific_forked_version(monkeypatch):
    monkeypatch.setattr(utils.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(utils.platform, "libc_ver", lambda: ("glibc", "2.36"))
    calls = _mock_install(monkeypatch, [(True, None)])

    assert install_llama_cpp_python("/tmp/fake-config", True, "0.3.35+homellm") is True
    expected_url = (
        f"https://github.com/{LLAMA_CPP_PYTHON_WHEEL_REPO}/releases/download/0.3.35"
        "/llama_cpp_python-0.3.35+homellm-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl"
    )
    assert calls[0][0] == expected_url


def test_install_llama_cpp_python_local_wheel(monkeypatch):
    monkeypatch.setattr(utils.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(utils.platform, "libc_ver", lambda: ("musl", "1"))
    calls = _mock_install(monkeypatch, [(True, None)])

    assert install_llama_cpp_python("/tmp/fake-config", True, "llama_cpp_python-0.3.33-py3-none-musllinux_1_2_aarch64.whl") is True
    assert calls[0][0] == os.path.join(os.path.dirname(utils.__file__), "llama_cpp_python-0.3.33-py3-none-musllinux_1_2_aarch64.whl")


def test_install_llama_cpp_python_raises_on_error(monkeypatch):
    monkeypatch.setattr(utils.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(utils.platform, "libc_ver", lambda: ("musl", "1"))
    _mock_install(monkeypatch, [(False, "boom")])

    with pytest.raises(LlamaCppPythonInstallError, match="boom"):
        install_llama_cpp_python("/tmp/fake-config", raise_on_error=True)


def _release(tag, wheel_names):
    return {
        "tag_name": tag,
        "assets": [{"name": name} for name in wheel_names],
    }


def _mock_releases(monkeypatch, releases, status=200):
    class FakeResponse:
        def __init__(self):
            self.status = status

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            return releases

    class FakeSession:
        def __init__(self):
            self.url = None
            self.params = None

        def get(self, url, **kwargs):
            self.url = url
            self.params = kwargs.get("params")
            return FakeResponse()

    session = FakeSession()
    monkeypatch.setattr(
        utils.aiohttp_client, "async_get_clientsession", lambda _hass: session
    )
    return session


def _musl_aarch64_release_payload():
    return [
        _release("v0.3.35", [
            f"llama_cpp_python-0.3.35-py3-none-musllinux_1_2_aarch64.whl",
            f"llama_cpp_python-0.3.35-py3-none-musllinux_1_2_x86_64.whl",
            f"llama_cpp_python-0.3.35.tar.gz",
        ]),
        # GPU variant releases must be skipped
        _release("v0.3.35-cu124", [
            "llama_cpp_python-0.3.35-py3-none-manylinux_2_35_x86_64.whl",
        ]),
        # older version without a wheel for this platform must be skipped
        _release("v0.3.34", [
            "llama_cpp_python-0.3.34-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        ]),
        _release("v0.3.33", [
            "llama_cpp_python-0.3.33-py3-none-musllinux_1_2_aarch64.whl",
        ]),
        # Local-version fork builds are identified by their artifact version, not
        # by the release tag (which may omit the local identifier).
        _release("0.3.35", [
            "llama_cpp_python-0.3.35+homellm-py3-none-musllinux_1_2_aarch64.whl",
        ]),
    ]


@pytest.mark.asyncio
async def test_get_available_llama_cpp_versions_lists_installable_upstream_releases(monkeypatch, hass):
    monkeypatch.setattr(utils.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(utils.platform, "libc_ver", lambda: ("musl", "1"))
    session = _mock_releases(monkeypatch, _musl_aarch64_release_payload())

    versions = await get_available_llama_cpp_versions(hass)

    assert session.url.startswith(
        f"https://api.github.com/repos/{LLAMA_CPP_PYTHON_WHEEL_REPO}/releases"
    )
    assert session.params == {"per_page": 50, "page": 1}
    remote = [version for version, is_local in versions if not is_local]
    assert remote == ["0.3.35+homellm", "0.3.35", "0.3.33"]


@pytest.mark.asyncio
async def test_get_available_llama_cpp_versions_falls_back_on_error(monkeypatch, hass):
    monkeypatch.setattr(utils.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(utils.platform, "libc_ver", lambda: ("musl", "1"))
    _mock_releases(monkeypatch, None, status=500)

    versions = await get_available_llama_cpp_versions(hass)

    remote = [version for version, is_local in versions if not is_local]
    assert remote == [EMBEDDED_LLAMA_CPP_PYTHON_VERSION]


def test_load_extension_reports_traceback_on_broken_import(monkeypatch, tmp_path):
    (tmp_path / "llama_cpp.py").write_text(
        "raise RuntimeError('llama_cpp import intentionally broken for triage test')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    parent_conn, child_conn = multiprocessing.Pipe()
    _load_extension(child_conn)
    result = parent_conn.recv()

    assert "Traceback" in result
    assert "intentionally broken" in result


def test_load_extension_ok_when_not_installed():
    # llama-cpp-python is not installed in the test environment
    parent_conn, child_conn = multiprocessing.Pipe()
    _load_extension(child_conn)
    assert parent_conn.recv() == "ok"


@pytest.mark.asyncio
async def test_validate_llama_cpp_python_installation_spawns_and_passes(monkeypatch, hass):
    # end-to-end: the spawned child must be able to import the integration module
    # and report success for a not-installed (or importable) llama_cpp
    monkeypatch.setenv("PYTHONPATH", REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    validate_llama_cpp_python_installation()


@pytest.mark.asyncio
async def test_validate_llama_cpp_python_installation_config_dir_not_on_sys_path(hass):
    # HA >= 2026.9 launches python with -P (home-assistant/core#180967): the
    # config dir is no longer on the parent's sys.path, and a spawned child
    # inherits exactly that path with a fresh sys.modules. The config dir must
    # be mounted in time for start() to snapshot it into the child, or the
    # child dies with ModuleNotFoundError: No module named 'custom_components'
    # and the validation fails with a bare exit code.
    while REPO_ROOT in sys.path:
        sys.path.remove(REPO_ROOT)
    try:
        validate_llama_cpp_python_installation()
        # the function must not leave the config dir mounted
        assert REPO_ROOT not in sys.path
    finally:
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
