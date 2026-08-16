"""Tests for the CLI dependency guard (wrong-interpreter fail-fast)."""

import autocut.cli as cli


def test_missing_transcribe_dep_fails_fast(monkeypatch, capsys, tmp_path):
    # Simulate running under an interpreter without the transcription stack.
    real_find_spec = cli.importlib.util.find_spec

    def fake_find_spec(name, *a, **k):
        if name in ("faster_whisper", "ctranslate2"):
            return None
        return real_find_spec(name, *a, **k)

    monkeypatch.setattr(cli.importlib.util, "find_spec", fake_find_spec)

    rc = cli.main(["--root", str(tmp_path), "all", "ep001"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "faster_whisper" in err
    assert ".venv" in err  # points the user at the project venv


def test_present_deps_do_not_block(tmp_path):
    # A stage with no heavy deps (validate) must pass the guard and reach its
    # normal error path (no EDL in an empty root), not the dependency error.
    rc = cli.main(["--root", str(tmp_path), "validate", "ep001"])
    assert rc == 1  # FileNotFoundError, handled — but not blocked by the guard.


def test_lazy_runtime_import_gets_friendly_hint(monkeypatch, capsys, tmp_path):
    # A stage that lazily imports a known-but-missing runtime dep (e.g. the
    # review server's fastapi) should be translated into the actionable hint.
    # build_parser resolves _cmd_validate by name at call time, so patching the
    # module attribute redirects the dispatched function.
    def boom(ep, args):
        raise ModuleNotFoundError("No module named 'fastapi'", name="fastapi")

    monkeypatch.setattr(cli, "_cmd_validate", boom)

    rc = cli.main(["--root", str(tmp_path), "validate", "ep001"])
    assert rc == 1
    assert "fastapi" in capsys.readouterr().err


def test_unknown_module_error_is_not_swallowed(monkeypatch, tmp_path):
    # A genuine bug (missing a non-runtime module) must surface as a traceback,
    # not be masked by the friendly dependency hint.
    def boom(ep, args):
        raise ModuleNotFoundError("No module named 'totally_internal'", name="totally_internal")

    monkeypatch.setattr(cli, "_cmd_validate", boom)

    try:
        cli.main(["--root", str(tmp_path), "validate", "ep001"])
    except ModuleNotFoundError as e:
        assert e.name == "totally_internal"
    else:
        raise AssertionError("expected the unknown ModuleNotFoundError to propagate")
