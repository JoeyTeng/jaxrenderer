from collections.abc import Callable
import sys
from types import ModuleType

import pytest
from tools import kaggle_cli


def _mock_cli_module(main: Callable[[], None]) -> ModuleType:
    module = ModuleType("kaggle.cli")
    setattr(module, "main", main)
    return module


def test_import_output_goes_to_stderr_and_cli_stdout_stays_structured(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    original_argv = ["kaggle_cli.py", "--no-warn", "quota", "--format", "json"]
    monkeypatch.setattr(sys, "argv", original_argv.copy())

    def cli_main() -> None:
        assert sys.argv == ["kaggle", *original_argv[1:]]
        _ = print('[{"resource":"TPU","remaining":"1.00h"}]')

    def fake_import(module_name: str) -> ModuleType:
        assert module_name == "kaggle.cli"
        assert sys.argv[0] == "kaggle"
        assert sys.argv[1:] == original_argv[1:]
        _ = print("mock authentication and version notices")
        return _mock_cli_module(cli_main)

    monkeypatch.setattr(kaggle_cli.importlib, "import_module", fake_import)
    kaggle_cli.main()

    captured = capsys.readouterr()
    assert captured.out == '[{"resource":"TPU","remaining":"1.00h"}]\n'
    assert captured.err == "mock authentication and version notices\n"
    assert sys.argv == original_argv


def test_cli_exit_status_propagates_after_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_argv = ["kaggle_cli.py", "--no-warn", "quota", "--format", "json"]
    monkeypatch.setattr(sys, "argv", original_argv.copy())

    def cli_main() -> None:
        assert sys.argv == ["kaggle", *original_argv[1:]]
        raise SystemExit(7)

    def fake_import(module_name: str) -> ModuleType:
        assert module_name == "kaggle.cli"
        return _mock_cli_module(cli_main)

    monkeypatch.setattr(kaggle_cli.importlib, "import_module", fake_import)
    with pytest.raises(SystemExit) as raised:
        kaggle_cli.main()
    assert raised.value.code == 7
    assert sys.argv == original_argv


def test_import_failure_restores_argv_and_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    original_argv = ["kaggle_cli.py", "--no-warn", "quota", "--format", "json"]
    monkeypatch.setattr(sys, "argv", original_argv.copy())

    def failing_import(module_name: str) -> ModuleType:
        assert module_name == "kaggle.cli"
        assert sys.argv == ["kaggle", *original_argv[1:]]
        _ = print("partial import notice")
        raise RuntimeError("mock import failure")

    monkeypatch.setattr(kaggle_cli.importlib, "import_module", failing_import)
    with pytest.raises(RuntimeError, match="mock import failure"):
        kaggle_cli.main()

    captured = capsys.readouterr()
    assert sys.argv == original_argv
    assert captured.out == ""
    assert captured.err == "partial import notice\n"
