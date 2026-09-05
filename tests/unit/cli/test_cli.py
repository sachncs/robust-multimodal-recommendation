"""Tests for the morel CLI."""

from __future__ import annotations

import os

from morel.cli import main


def test_cli_help(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["morel"])
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "morel" in captured.out


def test_cli_unknown_command() -> None:
    import pytest

    with pytest.raises(SystemExit):
        main(["nope"])


def test_cli_data_mask(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["morel", "data", "mask", "--items", "5", "--modalities", "3", "--ratio", "0.4", "--out", str(tmp_path / "m.npy")])
    assert main(["data", "mask", "--items", "5", "--modalities", "3", "--ratio", "0.4", "--out", str(tmp_path / "m.npy")]) == 0
    import numpy as np

    assert (tmp_path / "m.npy").exists()
    arr = np.load(tmp_path / "m.npy")
    assert arr.shape == (5, 3)


def test_cli_train_stub() -> None:
    assert main(["train"]) == 0


def test_cli_serve_help() -> None:
    import pytest

    with pytest.raises(SystemExit):
        main(["serve", "--help"])
