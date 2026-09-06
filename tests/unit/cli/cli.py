"""Tests for the morel CLI."""

from __future__ import annotations

from morel.cli import main


class Checker:
    """Aggregated test methods for this module."""

    def cli(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr("sys.argv", ["morel"])
        assert main([]) == 0
        captured = capsys.readouterr()
        assert "morel" in captured.out

    def unknown(self) -> None:
        import pytest

        with pytest.raises(SystemExit):
            main(["nope"])

    def data(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            "sys.argv",
            [
                "morel",
                "data",
                "mask",
                "--items",
                "5",
                "--modalities",
                "3",
                "--ratio",
                "0.4",
                "--out",
                str(tmp_path / "m.npy"),
            ],
        )
        assert (
            main(
                [
                    "data",
                    "mask",
                    "--items",
                    "5",
                    "--modalities",
                    "3",
                    "--ratio",
                    "0.4",
                    "--out",
                    str(tmp_path / "m.npy"),
                ]
            )
            == 0
        )
        import numpy as np

        assert (tmp_path / "m.npy").exists()
        arr = np.load(tmp_path / "m.npy")
        assert arr.shape == (5, 3)

    def train(self) -> None:
        assert main(["train", "completion"]) == 0

    def serve(self) -> None:
        import pytest

        with pytest.raises(SystemExit):
            main(["serve", "--help"])