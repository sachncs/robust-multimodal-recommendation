"""Tests for morel.train.loss."""

from __future__ import annotations

import torch

from morel.train.loss import BPR, Composite, Reconstruction, ce


class Checker:
    """Aggregated test methods for this module."""

    def mask(self) -> None:
        # When mask is all-ones, no missing elements -> loss is 0
        pred = {"v": torch.zeros(2, 4), "t": torch.zeros(2, 2)}
        target = {"v": torch.zeros(2, 4), "t": torch.zeros(2, 2)}
        mask = torch.ones(2, 2)
        loss = Reconstruction()
        val = float(loss.forward(pred, target, mask))
        assert abs(val) < 1e-6

    def missing(self) -> None:
        pred = {"v": torch.zeros(2, 4)}
        target = {"v": torch.ones(2, 4)}
        mask = torch.zeros(2, 1)  # modality missing everywhere
        loss = Reconstruction()
        val = float(loss.forward(pred, target, mask))
        assert val > 0

    def shape(self) -> None:
        pos = torch.tensor([0.5, 0.3])
        neg = torch.tensor([0.1, 0.2])
        b = BPR(pos=pos, neg=neg)
        loss = b.forward({}, {}, torch.zeros(1))
        assert loss.dim() == 0

    def components(self) -> None:
        pred = {"v": torch.zeros(2, 4)}
        target = {"v": torch.ones(2, 4)}
        mask = torch.zeros(2, 1)
        components = {"a": Reconstruction()}
        weights = {"a": 2.0}
        comp = Composite(components, weights)
        single = float(Reconstruction().forward(pred, target, mask))
        total = float(comp.forward(pred, target, mask))
        assert abs(total - 2.0 * single) < 1e-5

    def helper(self) -> None:
        logits = torch.tensor([[2.0, 1.0, 0.5]])
        target = torch.tensor([0])
        loss = ce(logits, target)
        assert loss.item() > 0