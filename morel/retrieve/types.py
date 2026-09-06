"""Type aliases for the retrieval module."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

Strategy = Callable[
    [int, dict[str, np.ndarray], np.ndarray, sp.csr_matrix],
    set[int],
]

StrategyFn = Callable[
    [
        int,
        dict[str, np.ndarray],
        np.ndarray,
        sp.csr_matrix,
    ],
    set[int],
]

__all__ = ["Strategy", "StrategyFn"]
