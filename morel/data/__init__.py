"""Public API for the morel.data package.

Lifecycle: acquire -> validate -> extract -> build -> mask -> store.
Every stage produces artifacts with manifests.
"""

from morel.core.errors import DataError
from morel.core.registry import Registry
from morel.data.acquire import download, download_legacy, fetch
from morel.data.build import bipartite, interactions, item_cooccurrence, kcore
from morel.data.extract import FeatureEncoder, fingerprint, random, text, visual
from morel.data.manifest import (
    Manifest,
    checksum,
)
from morel.data.manifest import (
    load as load_manifest,
)
from morel.data.manifest import (
    path_for as manifest_path,
)
from morel.data.manifest import (
    save as save_manifest,
)
from morel.data.mask import Mask, bernoulli, block, stack, structured
from morel.data.store import load_graph, load_npz, save_graph, save_npz
from morel.data.stream import (
    exact_two_pass_interactions,
    review_stream,
    streaming_interactions,
    streaming_item_cooccurrence,
)
from morel.data.validate import features, graph
from morel.data.validate import interactions as validate_interactions
from morel.data.validate import mask as validate_mask

#: Selectable masking strategies, keyed by ``config.masking.kind``.
#:
#: A strategy takes ``(items, modalities)`` plus ``ratio`` and ``seed`` and
#: returns a :class:`~morel.data.mask.Mask`. It takes the full argument set
#: even if it ignores part of it, so strategies stay interchangeable.
MASKS: Registry[Mask] = Registry("masking strategy")


@MASKS.register("bernoulli")
def build_bernoulli_mask(*, items: int, modalities: int, ratio: float, seed: int) -> Mask:
    """Mask each (item, modality) pair independently with probability ``ratio``."""
    return bernoulli(items, modalities, ratio, seed=seed)


@MASKS.register("block")
def build_block_mask(*, items: int, modalities: int, ratio: float, seed: int) -> Mask:
    """Mask a contiguous span of modalities per item.

    ``ratio`` is the fraction of an item's modalities to drop, so the knob
    means the same thing as it does for the Bernoulli strategy. The span is
    capped at ``modalities - 1`` because a Mask must leave every item at least
    one observed modality -- with nothing observed there is nothing to
    complete from.

    Raises
    ------
        DataError: If there is only one modality, where block masking could
            only produce fully-unobserved items.
    """
    if modalities < 2:
        raise DataError(
            f"block masking needs at least 2 modalities, got {modalities}; "
            "every item must keep one observed modality"
        )
    span = min(max(1, round(modalities * ratio)), modalities - 1)
    return block(items, modalities, span, seed=seed)


__all__ = [
    "MASKS",
    "FeatureEncoder",
    "Manifest",
    "Mask",
    "bernoulli",
    "bipartite",
    "block",
    "build_bernoulli_mask",
    "build_block_mask",
    "checksum",
    "download",
    "download_legacy",
    "exact_two_pass_interactions",
    "features",
    "fetch",
    "fingerprint",
    "graph",
    "interactions",
    "item_cooccurrence",
    "kcore",
    "load_graph",
    "load_manifest",
    "load_npz",
    "manifest_path",
    "random",
    "review_stream",
    "save_graph",
    "save_manifest",
    "save_npz",
    "stack",
    "streaming_interactions",
    "streaming_item_cooccurrence",
    "structured",
    "text",
    "validate_interactions",
    "validate_mask",
    "visual",
]
