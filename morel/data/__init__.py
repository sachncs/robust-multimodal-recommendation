"""Public API for the morel.data package.

Lifecycle: acquire -> validate -> extract -> build -> mask -> store.
Every stage produces artifacts with manifests.
"""

from morel.core.errors import Cfg, DataError
from morel.data.acquire import download, fetch
from morel.data.build import bipartite, cooccurrence, interactions, kcore
from morel.data.extract import (
    FeatureEncoder,
    Random,
    Sentence,
    Vision,
    fingerprint,
    random,
    text,
    visual,
)
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
    exact_interactions,
    review,
    stream_cooc,
    stream_interactions,
)
from morel.data.validate import features, graph
from morel.data.validate import interactions as validate_interactions
from morel.data.validate import mask as check


def build_extractor(name: str, *, dim: int, batch: int = 64, seed: int = 0) -> FeatureEncoder:
    """Build the feature encoder selected by ``config.encoder.{text,visual}``.

    Args:
        name: Encoder name. One of ``"random"``, a sentence-transformers model
            name, or a torchvision model name.
        dim: Output feature dimension.
        batch: Inference batch size (used by NN backbones).
        seed: RNG seed for the random encoder.

    Returns
    -------
        A :class:`FeatureEncoder` instance.

    Raises
    ------
        DataError: If ``name`` is not a known encoder.
    """
    if name == "random":
        return Random(dim, seed=seed)
    if name == "sentence-transformers/all-MiniLM-L6-v2":
        return Sentence(name, batch=batch)
    if name == "resnet50":
        return Vision(name, batch=batch)
    raise Cfg(
        f"unknown feature extractor {name!r}; available: random, "
        f"sentence-transformers/all-MiniLM-L6-v2, resnet50"
    )


#: Map from config name to encoder class for introspection.
EXTRACTORS: dict[str, type[FeatureEncoder]] = {
    "random": Random,
    "sentence-transformers/all-MiniLM-L6-v2": Sentence,
    "resnet50": Vision,
}


def build_mask(kind: str, *, items: int, modalities: int, ratio: float, seed: int) -> Mask:
    """Build the masking strategy selected by ``config.masking.kind``.

    Args:
        kind: Mask name. One of ``"bernoulli"``, ``"block"``.
        items: Number of items to mask.
        modalities: Number of modalities per item.
        ratio: Fraction of modalities to drop.
        seed: RNG seed for reproducibility.

    Returns
    -------
        A :class:`Mask` instance.

    Raises
    ------
        DataError: If ``kind`` is not a known mask name.
    """
    if kind == "bernoulli":
        return bernoulli(items, modalities, ratio, seed=seed)
    if kind == "block":
        if modalities < 2:
            raise DataError(
                f"block masking needs at least 2 modalities, got {modalities}; "
                "every item must keep one observed modality"
            )
        span = min(max(1, round(modalities * ratio)), modalities - 1)
        return block(items, modalities, span, seed=seed)
    raise DataError(f"unknown masking kind {kind!r}; available: bernoulli, block")


#: Map from config name to mask factory for introspection.
MASKS: dict[str, object] = {
    "bernoulli": bernoulli,
    "block": block,
}


__all__ = [
    "EXTRACTORS",
    "MASKS",
    "FeatureEncoder",
    "Manifest",
    "Mask",
    "Random",
    "Sentence",
    "Vision",
    "bernoulli",
    "bipartite",
    "block",
    "build_extractor",
    "build_mask",
    "check",
    "checksum",
    "cooccurrence",
    "download",
    "exact_interactions",
    "features",
    "fetch",
    "fingerprint",
    "graph",
    "interactions",
    "kcore",
    "load_graph",
    "load_manifest",
    "load_npz",
    "manifest_path",
    "random",
    "review",
    "save_graph",
    "save_manifest",
    "save_npz",
    "stack",
    "stream_cooc",
    "stream_interactions",
    "structured",
    "text",
    "validate_interactions",
    "visual",
]
