"""Public API for the morel.data package.

Lifecycle: acquire -> validate -> extract -> build -> mask -> store.
Every stage produces artifacts with manifests.
"""

from morel.core.errors import Cfg, Datum
from morel.data.acquire import download, fetch
from morel.data.build import bipartite, cooccurrence, interactions, kcore
from morel.data.extract import (
    Feature,
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
    locate as manifest_path,
)
from morel.data.manifest import (
    save as save_manifest,
)
from morel.data.mask import Mask, bernoulli, block, stack, structured
from morel.data.store import load_graph, load_npz, save_graph, store
from morel.data.stream import (
    cooc,
    confirmed,
    review,
    stream,
)
from morel.data.validate import features, graph
from morel.data.validate import interactions as validate_interactions
from morel.data.validate import mask as check


def assemble(name: str, *, dim: int, batch: int = 64, seed: int = 0) -> Feature:
    """Build the feature encoder selected by ``config.encoder.{text,visual}``.

    Args:
        name: Encoder name. One of ``"random"``, a sentence-transformers model
            name, or a torchvision model name.
        dim: Output feature dimension.
        batch: Inference batch size (used by NN backbones).
        seed: RNG seed for the random encoder.

    Returns
    -------
        A :class:`Feature` instance.

    Raises
    ------
        Datum: If ``name`` is not a known encoder.
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
EXTRACTORS: dict[str, type[Feature]] = {
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
        Datum: If ``kind`` is not a known mask name.
    """
    if kind == "bernoulli":
        return bernoulli(items, modalities, ratio, seed=seed)
    if kind == "block":
        if modalities < 2:
            raise Datum(
                f"block masking needs at least 2 modalities, got {modalities}; "
                "every item must keep one observed modality"
            )
        span = min(max(1, round(modalities * ratio)), modalities - 1)
        return block(items, modalities, span, seed=seed)
    raise Datum(f"unknown masking kind {kind!r}; available: bernoulli, block")


#: Map from config name to mask factory for introspection.
MASKS: dict[str, object] = {
    "bernoulli": bernoulli,
    "block": block,
}


__all__ = [
    "EXTRACTORS",
    "MASKS",
    "Feature",
    "Manifest",
    "Mask",
    "Random",
    "Sentence",
    "Vision",
    "bernoulli",
    "bipartite",
    "block",
    "assemble",
    "build_mask",
    "check",
    "checksum",
    "cooc",
    "cooccurrence",
    "download",
    "confirmed",
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
    "store",
    "stack",
    "stream",
    "structured",
    "text",
    "validate_interactions",
    "visual",
]
