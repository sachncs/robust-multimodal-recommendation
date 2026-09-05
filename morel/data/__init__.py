"""Public API for the morel.data package.

Lifecycle: acquire -> validate -> extract -> build -> mask -> store.
Every stage produces artifacts with manifests.
"""

from morel.data.acquire import download, fetch, download_legacy
from morel.data.build import bipartite, interactions, item_cooccurrence, kcore
from morel.data.extract import FeatureEncoder, fingerprint, random, text, visual
from morel.data.manifest import (
    Manifest,
    checksum,
    load as load_manifest,
    path_for as manifest_path,
    save as save_manifest,
)
from morel.data.mask import Mask, bernoulli, block, stack, structured
from morel.data.store import load_graph, load_npz, save_graph, save_npz
from morel.data.validate import features, graph, interactions as validate_interactions, mask as validate_mask

__all__ = [
    "FeatureEncoder",
    "Manifest",
    "Mask",
    "bernoulli",
    "bipartite",
    "block",
    "checksum",
    "download",
    "download_legacy",
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
    "save_graph",
    "save_manifest",
    "save_npz",
    "stack",
    "structured",
    "text",
    "validate_interactions",
    "validate_mask",
    "visual",
]
