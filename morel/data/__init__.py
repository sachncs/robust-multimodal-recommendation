"""Public API for the morel.data package.

Lifecycle: acquire -> validate -> extract -> build -> mask -> store.
Every stage produces artifacts with manifests.
"""

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
