"""Ablation conditions as configuration transforms.

``config.eval.ablations`` names the conditions an experiment should compare,
but nothing could turn a name into a runnable setup: the names were inert
strings. An ablation is precisely "the same configuration with one component
removed", so each condition is registered here as a function from a ``Config``
to an ablated ``Config``.

Expressing ablations this way means a condition runs through exactly the same
pipeline as the baseline, with only the named component changed. Nothing about
the ablation lives in the model code, so a new condition needs no change to
the pipeline.

Example:
    Registering a project-specific condition::

        from morel.eval.ablation import ABLATIONS

        @ABLATIONS.register("no_text")
        def drop_text(config):
            return config  # ... return a Config with the text modality removed

    Then add ``no_text`` to ``eval.ablations`` and it is included in the sweep.
"""

from __future__ import annotations

from dataclasses import replace

from morel.core.config import Config
from morel.core.registry import Registry

#: Ablation conditions, keyed by the names in ``config.eval.ablations``.
#:
#: A condition takes the baseline ``Config`` and returns the ablated one. It
#: must not mutate its argument; ``Config`` is frozen, so use
#: :func:`dataclasses.replace`.
ABLATIONS: Registry[Config] = Registry("ablation")

#: Name used for the unmodified configuration in an ablation sweep.
BASELINE = "baseline"


@ABLATIONS.register("no_retrieval")
def no_retrieval(config: Config) -> Config:
    """Remove graph retrieval, leaving each item to be completed on its own."""
    return replace(config, retrieve=replace(config.retrieve, kind="none"))


@ABLATIONS.register("no_pe")
def no_pe(config: Config) -> Config:
    """Remove the Laplacian positional encoding by requesting zero dimensions."""
    return replace(config, encode=replace(config.encode, pe=0))


@ABLATIONS.register("no_codebook")
def no_codebook(config: Config) -> Config:
    """Replace the codebook with a pass-through, removing quantization."""
    return replace(config, codebook=replace(config.codebook, kind="identity"))


def ablate(config: Config, name: str) -> Config:
    """Return ``config`` with the named ablation applied.

    Args:
        config: Baseline configuration.
        name: A registered ablation name, or :data:`BASELINE` for no change.

    Returns
    -------
        The ablated configuration. The input is never mutated.

    Raises
    ------
        ConfigError: If ``name`` is not a registered ablation.
    """
    if name == BASELINE:
        return config
    return ABLATIONS.get(name)(config)


def conditions(config: Config) -> tuple[str, ...]:
    """Return the sweep's condition names: the baseline then each ablation."""
    return (BASELINE, *config.eval.ablations)


__all__ = ["ABLATIONS", "BASELINE", "ablate", "conditions", "no_codebook", "no_pe", "no_retrieval"]
