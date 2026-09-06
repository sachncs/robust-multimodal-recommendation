# morel — Paper Fidelity

The fidelity registry lives in `morel.core.fidelity`. Every component declares
its fidelity status, paper reference, implementation location, and the test
that proves the behavior. This document is rendered from the registry by
`morel render-fidelity`.

## Status legend

- **EXACT** — implementation matches the paper definition.
- **APPROXIMATE** — implementation is a faithful interpretation with one or
  more documented deviations.
- **INCORRECT** — known deviation; see deviation note.
- **UNKNOWN** — paper does not specify enough to claim a status.

## Components

The table below is auto-rendered from `morel.core.fidelity.registry` by
`morel render-fidelity`. Re-run `morel render-fidelity docs/FIDELITY.md
docs/FIDELITY.json` after registering new entries.

<!-- FIDELITY:BEGIN -->
<!-- This block is replaced by `morel render-fidelity`. -->
| Component | Status | Paper | Equation | Implementation | Test | Deviation |
|-----------|--------|-------|----------|----------------|------|-----------|
<!-- FIDELITY:END -->

## Deviations

See the deviation column in the auto-rendered table above.
