# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities to **sachncs@gmail.com**.

Do **not** open a public GitHub issue for security-related problems.

## Disclosure Process

1. Email a description of the vulnerability and reproduction steps to **sachncs@gmail.com**.
2. The maintainers will acknowledge receipt within 72 hours.
3. A patch will be developed privately and a coordinated disclosure timeline agreed upon.
4. A CVE will be requested if appropriate.

## Supported Versions

| Version | Supported |
|---------|-----------|
| `>=0.8.0` | ✅ Active |
| `<0.8.0` | ❌ End of life |

## Scope

The package:

- Loads remote datasets via HTTPS with optional SHA256 verification.
- Loads Sentence-Transformers and ResNet-50 weights from public model hubs.
- Does **not** load arbitrary pickle files or remote code.

When loading artifacts:

- Checkpoint files use `torch.load(..., weights_only=True)` (default).
- `np.load` is called with `allow_pickle=False`.
- Remote downloads validate SHA256 where manifests are present.

## Out of scope

- Vulnerabilities in upstream libraries (PyTorch, NumPy, Sentence-Transformers, etc.) should be reported upstream.
