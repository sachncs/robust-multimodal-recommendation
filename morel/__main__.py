"""Allow ``python -m morel``."""

from morel.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
