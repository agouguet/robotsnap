"""``python -m robotsnap``: the one entry point, one sub-command per run."""

from robotsnap.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
