"""``python -m robotsnap.viewer``: run the viewer's command line entry point."""

from __future__ import annotations

import sys

from robotsnap.viewer.run import main

if __name__ == "__main__":
    sys.exit(main())
