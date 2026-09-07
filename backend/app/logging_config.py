"""One consistent, structured-enough logging setup for the backend process.

No external logging service in this phase -- everything goes to stdout,
same as the rest of this repo's tooling.
"""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:
        # Already configured (e.g. re-entrant create_app() calls in tests) --
        # just make sure the level is current, don't add duplicate handlers.
        root.setLevel(level.upper())
        return
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
