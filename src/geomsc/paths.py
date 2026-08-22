"""Where the large assets live.

None of the model checkpoints, datasets or cached activations used by this work are in this
repository; together they are roughly 24 GB, and most are third-party downloads we cannot
redistribute. ``docs/DATA.md`` lists every one of them and where to obtain it.

Scripts locate them through this module, which reads two environment variables:

``GEOMSC_DATA``
    Root for datasets and cached activations. The layout scripts expect underneath it is documented
    in ``docs/DATA.md`` (``hematopoiesis/``, ``pancreas/``, ``branchpoint/``, ``cellcycle/`` and so
    on), and mirrors the directory the original runs used.

``GEOMSC_MODELS``
    Root for model checkpoints (MaxToki, scGPT, Geneformer, STATE, UCE, ESM-2).

Both default to ``./data`` and ``./models`` under the repository, which are gitignored. A script
that needs an asset and cannot find it fails immediately with the path it wanted, rather than part
way through a long run.
"""

from __future__ import annotations

import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

DATA = pathlib.Path(os.environ.get("GEOMSC_DATA", REPO_ROOT / "data"))
MODELS = pathlib.Path(os.environ.get("GEOMSC_MODELS", REPO_ROOT / "models"))


def data(*parts) -> pathlib.Path:
    """Path to a dataset or cached activation under :data:`DATA`."""
    return DATA.joinpath(*parts)


def models(*parts) -> pathlib.Path:
    """Path to a checkpoint under :data:`MODELS`."""
    return MODELS.joinpath(*parts)


def require(path, what: str = "asset") -> pathlib.Path:
    """Return ``path`` if it exists, otherwise fail with an actionable message."""
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"missing {what}: {p}\n"
            f"This repository does not ship large assets. See docs/DATA.md for where to get this "
            f"one, then point GEOMSC_DATA / GEOMSC_MODELS at your copy."
        )
    return p
