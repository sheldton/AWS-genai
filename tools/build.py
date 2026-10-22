#!/usr/bin/env python3
"""Write notebooks/*.ipynb from the jupytext sources in src/ (percent format).

    .venv/bin/python tools/build.py            all modules
    .venv/bin/python tools/build.py M03 M05    only those
"""
import sys
from pathlib import Path

import jupytext

HOME = Path(__file__).resolve().parent.parent
KERNEL = {"name": "mlaidp", "display_name": "MLAIDP (Python 3.12)", "language": "python"}

picked = [a.upper() for a in sys.argv[1:]]
for src in sorted((HOME / "src").glob("M*.py")):
    if picked and src.name[:3].upper() not in picked:
        continue
    nb = jupytext.read(src)
    nb.metadata["kernelspec"] = KERNEL
    nb.metadata.pop("jupytext", None)
    out = HOME / "notebooks" / (src.stem + ".ipynb")
    jupytext.write(nb, out, fmt="ipynb")
    print("wrote", out.relative_to(HOME), f"({len(nb.cells)} cells)")
