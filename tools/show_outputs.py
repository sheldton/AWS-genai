#!/usr/bin/env python3
"""Print the text of every output of an executed notebook (for reviewing a run without Jupyter).
    .venv/bin/python tools/show_outputs.py M01 [first_cell] [last_cell]"""
import re, sys
from pathlib import Path
import nbformat
HOME = Path(__file__).resolve().parent.parent
path = next((HOME / "notebooks" / "executed").glob(f"{sys.argv[1].upper()}_*.ipynb"))
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
hi = int(sys.argv[3]) if len(sys.argv) > 3 else 10**6
nb = nbformat.read(path, 4)
for i, c in enumerate(nb.cells):
    if c.cell_type != "code" or not lo <= i <= hi:
        continue
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            print(f"--- cell {i} ERROR:", o["ename"], o["evalue"][:400]); continue
        h = o.get("data", {}).get("text/html", "")
        t = o.get("text") or ("" if h else o.get("data", {}).get("text/plain", "")) or ""
        if h:
            t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"<style.*?</style>", "", h, flags=re.S)))
        if "image/png" in o.get("data", {}):
            t += " [image]"
        print(f"--- cell {i}:", t.strip()[:1800])
