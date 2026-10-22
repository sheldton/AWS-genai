#!/usr/bin/env python3
"""Execute notebooks on the notebook kernel and report errors and time per cell.

    .venv/bin/python tools/run.py M03            build + execute notebooks/M03_*.ipynb -> notebooks/executed/
    .venv/bin/python tools/run.py M03 --keep     skip the cells tagged "cleanup" (keep resources for a re-run)

Exit code 1 if any cell raised. Cell times are printed so you can see the slow cells.
"""
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

HOME = Path(__file__).resolve().parent.parent
args = [a for a in sys.argv[1:] if not a.startswith("--")]
keep = "--keep" in sys.argv
subprocess.run([sys.executable, str(HOME / "tools" / "build.py"), *args], check=True)
failed = False
for m in args:
    for path in sorted((HOME / "notebooks").glob(f"{m.upper()}_*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        if keep:
            nb.cells = [c for c in nb.cells if "cleanup" not in c.get("metadata", {}).get("tags", [])]
        times = []
        client = NotebookClient(nb, timeout=1800, kernel_name="mlaidp",
                                resources={"metadata": {"path": str(path.parent)}})
        t_all = time.time()
        with client.setup_kernel():
            for i, cell in enumerate(nb.cells):
                if cell.cell_type != "code":
                    continue
                t0 = time.time()
                try:
                    client.execute_cell(cell, i)
                except CellExecutionError as e:
                    failed = True
                    print(f"  ERROR in cell {i}: {str(e).splitlines()[-1][:300]}")
                times.append((i, time.time() - t0, cell.source.splitlines()[0][:70] if cell.source else ""))
        out = HOME / "notebooks" / "executed" / path.name
        nbformat.write(nb, out)
        print(f"{path.name}: {int(time.time() - t_all)} s total -> {out.relative_to(HOME)}")
        for i, s, first in times:
            if s >= 3:
                print(f"  cell {i:3d} {s:7.1f} s  {first}")
sys.exit(1 if failed else 0)
