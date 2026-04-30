"""Sanity-check the generated walkthrough notebook structure."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "notebooks" / "walkthrough.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))

cells = nb["cells"]
md = sum(1 for c in cells if c["cell_type"] == "markdown")
co = sum(1 for c in cells if c["cell_type"] == "code")
co_out = sum(1 for c in cells if c["cell_type"] == "code" and c.get("outputs"))
img = sum(
    1
    for c in cells
    if c["cell_type"] == "code"
    for o in c.get("outputs", [])
    if "image/png" in o.get("data", {})
)
size_kb = os.path.getsize(path) // 1024
print(f"Total cells:            {len(cells)}")
print(f"Markdown cells:         {md}")
print(f"Code cells:             {co}")
print(f"Code cells with output: {co_out}")
print(f"Cells with PNG output:  {img}")
print(f"File size:              {size_kb} KB")
