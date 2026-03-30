"""Run vector_healer on a PDF page and plot the healed LineStrings."""
import sys, os

# Add repo root so we can import utils.vector_healer, but we need to
# prevent utils/__init__.py from importing PyTorch.  Temporarily replace
# the utils package init before importing.
_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _repo)

# Pre-register a minimal utils package so the real __init__ never runs.
import types
_utils_pkg = types.ModuleType("utils")
_utils_pkg.__path__ = [os.path.join(_repo, "utils")]
_utils_pkg.__package__ = "utils"
sys.modules["utils"] = _utils_pkg

# Stub the data.floorplancad.dataclass_define import (only needed by
# healed_to_svg_data, which we don't call here).
from dataclasses import dataclass
_fake_dc = types.ModuleType("data.floorplancad.dataclass_define")

@dataclass
class _SVGData:
    viewBox: list
    coords: list
    colors: list
    widths: list
    primitive_ids: list
    layer_ids: list
    semantic_ids: list
    instance_ids: list
    primitive_lengths: list

_fake_dc.SVGData = _SVGData
sys.modules["data"] = types.ModuleType("data")
sys.modules["data.floorplancad"] = types.ModuleType("data.floorplancad")
sys.modules["data.floorplancad.dataclass_define"] = _fake_dc

# Now safe to import.
from utils.vector_healer import heal_pdf_vectors

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

PDF_PATH = "C:/Users/tenzi/Downloads/PLANS_Electrical.pdf"
PAGE = 3  # 0-indexed → page 4
OUTPUT_PNG = "C:/Users/tenzi/vecformer/healed_page4.png"

print(f"Extracting and healing vectors from page {PAGE + 1} ...")
primitives = heal_pdf_vectors(PDF_PATH, page_number=PAGE)
print(f"  -> {len(primitives)} healed primitives")
total_segs = sum(len(p.coords) for p in primitives)
print(f"  -> {total_segs} total line segments")

# ── plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(24, 17), dpi=150)
ax.set_aspect("equal")
ax.set_title(f"Healed Vectors — PLANS_Electrical.pdf, Page 4\n"
             f"{len(primitives)} primitives, {total_segs} segments", fontsize=14)

for prim in primitives:
    r, g, b = prim.color
    color = (r / 255.0, g / 255.0, b / 255.0)
    lines = [((s[0], s[1]), (s[2], s[3])) for s in prim.coords]
    lc = LineCollection(lines, colors=[color], linewidths=max(prim.width * 0.3, 0.2))
    ax.add_collection(lc)

ax.autoscale_view()
ax.invert_yaxis()  # PDF coordinates: y grows downward
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
ax.set_xlabel("x (PDF points)")
ax.set_ylabel("y (PDF points)")

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Plot saved to {OUTPUT_PNG}")
plt.close()
