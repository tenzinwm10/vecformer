"""Diagnostic plots for verifying vector healing quality."""
import sys, os, types

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _repo)

_utils_pkg = types.ModuleType("utils")
_utils_pkg.__path__ = [os.path.join(_repo, "utils")]
_utils_pkg.__package__ = "utils"
sys.modules["utils"] = _utils_pkg

from dataclasses import dataclass
_fake_dc = types.ModuleType("data.floorplancad.dataclass_define")

@dataclass
class _SVGData:
    viewBox: list; coords: list; colors: list; widths: list
    primitive_ids: list; layer_ids: list; semantic_ids: list
    instance_ids: list; primitive_lengths: list

_fake_dc.SVGData = _SVGData
sys.modules["data"] = types.ModuleType("data")
sys.modules["data.floorplancad"] = types.ModuleType("data.floorplancad")
sys.modules["data.floorplancad.dataclass_define"] = _fake_dc

from utils.vector_healer import heal_pdf_vectors, _extract_raw_segments, _group_by_style

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import colorsys

PDF_PATH = "C:/Users/tenzi/Downloads/PLANS_Electrical.pdf"
PAGE = 3
OUTPUT = "C:/Users/tenzi/vecformer/healed_diagnostic.png"

# ── Run healer ───────────────────────────────────────────────────
print("Healing vectors ...")
primitives = heal_pdf_vectors(PDF_PATH, page_number=PAGE)

# ── Also extract raw (unhealed) segments for comparison ──────────
doc = fitz.open(PDF_PATH)
page = doc[PAGE]
raw_segs = _extract_raw_segments(page)
doc.close()

# ── Find interesting crop regions ────────────────────────────────
# Look for clusters of curved primitives (the electrical symbols).
# Pick a region with those yellow circles on the left side.
crops = [
    {"label": "A — Electrical symbols (left)",
     "bounds": (200, 500, 550, 750)},
    {"label": "B — Central detail",
     "bounds": (750, 650, 1100, 950)},
    {"label": "C — Bottom fixtures row",
     "bounds": (500, 1300, 1100, 1500)},
]

# ── Generate distinct colors per primitive ───────────────────────
def distinct_color(i, n):
    hue = (i * 0.618033988749895) % 1.0  # golden ratio spread
    return colorsys.hsv_to_rgb(hue, 0.85, 0.9)

# ── Plot ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(crops), 2, figsize=(20, 8 * len(crops)), dpi=200)
fig.suptitle("Vector Healing Diagnostic — PLANS_Electrical.pdf, Page 4",
             fontsize=16, fontweight="bold", y=0.995)

for row, crop in enumerate(crops):
    x0, y0, x1, y1 = crop["bounds"]
    ax_raw = axes[row, 0]
    ax_heal = axes[row, 1]

    # ── LEFT: Raw (unhealed) segments ────────────────────────
    ax_raw.set_title(f"RAW segments — {crop['label']}", fontsize=11)
    raw_lines = []
    raw_colors = []
    for seg, rgb, w in raw_segs:
        sx1, sy1, sx2, sy2 = seg
        # Check if segment overlaps the crop
        if max(sx1, sx2) < x0 or min(sx1, sx2) > x1:
            continue
        if max(sy1, sy2) < y0 or min(sy1, sy2) > y1:
            continue
        raw_lines.append(((sx1, sy1), (sx2, sy2)))
        raw_colors.append((rgb[0]/255, rgb[1]/255, rgb[2]/255))

    if raw_lines:
        lc = LineCollection(raw_lines, colors=raw_colors, linewidths=0.5)
        ax_raw.add_collection(lc)
    ax_raw.set_xlim(x0, x1)
    ax_raw.set_ylim(y1, y0)  # inverted y
    ax_raw.set_aspect("equal")
    ax_raw.set_facecolor("white")
    ax_raw.set_xlabel("x"); ax_raw.set_ylabel("y")
    ax_raw.text(0.02, 0.02, f"{len(raw_lines)} segments",
                transform=ax_raw.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", fc="lightyellow"))

    # ── RIGHT: Healed primitives, each in a unique color ─────
    ax_heal.set_title(f"HEALED primitives — {crop['label']}", fontsize=11)
    prims_in_crop = []
    for prim in primitives:
        # Check if any segment of this primitive overlaps the crop
        dominated = False
        for seg in prim.coords:
            sx1, sy1, sx2, sy2 = seg
            if max(sx1, sx2) >= x0 and min(sx1, sx2) <= x1 and \
               max(sy1, sy2) >= y0 and min(sy1, sy2) <= y1:
                dominated = True
                break
        if dominated:
            prims_in_crop.append(prim)

    for i, prim in enumerate(prims_in_crop):
        color = distinct_color(i, len(prims_in_crop))
        lines = [((s[0], s[1]), (s[2], s[3])) for s in prim.coords]
        lc = LineCollection(lines, colors=[color], linewidths=1.2)
        ax_heal.add_collection(lc)

    ax_heal.set_xlim(x0, x1)
    ax_heal.set_ylim(y1, y0)
    ax_heal.set_aspect("equal")
    ax_heal.set_facecolor("white")
    ax_heal.set_xlabel("x"); ax_heal.set_ylabel("y")

    total_segs = sum(len(p.coords) for p in prims_in_crop)
    ax_heal.text(0.02, 0.02,
                 f"{len(prims_in_crop)} primitives, {total_segs} segments\n"
                 f"(each primitive = unique color)",
                 transform=ax_heal.transAxes, fontsize=9,
                 bbox=dict(boxstyle="round", fc="lightcyan"))

plt.tight_layout()
plt.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Diagnostic saved to {OUTPUT}")
print(f"  {len(crops)} crop regions, raw vs healed side-by-side")
plt.close()
