"""
PDF vector path extraction and healing for VecFormer ingestion.

Extracts vector paths from PDF pages via PyMuPDF, merges fragmented
line segments (common in "exploded" CAD arc exports) using Shapely,
simplifies geometry to stay within the backbone's per-primitive token
budget, and returns results formatted as VecFormer SVGData-compatible
line segments: list[dict] with coords [x1, y1, x2, y2] per segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

from data.floorplancad.dataclass_define import SVGData


# VecFormer serializes each primitive into a window of line segments.
# The backbone patch size is 1024, but individual primitives should
# stay well below that to avoid dominating a single attention window.
# 128 segments per primitive is a practical upper bound.
MAX_SEGMENTS_PER_PRIMITIVE = 128

# Two segment endpoints closer than this (in PDF points) are treated
# as touching and eligible for merge.
SNAP_TOLERANCE = 0.5


# ── helpers ──────────────────────────────────────────────────────────


@dataclass
class HealedPrimitive:
    """One continuous polyline after healing, stored as VecFormer line segments."""
    coords: list[list[float]]       # N x [x1, y1, x2, y2]
    color: list[int]                # [r, g, b]
    width: float
    source_page: int


def _extract_raw_segments(
    page: fitz.Page,
) -> list[tuple[list[float], list[int], float]]:
    """Return every stroke segment on *page* as ([x1,y1,x2,y2], [r,g,b], width)."""
    segments: list[tuple[list[float], list[int], float]] = []
    paths = page.get_drawings()

    for path in paths:
        color_raw = path.get("color")
        if color_raw is None:
            rgb = [0, 0, 0]
        else:
            rgb = [int(c * 255) for c in color_raw[:3]]

        width = path.get("width", 0.1) or 0.1

        for item in path["items"]:
            kind = item[0]
            if kind == "l":  # line
                p1, p2 = item[1], item[2]
                segments.append(([p1.x, p1.y, p2.x, p2.y], rgb, width))
            elif kind == "c":  # cubic bezier -> linearize
                pts = item[1:]  # 4 fitz.Point control points
                prev = pts[0]
                for t_num in range(1, 9):
                    t = t_num / 8.0
                    t2 = t * t
                    t3 = t2 * t
                    mt = 1 - t
                    mt2 = mt * mt
                    mt3 = mt2 * mt
                    x = (mt3 * pts[0].x + 3 * mt2 * t * pts[1].x
                         + 3 * mt * t2 * pts[2].x + t3 * pts[3].x)
                    y = (mt3 * pts[0].y + 3 * mt2 * t * pts[1].y
                         + 3 * mt * t2 * pts[2].y + t3 * pts[3].y)
                    cur = fitz.Point(x, y)
                    segments.append(([prev.x, prev.y, cur.x, cur.y], rgb, width))
                    prev = cur
            elif kind == "qu":  # quad (4 corners) -> 4 edges
                quad = item[1]  # fitz.Quad with .ul, .ur, .lr, .ll
                corners = [quad.ul, quad.ur, quad.lr, quad.ll]
                for i in range(4):
                    p1 = corners[i]
                    p2 = corners[(i + 1) % 4]
                    segments.append(([p1.x, p1.y, p2.x, p2.y], rgb, width))
            elif kind == "re":  # rectangle
                rect = item[1]  # fitz.Rect
                x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
                for edge in [
                    [x0, y0, x1, y0],
                    [x1, y0, x1, y1],
                    [x1, y1, x0, y1],
                    [x0, y1, x0, y0],
                ]:
                    segments.append((edge, rgb, width))

    return segments


def _group_by_style(
    segments: list[tuple[list[float], list[int], float]],
) -> dict[tuple, list[list[float]]]:
    """Bucket segments by (r, g, b, width) so only same-style lines merge."""
    groups: dict[tuple, list[list[float]]] = {}
    for coords, rgb, width in segments:
        key = (rgb[0], rgb[1], rgb[2], round(width, 4))
        groups.setdefault(key, []).append(coords)
    return groups


def _segments_to_linestrings(
    segments: list[list[float]],
) -> list[LineString]:
    """Convert [x1,y1,x2,y2] segments into Shapely LineStrings."""
    lines: list[LineString] = []
    for seg in segments:
        x1, y1, x2, y2 = seg
        if abs(x2 - x1) > 1e-12 or abs(y2 - y1) > 1e-12:
            lines.append(LineString([(x1, y1), (x2, y2)]))
    return lines


def _snap_and_merge(
    lines: list[LineString],
    tolerance: float,
) -> list[LineString]:
    """Snap nearby endpoints then merge end-to-end segments into polylines."""
    if not lines:
        return []

    # Snap: buffer each line by tolerance, union, then extract centerlines.
    # linemerge handles the topological merge of segments sharing endpoints.
    multi = MultiLineString(lines)
    snapped = multi.buffer(tolerance, cap_style="flat").boundary
    # boundary of a polygon ring is a closed LineString / MultiLineString;
    # fall back to the raw merge if snapping produces unexpected geometry.
    try:
        merged = linemerge(snapped)
    except Exception:
        merged = linemerge(multi)

    # If the snap-based approach collapses geometry, fall back to direct merge.
    if merged.is_empty:
        merged = linemerge(multi)

    if isinstance(merged, LineString):
        return [merged] if not merged.is_empty else []
    if isinstance(merged, MultiLineString):
        return [g for g in merged.geoms if not g.is_empty]

    # Unexpected geometry type -- fall back to unmerged input.
    return lines


def _simplify_linestring(
    ls: LineString,
    max_segments: int,
) -> LineString:
    """Reduce a LineString to at most *max_segments* segments.

    Uses iterative Douglas-Peucker simplification, doubling the
    tolerance until the vertex count is within budget.
    """
    coords = list(ls.coords)
    if len(coords) - 1 <= max_segments:
        return ls

    # Start with a tiny tolerance relative to the bounding extent.
    bounds = ls.bounds  # (minx, miny, maxx, maxy)
    extent = max(bounds[2] - bounds[0], bounds[3] - bounds[1], 1e-6)
    tol = extent * 1e-4

    simplified = ls
    for _ in range(20):
        simplified = ls.simplify(tol, preserve_topology=True)
        if len(simplified.coords) - 1 <= max_segments:
            return simplified
        tol *= 2.0

    # Hard cap: uniformly sample max_segments+1 vertices along the curve.
    total_len = ls.length
    if total_len < 1e-12:
        return LineString([coords[0], coords[-1]])
    step = total_len / max_segments
    sampled = [ls.interpolate(i * step) for i in range(max_segments)]
    sampled.append(ls.interpolate(total_len))
    return LineString([(p.x, p.y) for p in sampled])


def _linestring_to_segments(ls: LineString) -> list[list[float]]:
    """Convert a Shapely LineString to VecFormer [x1,y1,x2,y2] segments."""
    coords = list(ls.coords)
    segs: list[list[float]] = []
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        segs.append([x1, y1, x2, y2])
    return segs


# ── public API ───────────────────────────────────────────────────────


def heal_pdf_vectors(
    pdf_path: str,
    page_number: int = 0,
    snap_tolerance: float = SNAP_TOLERANCE,
    max_segments: int = MAX_SEGMENTS_PER_PRIMITIVE,
) -> list[HealedPrimitive]:
    """Extract and heal vector paths from a PDF page.

    1. Extracts all vector drawing commands (lines, curves, rects) via PyMuPDF.
    2. Groups segments by visual style (color + width).
    3. Snaps nearby endpoints and merges end-to-end segments via ``linemerge``.
    4. Simplifies each merged polyline to at most *max_segments* segments
       so a single primitive stays within the Transformer's token budget.

    Args:
        pdf_path:       Path to the PDF file.
        page_number:    Zero-indexed page to process.
        snap_tolerance: Max distance (PDF points) for endpoint snapping.
        max_segments:   Upper bound on line segments per output primitive.

    Returns:
        List of ``HealedPrimitive``, each containing consecutive
        ``[x1, y1, x2, y2]`` segments ready for the VecFormer pipeline.
    """
    doc = fitz.open(pdf_path)
    if page_number >= len(doc):
        raise IndexError(
            f"Page {page_number} out of range (document has {len(doc)} pages)")
    page = doc[page_number]

    raw = _extract_raw_segments(page)
    if not raw:
        doc.close()
        return []

    groups = _group_by_style(raw)
    primitives: list[HealedPrimitive] = []

    for (r, g, b, width), seg_list in groups.items():
        lines = _segments_to_linestrings(seg_list)
        merged = _snap_and_merge(lines, snap_tolerance)

        for ls in merged:
            ls = _simplify_linestring(ls, max_segments)
            segs = _linestring_to_segments(ls)
            if segs:
                primitives.append(HealedPrimitive(
                    coords=segs,
                    color=[r, g, b],
                    width=width,
                    source_page=page_number,
                ))

    doc.close()
    return primitives


def healed_to_svg_data(
    primitives: list[HealedPrimitive],
    page_rect: Optional[tuple[float, float, float, float]] = None,
) -> SVGData:
    """Convert healed primitives into an ``SVGData`` ready for the VecFormer
    preprocessing / transform pipeline.

    Args:
        primitives:  Output of ``heal_pdf_vectors``.
        page_rect:   (x0, y0, x1, y1) page bounds.  If *None*, computed from
                     the primitives' bounding box.

    Returns:
        ``SVGData`` with coords in ``[x1, y1, x2, y2]`` line mode,
        ``semantic_ids`` and ``instance_ids`` set to the unlabeled defaults
        (35 and -1 respectively).
    """
    all_coords: list[list[float]] = []
    all_colors: list[list[int]] = []
    all_widths: list[float] = []
    all_prim_ids: list[int] = []
    all_layer_ids: list[int] = []
    sem_ids: list[int] = []
    inst_ids: list[int] = []
    prim_lengths: list[float] = []

    for prim_id, prim in enumerate(primitives):
        n_segs = len(prim.coords)
        for seg in prim.coords:
            all_coords.append(seg)
            all_colors.append(prim.color)
            all_widths.append(prim.width)
            all_prim_ids.append(prim_id)
            all_layer_ids.append(0)  # PDF has no CAD layer concept

        # Approximate primitive arc length from segments
        length = 0.0
        for seg in prim.coords:
            dx = seg[2] - seg[0]
            dy = seg[3] - seg[1]
            length += (dx * dx + dy * dy) ** 0.5
        prim_lengths.append(length)
        sem_ids.append(35)   # unlabeled
        inst_ids.append(-1)  # unlabeled

    # Compute viewBox
    if page_rect is not None:
        x0, y0, x1, y1 = page_rect
        view_box = [x0, y0, x1 - x0, y1 - y0]
    elif all_coords:
        import math
        xs = [c for seg in all_coords for c in (seg[0], seg[2])]
        ys = [c for seg in all_coords for c in (seg[1], seg[3])]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        view_box = [x0, y0, x1 - x0 or 1.0, y1 - y0 or 1.0]
    else:
        view_box = [0.0, 0.0, 1.0, 1.0]

    return SVGData(
        viewBox=view_box,
        coords=all_coords,
        colors=all_colors,
        widths=all_widths,
        primitive_ids=all_prim_ids,
        layer_ids=all_layer_ids,
        semantic_ids=sem_ids,
        instance_ids=inst_ids,
        primitive_lengths=prim_lengths,
    )
