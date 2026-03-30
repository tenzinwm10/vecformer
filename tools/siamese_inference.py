#!/usr/bin/env python3
"""
Siamese inference: find legend matches in a full floor plan.

Loads the pretrained VecFormer backbone (frozen), passes both a legend
crop and a full plan through it, then runs the SiameseHead cross-
attention to score every plan primitive against the legend query.

Usage
-----
    python tools/siamese_inference.py \
        --checkpoint weights/vecformer_archcad.pth \
        --legend     legend_crop.json \
        --floor_plan floor_plan.json \
        --output     siamese_matches.json \
        --threshold  0.85

Output JSON
-----------
    {
      "matches": [
        {
          "primitive_ids": [12, 13, 14],
          "bbox": [x0, y0, x1, y1],
          "mean_score": 0.93
        },
        ...
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import sys
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch_scatter import scatter

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from data.floorplancad.dataclass_define import SVGData, VecDataTransformArgs
from data.floorplancad.transform_utils import (
    to_tensor,
    norm_coords,
    to_vec_data,
)
from model.vecformer.configuration_vecformer import VecFormerConfig
from model.vecformer.modeling_vecformer import VecFormer
from model.vecformer.siamese_head import VecFormerSiameseHead

# Backbone output dim (dec_channels[0] from default PTv3 config).
_BACKBONE_DIM = 64


# =====================================================================
# Data preparation
# =====================================================================

def _load_json(path: str) -> SVGData:
    with open(path, "r") as f:
        return SVGData(**json.load(f))


def _prepare(svg: SVGData, device: torch.device):
    """Transform raw SVGData through the eval pipeline.

    Returns coords, feats, prim_ids, layer_ids, cu_seqlens — all on
    *device*, plus the raw normalised line-segment coordinates for
    bounding-box extraction.
    """
    eval_args = VecDataTransformArgs(
        random_vertical_flip=0.0,
        random_horizontal_flip=0.0,
        random_rotate=False,
        random_scale=(1.0, 1.0),
        random_translation=(0.0, 0.0),
    )
    td = to_tensor(svg)
    is_line = td.coords.shape[-1] == 4
    if is_line:
        td.coords = td.coords.reshape(-1, 2, 2)
    td.coords = norm_coords(
        td.coords, td.viewBox,
        eval_args.norm_range[0], eval_args.norm_range[1],
    )
    if len(td.coords.shape) == 3:
        td.coords = td.coords.reshape(-1, 4)
    # Keep the full normalised line coords for bbox computation later.
    raw_line_coords = td.coords.clone()  # (N_lines, 4)
    raw_prim_ids = td.primitive_ids.clone()

    vec = to_vec_data(td)
    coords = vec.coords.to(device)
    feats = vec.feats.to(device)
    prim_ids = vec.prim_ids.to(device).int()
    layer_ids = vec.layer_ids.to(device).int()
    n = coords.shape[0]
    cu = torch.tensor([0, n], dtype=torch.int32, device=device)
    return coords, feats, prim_ids, layer_ids, cu, raw_line_coords, raw_prim_ids


# =====================================================================
# Backbone feature extraction
# =====================================================================

@torch.no_grad()
def extract_features(
    model: VecFormer,
    coords: torch.Tensor,
    feats: torch.Tensor,
    prim_ids: torch.Tensor,
    layer_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Run the frozen backbone + LFE → per-primitive features (P, 64)."""
    data_dict = model._get_data_dict(
        coords, feats, cu_seqlens,
        grid_size=0.01,
        prim_ids=prim_ids,
        layer_ids=layer_ids,
        sample_mode=model.config.sample_mode,
    )
    fusion_lids = model.prepare_primitive_layerid(
        prim_ids, layer_ids, cu_seqlens,
    )
    prim_feats, prim_cu = model.backbone(data_dict, cu_seqlens, prim_ids)
    prim_feats = model.lfe(prim_feats, prim_cu, fusion_lids)
    return prim_feats


# =====================================================================
# Post-processing: cluster high-scoring primitives into match groups
# =====================================================================

def _cluster_matches(
    scores: torch.Tensor,
    prim_centres: torch.Tensor,
    threshold: float,
    cluster_radius: float = 0.05,
) -> list[dict]:
    """Group high-scoring primitives into spatially contiguous clusters.

    Uses simple greedy agglomeration: pick the highest-scoring unvisited
    primitive, collect all unvisited primitives within *cluster_radius*
    that also exceed *threshold*, emit a match, repeat.
    """
    mask = scores >= threshold
    if mask.sum() == 0:
        return []

    idxs = torch.where(mask)[0]
    sc = scores[idxs]
    cx = prim_centres[idxs]

    # Sort by descending score.
    order = sc.argsort(descending=True)
    idxs = idxs[order]
    sc = sc[order]
    cx = cx[order]

    visited = torch.zeros(len(idxs), dtype=torch.bool)
    clusters: list[dict] = []

    for i in range(len(idxs)):
        if visited[i]:
            continue
        # Gather neighbours within radius.
        dists = torch.norm(cx - cx[i], dim=-1)
        neighbours = (~visited) & (dists < cluster_radius)
        visited[neighbours] = True

        member_ids = idxs[neighbours].tolist()
        member_scores = sc[neighbours]
        member_centres = cx[neighbours]

        x0 = member_centres[:, 0].min().item()
        y0 = member_centres[:, 1].min().item()
        x1 = member_centres[:, 0].max().item()
        y1 = member_centres[:, 1].max().item()
        # Add a small margin around single-primitive matches.
        margin = cluster_radius * 0.5
        x0 -= margin
        y0 -= margin
        x1 += margin
        y1 += margin

        clusters.append({
            "primitive_ids": member_ids,
            "bbox": [round(x0, 6), round(y0, 6), round(x1, 6), round(y1, 6)],
            "mean_score": round(member_scores.mean().item(), 6),
        })

    return clusters


def _denorm_bbox(bbox, viewBox, norm_range=(-0.5, 0.5)):
    lo, hi = norm_range
    span = hi - lo
    minx, miny, w, h = viewBox
    return [
        round(((bbox[0] - lo) / span) * w + minx, 4),
        round(((bbox[1] - lo) / span) * h + miny, 4),
        round(((bbox[2] - lo) / span) * w + minx, 4),
        round(((bbox[3] - lo) / span) * h + miny, 4),
    ]


# =====================================================================
# Model loading
# =====================================================================

def _load_backbone(
    checkpoint_path: Optional[str],
    device: torch.device,
) -> VecFormer:
    config = VecFormerConfig()
    model = VecFormer(config)

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        cleaned = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)

    model.to(device).eval()
    # Freeze the backbone — we only train the siamese head.
    for p in model.parameters():
        p.requires_grad = False
    return model


def _load_siamese_head(
    checkpoint_path: Optional[str],
    device: torch.device,
) -> VecFormerSiameseHead:
    head = VecFormerSiameseHead(input_dim=_BACKBONE_DIM)

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # Look for siamese head weights inside the checkpoint.
        head_state = {
            k.removeprefix("siamese_head."): v
            for k, v in state.items()
            if k.startswith("siamese_head.")
        }
        if head_state:
            head.load_state_dict(head_state, strict=False)

    return head.to(device).eval()


# =====================================================================
# CLI
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Siamese legend search with VecFormer backbone.",
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Backbone .pth checkpoint (omit for random init)")
    parser.add_argument("--siamese_checkpoint", default=None,
                        help="Siamese head checkpoint (omit for random init)")
    parser.add_argument("--legend", required=True,
                        help="Legend crop JSON (SVGData format)")
    parser.add_argument("--floor_plan", required=True,
                        help="Full floor plan JSON (SVGData format)")
    parser.add_argument("--output", default="siamese_matches.json")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--cluster_radius", type=float, default=0.05,
                        help="Spatial radius for grouping matched primitives")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[siamese] device = {device}")

    # ── load models ──────────────────────────────────────────
    print(f"[siamese] Loading backbone: {args.checkpoint or '(random init)'}")
    model = _load_backbone(args.checkpoint, device)

    print(f"[siamese] Loading siamese head: {args.siamese_checkpoint or '(random init)'}")
    head = _load_siamese_head(args.siamese_checkpoint, device)

    # ── encode legend ────────────────────────────────────────
    print(f"[siamese] Encoding legend: {args.legend}")
    l_svg = _load_json(args.legend)
    l_co, l_fe, l_pi, l_li, l_cu, _, _ = _prepare(l_svg, device)
    legend_tokens = extract_features(model, l_co, l_fe, l_pi, l_li, l_cu)
    print(f"  legend primitives: {legend_tokens.shape[0]}")

    # ── encode floor plan ────────────────────────────────────
    print(f"[siamese] Encoding floor plan: {args.floor_plan}")
    fp_svg = _load_json(args.floor_plan)
    f_co, f_fe, f_pi, f_li, f_cu, raw_coords, raw_pids = _prepare(fp_svg, device)
    plan_tokens = extract_features(model, f_co, f_fe, f_pi, f_li, f_cu)
    print(f"  plan primitives: {plan_tokens.shape[0]}")

    # ── siamese scoring ──────────────────────────────────────
    print(f"[siamese] Running cross-attention scoring ...")
    with torch.no_grad():
        scores = head(legend_tokens, plan_tokens)  # (N_plan_prims,)

    above = (scores >= args.threshold).sum().item()
    print(f"  primitives above {args.threshold}: {above}/{scores.shape[0]}")

    # ── compute primitive centres for clustering ─────────────
    # raw_coords is (N_lines, 4) in normalised space.
    cx = (raw_coords[:, 0] + raw_coords[:, 2]) / 2.0
    cy = (raw_coords[:, 1] + raw_coords[:, 3]) / 2.0
    pcx = scatter(cx, raw_pids.long(), dim=0, reduce="mean")
    pcy = scatter(cy, raw_pids.long(), dim=0, reduce="mean")
    prim_centres = torch.stack([pcx, pcy], dim=-1).to(device)

    # ── cluster into match groups ────────────────────────────
    matches_norm = _cluster_matches(
        scores, prim_centres, args.threshold, args.cluster_radius,
    )

    # ── denormalise bboxes ───────────────────────────────────
    matches = []
    for m in matches_norm:
        matches.append({
            "primitive_ids": m["primitive_ids"],
            "bbox": _denorm_bbox(m["bbox"], fp_svg.viewBox),
            "bbox_normalised": m["bbox"],
            "mean_score": m["mean_score"],
        })

    # ── write output ─────────────────────────────────────────
    result = {
        "legend_file": args.legend,
        "floor_plan_file": args.floor_plan,
        "threshold": args.threshold,
        "total_plan_primitives": int(scores.shape[0]),
        "num_matches": len(matches),
        "matches": matches,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[siamese] {len(matches)} match group(s) → {args.output}")


if __name__ == "__main__":
    main()
