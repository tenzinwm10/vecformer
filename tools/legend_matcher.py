#!/usr/bin/env python3
"""
One-shot legend matching for VecFormer.

Encodes a legend crop and a full floor plan through the pretrained
VecFormer backbone, then sweeps the floor plan with a spatial sliding
window, comparing local cluster embeddings to the legend embedding via
cosine similarity.  Outputs a JSON file of bounding-box matches.

Usage
-----
    python tools/legend_matcher.py \
        --checkpoint  vecformer_archcad.pth \
        --legend      legend_crop.json \
        --floor_plan  floor_plan.json \
        --output      matches.json \
        --threshold   0.90 \
        --window_size 0.15 \
        --stride      0.05

All spatial values (window_size, stride) are expressed as fractions of
the drawing's normalised coordinate range (i.e. 0.15 = 15 % of the
floor plan extent).
"""
from __future__ import annotations

import argparse
import json
import sys
import os

import torch
import torch.nn.functional as F

# ── repo imports ─────────────────────────────────────────────────────
# Ensure the repo root is on sys.path so relative imports work when
# the script is invoked from any working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.floorplancad.dataclass_define import SVGData, VecDataTransformArgs
from data.floorplancad.transform_utils import (
    to_tensor,
    norm_coords,
    to_vec_data,
)
from model.vecformer.configuration_vecformer import VecFormerConfig
from model.vecformer.modeling_vecformer import VecFormer

# ── constants ────────────────────────────────────────────────────────
# Output dim of backbone after PTv3 enc→dec→pool→output_proj.
# Configured by cad_decoder_config["input_dim"] (default 64).
# We project to 512 for the public `get_embedding` API via a learned
# projection that ships with the checkpoint (or a simple linear probe
# when the checkpoint lacks one).
_LATENT_DIM = 512
_BACKBONE_DIM = 64  # dec_channels[0] from default config


# =====================================================================
# 1.  Model loading
# =====================================================================

def _load_model(checkpoint_path: str, device: torch.device) -> VecFormer:
    """Instantiate VecFormer with default config and load *checkpoint_path*."""
    config = VecFormerConfig()
    model = VecFormer(config)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # HuggingFace Trainer checkpoints nest weights under various keys.
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "module" in state:
        state = state["module"]

    # Strip "module." prefix from DDP-wrapped checkpoints.
    cleaned: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        cleaned[k.removeprefix("module.")] = v

    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()
    model.set_inference_mode(True)
    return model


def _build_projection(checkpoint_state: dict, device: torch.device) -> torch.nn.Linear:
    """Return (or create) a Linear that maps backbone_dim → 512.

    If the checkpoint already contains ``embed_proj.weight`` we reload
    it; otherwise we initialise a fresh projection (works for zero-shot
    cosine similarity but benefits from fine-tuning).
    """
    proj = torch.nn.Linear(_BACKBONE_DIM, _LATENT_DIM, bias=False)
    if "embed_proj.weight" in checkpoint_state:
        proj.load_state_dict(
            {"weight": checkpoint_state["embed_proj.weight"]}, strict=True
        )
    else:
        torch.nn.init.orthogonal_(proj.weight)
    return proj.to(device).eval()


# =====================================================================
# 2.  Data preparation  (JSON → tensors ready for the backbone)
# =====================================================================

def _load_json(path: str) -> SVGData:
    with open(path, "r") as f:
        raw = json.load(f)
    return SVGData(**raw)


def _prepare_inputs(svg: SVGData, device: torch.device):
    """Run the same transform chain used by FloorPlanCAD at eval time.

    Returns the dict expected by ``VecFormer._get_data_dict`` plus the
    raw normalised coordinates (needed for spatial windowing).
    """
    eval_args = VecDataTransformArgs(
        random_vertical_flip=0.0,
        random_horizontal_flip=0.0,
        random_rotate=False,
        random_scale=(1.0, 1.0),
        random_translation=(0.0, 0.0),
    )
    tensor_data = to_tensor(svg)

    # Reshape to (N, 2, 2) for norm then back to (N, 4).
    is_line = tensor_data.coords.shape[-1] == 4
    if is_line:
        tensor_data.coords = tensor_data.coords.reshape(-1, 2, 2)

    tensor_data.coords = norm_coords(
        tensor_data.coords, tensor_data.viewBox,
        eval_args.norm_range[0], eval_args.norm_range[1],
    )

    if len(tensor_data.coords.shape) == 3:
        tensor_data.coords = tensor_data.coords.reshape(-1, 4)

    vec = to_vec_data(tensor_data)

    # Move to device.
    coords = vec.coords.to(device)
    feats = vec.feats.to(device)
    prim_ids = vec.prim_ids.to(device).int()
    layer_ids = vec.layer_ids.to(device).int()
    n = coords.shape[0]
    cu_seqlens = torch.tensor([0, n], dtype=torch.int32, device=device)

    return coords, feats, prim_ids, layer_ids, cu_seqlens


# =====================================================================
# 3.  Embedding extraction
# =====================================================================

@torch.no_grad()
def _run_backbone(
    model: VecFormer,
    coords: torch.Tensor,
    feats: torch.Tensor,
    prim_ids: torch.Tensor,
    layer_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Run the VecFormer backbone + LFE and return per-primitive features.

    Returns
    -------
    prim_feats : Tensor, shape (num_primitives, _BACKBONE_DIM)
    """
    from torch_scatter import scatter as _scatter  # noqa: delayed import

    data_dict = model._get_data_dict(
        coords, feats, cu_seqlens,
        grid_size=0.01,
        prim_ids=prim_ids,
        layer_ids=layer_ids,
        sample_mode=model.config.sample_mode,
    )
    fusion_layer_ids = model.prepare_primitive_layerid(
        prim_ids, layer_ids, cu_seqlens,
    )

    prim_feats, prim_cu = model.backbone(data_dict, cu_seqlens, prim_ids)
    prim_feats = model.lfe(prim_feats, prim_cu, fusion_layer_ids)
    return prim_feats


@torch.no_grad()
def get_embedding(
    model: VecFormer,
    proj: torch.nn.Linear,
    coords: torch.Tensor,
    feats: torch.Tensor,
    prim_ids: torch.Tensor,
    layer_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Encode a set of lines into a single 512-dim latent vector.

    1. Backbone → per-primitive features  (num_prims, 64)
    2. Mean-pool over all primitives       (64,)
    3. Linear projection                   (512,)
    4. L2-normalise

    Parameters
    ----------
    model : VecFormer
        Pretrained model (only the backbone + LFE are used).
    proj : nn.Linear
        64 → 512 projection head.
    coords, feats, prim_ids, layer_ids, cu_seqlens :
        Tensors produced by ``_prepare_inputs``.

    Returns
    -------
    Tensor of shape ``(512,)`` – unit-normalised embedding.
    """
    prim_feats = _run_backbone(
        model, coords, feats, prim_ids, layer_ids, cu_seqlens,
    )
    pooled = prim_feats.mean(dim=0)          # (64,)
    latent = proj(pooled)                     # (512,)
    return F.normalize(latent, dim=0)         # unit L2


@torch.no_grad()
def _get_primitive_embeddings(
    model: VecFormer,
    proj: torch.nn.Linear,
    coords: torch.Tensor,
    feats: torch.Tensor,
    prim_ids: torch.Tensor,
    layer_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return projected, normalised per-primitive embeddings + their centres.

    Returns
    -------
    prim_embeddings : (num_prims, 512)   L2-normalised
    prim_centres    : (num_prims, 2)     in normalised coords [-0.5, 0.5]
    """
    from torch_scatter import scatter as _scatter

    prim_feats = _run_backbone(
        model, coords, feats, prim_ids, layer_ids, cu_seqlens,
    )
    # Project and normalise.
    prim_embs = F.normalize(proj(prim_feats), dim=-1)   # (P, 512)

    # Primitive centres from the raw normalised coords.
    # coords is (N, 2) – midpoints of each line segment.
    cx = coords[:, 0]
    cy = coords[:, 1]
    pcx = _scatter(cx, prim_ids.long(), dim=0, reduce="mean")
    pcy = _scatter(cy, prim_ids.long(), dim=0, reduce="mean")
    prim_centres = torch.stack([pcx, pcy], dim=-1)       # (P, 2)

    return prim_embs, prim_centres


# =====================================================================
# 4.  Sliding-window matcher
# =====================================================================

def _sliding_window_match(
    legend_emb: torch.Tensor,
    prim_embs: torch.Tensor,
    prim_centres: torch.Tensor,
    window_size: float,
    stride: float,
    threshold: float,
    norm_range: tuple[float, float] = (-0.5, 0.5),
) -> list[dict]:
    """Sweep a square window across the floor plan and score local clusters.

    For each window position the function:
      1. Selects primitives whose centre falls inside the window.
      2. Mean-pools their embeddings → local embedding.
      3. Computes cosine similarity against *legend_emb*.
      4. Keeps hits above *threshold*.

    Returns a list of ``{"bbox": [x0,y0,x1,y1], "score": float}`` dicts
    in the *normalised* coordinate frame.
    """
    lo, hi = norm_range
    half = window_size / 2.0

    raw_hits: list[dict] = []

    y = lo + half
    while y <= hi - half:
        x = lo + half
        while x <= hi - half:
            x0, y0 = x - half, y - half
            x1, y1 = x + half, y + half

            mask = (
                (prim_centres[:, 0] >= x0)
                & (prim_centres[:, 0] < x1)
                & (prim_centres[:, 1] >= y0)
                & (prim_centres[:, 1] < y1)
            )

            if mask.sum() < 1:
                x += stride
                continue

            local_emb = F.normalize(prim_embs[mask].mean(dim=0), dim=0)
            score = torch.dot(legend_emb, local_emb).item()

            if score >= threshold:
                raw_hits.append({
                    "bbox": [
                        round(x0, 6), round(y0, 6),
                        round(x1, 6), round(y1, 6),
                    ],
                    "score": round(score, 6),
                })

            x += stride
        y += stride

    # ── greedy NMS (IoU-based) to de-duplicate overlapping windows ──
    if not raw_hits:
        return []

    raw_hits.sort(key=lambda h: h["score"], reverse=True)
    kept: list[dict] = []
    for hit in raw_hits:
        if all(_iou(hit["bbox"], k["bbox"]) < 0.5 for k in kept):
            kept.append(hit)
    return kept


def _iou(a: list[float], b: list[float]) -> float:
    """Axis-aligned bounding-box IoU."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# =====================================================================
# 5.  Coordinate de-normalisation
# =====================================================================

def _denorm_bbox(
    bbox: list[float],
    viewBox: list[float],
    norm_range: tuple[float, float] = (-0.5, 0.5),
) -> list[float]:
    """Map a bbox from normalised coords back to original SVG coords."""
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
# 6.  CLI entry point
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot legend matching with VecFormer.",
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to vecformer_archcad.pth")
    parser.add_argument("--legend", required=True,
                        help="Path to legend_crop.json")
    parser.add_argument("--floor_plan", required=True,
                        help="Path to floor_plan.json")
    parser.add_argument("--output", default="matches.json",
                        help="Output JSON path (default: matches.json)")
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="Cosine-similarity threshold (default: 0.90)")
    parser.add_argument("--window_size", type=float, default=0.15,
                        help="Sliding window size as fraction of norm range "
                             "(default: 0.15)")
    parser.add_argument("--stride", type=float, default=0.05,
                        help="Sliding window stride (default: 0.05)")
    parser.add_argument("--device", default=None,
                        help="torch device (default: cuda if available)")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[legend_matcher] device = {device}")

    # ── load model ───────────────────────────────────────────────
    print(f"[legend_matcher] Loading checkpoint: {args.checkpoint}")
    model = _load_model(args.checkpoint, device)

    ckpt_state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt_state:
        ckpt_state = ckpt_state["state_dict"]
    proj = _build_projection(ckpt_state, device)

    # ── encode legend ────────────────────────────────────────────
    print(f"[legend_matcher] Encoding legend: {args.legend}")
    legend_svg = _load_json(args.legend)
    l_coords, l_feats, l_pids, l_lids, l_cu = _prepare_inputs(legend_svg, device)
    legend_emb = get_embedding(model, proj, l_coords, l_feats, l_pids, l_lids, l_cu)

    # ── encode floor plan (per-primitive) ────────────────────────
    print(f"[legend_matcher] Encoding floor plan: {args.floor_plan}")
    fp_svg = _load_json(args.floor_plan)
    f_coords, f_feats, f_pids, f_lids, f_cu = _prepare_inputs(fp_svg, device)
    prim_embs, prim_centres = _get_primitive_embeddings(
        model, proj, f_coords, f_feats, f_pids, f_lids, f_cu,
    )

    # ── sliding window ───────────────────────────────────────────
    print(f"[legend_matcher] Running sliding window "
          f"(size={args.window_size}, stride={args.stride}, thr={args.threshold})")
    matches_norm = _sliding_window_match(
        legend_emb, prim_embs, prim_centres,
        window_size=args.window_size,
        stride=args.stride,
        threshold=args.threshold,
    )

    # ── de-normalise to original SVG coords ──────────────────────
    matches = []
    for m in matches_norm:
        matches.append({
            "bbox": _denorm_bbox(m["bbox"], fp_svg.viewBox),
            "bbox_normalised": m["bbox"],
            "score": m["score"],
        })

    # ── write output ─────────────────────────────────────────────
    result = {
        "legend_file": args.legend,
        "floor_plan_file": args.floor_plan,
        "checkpoint": args.checkpoint,
        "threshold": args.threshold,
        "window_size": args.window_size,
        "stride": args.stride,
        "num_matches": len(matches),
        "matches": matches,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[legend_matcher] {len(matches)} match(es) → {args.output}")


if __name__ == "__main__":
    main()
