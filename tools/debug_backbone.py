#!/usr/bin/env python3
"""
Smoke-test for VecFormer: loads the model (from checkpoint if available,
otherwise random init) and runs a dummy forward pass with 128 line
segments to verify that spconv / torch_scatter / flash_attn work
without segfaults.

Usage
-----
    # Random-init smoke test (no weights needed):
    python tools/debug_backbone.py

    # With a real checkpoint:
    python tools/debug_backbone.py --checkpoint weights/vecformer_archcad.pth

    # Force CPU (skip CUDA entirely):
    python tools/debug_backbone.py --device cpu
"""
from __future__ import annotations

import argparse
import sys
import os
import time

# ── repo root on sys.path ────────────────────────────────────────
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch

# ─────────────────────────────────────────────────────────────────
# 1.  Dependency check
# ─────────────────────────────────────────────────────────────────

def _check_deps() -> dict[str, str]:
    """Import every compiled dependency and return version strings."""
    results: dict[str, str] = {}

    # PyTorch
    results["torch"] = torch.__version__
    results["cuda_available"] = str(torch.cuda.is_available())
    if torch.cuda.is_available():
        results["cuda_version"] = torch.version.cuda or "n/a"
        results["gpu"] = torch.cuda.get_device_name(0)

    # spconv
    try:
        import spconv
        results["spconv"] = spconv.__version__
    except ImportError as e:
        results["spconv"] = f"MISSING ({e})"

    # torch_scatter
    try:
        import torch_scatter
        results["torch_scatter"] = "OK"
    except ImportError as e:
        results["torch_scatter"] = f"MISSING ({e})"

    # flash_attn
    try:
        import flash_attn
        results["flash_attn"] = flash_attn.__version__
    except ImportError:
        results["flash_attn"] = "not installed (optional)"

    return results


# ─────────────────────────────────────────────────────────────────
# 2.  Build dummy input (128 line segments, 1 batch)
# ─────────────────────────────────────────────────────────────────

def _make_dummy_input(
    n_segments: int = 128,
    n_primitives: int = 16,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """Synthesise a single-batch input matching FloorPlanCAD collate output.

    Creates *n_segments* line segments grouped into *n_primitives*
    primitives, with coordinates in [-0.5, 0.5] and plausible features.
    """
    torch.manual_seed(42)
    segs_per_prim = n_segments // n_primitives

    # ── coords: (N, 2)  midpoint-style after to_vec_data ─────
    # In VecFormer's forward, coords are the (cx, cy) midpoints of
    # each line segment (see transform_utils.py:168).
    coords = torch.rand(n_segments, 2, device=device) - 0.5

    # ── feats: (N, 7)  [length, dx, dy, cx, cy, pcx, pcy] ───
    # (the 3 colour channels are appended by the projection layer
    #  from the raw 10-dim features, but the backbone's in_channels
    #  config is 7 after to_vec_data splits coords/feats)
    feats = torch.randn(n_segments, 7, device=device) * 0.1

    # ── prim_ids, layer_ids ──────────────────────────────────
    prim_ids = torch.arange(n_primitives, device=device).repeat_interleave(segs_per_prim).int()
    layer_ids = torch.zeros(n_segments, dtype=torch.int32, device=device)

    # ── cu_seqlens (single batch) ────────────────────────────
    cu_seqlens = torch.tensor([0, n_segments], dtype=torch.int32, device=device)

    # ── labels (needed for prepare_targets path) ─────────────
    sem_ids = torch.randint(0, 35, (n_primitives,), device=device).int()
    inst_ids = torch.arange(n_primitives, dtype=torch.int32, device=device)
    prim_lengths = torch.ones(n_primitives, dtype=torch.float32, device=device)
    cu_numprims = torch.tensor([0, n_primitives], dtype=torch.int32, device=device)

    return dict(
        coords=coords,
        feats=feats,
        prim_ids=prim_ids,
        layer_ids=layer_ids,
        cu_seqlens=cu_seqlens,
        sem_ids=sem_ids,
        inst_ids=inst_ids,
        prim_lengths=prim_lengths,
        cu_numprims=cu_numprims,
    )


# ─────────────────────────────────────────────────────────────────
# 3.  Main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="VecFormer backbone smoke test")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pth checkpoint (omit for random init)")
    parser.add_argument("--device", default=None,
                        help="Force device (cpu / cuda / cuda:0)")
    parser.add_argument("--n_segments", type=int, default=128,
                        help="Number of dummy line segments (default 128)")
    args = parser.parse_args()

    # ── dependency report ────────────────────────────────────
    print("=" * 60)
    print("  VecFormer Debug — Dependency Check")
    print("=" * 60)
    deps = _check_deps()
    for k, v in deps.items():
        status = "OK" if "MISSING" not in v else "FAIL"
        print(f"  [{status}] {k:20s} {v}")
    print()

    fatal = [k for k, v in deps.items() if "MISSING" in v]
    if fatal:
        print(f"FATAL: missing dependencies: {fatal}")
        print("Install them before running this test.")
        sys.exit(1)

    # ── device ───────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── build model ──────────────────────────────────────────
    print("\n[1/4] Building VecFormer (default config) ...")
    from model.vecformer.configuration_vecformer import VecFormerConfig
    from model.vecformer.modeling_vecformer import VecFormer

    config = VecFormerConfig()
    model = VecFormer(config)

    if args.checkpoint:
        print(f"[1/4] Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        cleaned = {k.removeprefix("module."): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"       loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")
    else:
        print("[1/4] No checkpoint provided — using random initialisation")

    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"       Model parameters: {n_params:,}")

    # ── dummy input ──────────────────────────────────────────
    print(f"\n[2/4] Creating dummy input ({args.n_segments} line segments, "
          f"{args.n_segments // 8} primitives) ...")
    inputs = _make_dummy_input(n_segments=args.n_segments, device=device)
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"       {k:20s} {str(v.shape):20s} {v.dtype}")

    # ── backbone-only forward ────────────────────────────────
    print("\n[3/4] Running backbone forward pass ...")
    t0 = time.perf_counter()
    with torch.no_grad():
        data_dict = model._get_data_dict(
            inputs["coords"], inputs["feats"], inputs["cu_seqlens"],
            grid_size=0.01,
            prim_ids=inputs["prim_ids"],
            layer_ids=inputs["layer_ids"],
            sample_mode=config.sample_mode,
        )
        fusion_lids = model.prepare_primitive_layerid(
            inputs["prim_ids"], inputs["layer_ids"], inputs["cu_seqlens"],
        )
        prim_feats, prim_cu = model.backbone(
            data_dict, inputs["cu_seqlens"], inputs["prim_ids"],
        )
        prim_feats = model.lfe(prim_feats, prim_cu, fusion_lids)
    t1 = time.perf_counter()
    print(f"       Backbone output:  {prim_feats.shape}  (dtype={prim_feats.dtype})")
    print(f"       Prim cu_seqlens:  {prim_cu.tolist()}")
    print(f"       Time: {(t1 - t0)*1000:.1f} ms")

    # ── full forward (backbone + CAD decoder + loss) ─────────
    print("\n[4/4] Running full forward pass (backbone + decoder + loss) ...")
    model.train()  # need training mode for loss computation
    t0 = time.perf_counter()
    try:
        output = model(**inputs)
        t1 = time.perf_counter()
        print(f"       Loss:   {output.loss.item():.4f}")
        if output.dict_sublosses:
            for name, val in output.dict_sublosses.items():
                print(f"         {name}: {val.item():.4f}")
        print(f"       Time:   {(t1 - t0)*1000:.1f} ms")
    except Exception as e:
        t1 = time.perf_counter()
        print(f"       FAILED after {(t1 - t0)*1000:.1f} ms")
        print(f"       Error: {type(e).__name__}: {e}")
        sys.exit(1)

    # ── success ──────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  ALL CHECKS PASSED — no segfaults, no errors")
    print("=" * 60)
    print()
    print("Summary:")
    print(f"  - Device:           {device}")
    print(f"  - Input segments:   {args.n_segments}")
    print(f"  - Backbone output:  {prim_feats.shape}")
    print(f"  - Full forward:     loss = {output.loss.item():.4f}")
    print(f"  - Checkpoint:       {args.checkpoint or '(random init)'}")
    if not args.checkpoint:
        print()
        print("  NOTE: No pretrained weights were loaded.")
        print("  The VecFormer authors have confirmed that pretrained")
        print("  weights are not publicly available (see GitHub issue #5).")
        print("  You will need to train from scratch using scripts/train.sh")


if __name__ == "__main__":
    main()
