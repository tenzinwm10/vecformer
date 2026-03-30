# CLAUDE.md - VecFormer Codebase Guide

> **Paper:** "Point or Line? Using Line-based Representation for Panoptic Symbol Spotting in CAD Drawings" (NeurIPS 2025, arXiv 2505.23395)

## Project Overview

VecFormer performs **panoptic symbol spotting** on CAD floorplans. It jointly predicts semantic segmentation (35 classes: 30 "thing" + 5 "stuff") and instance segmentation over vectorized line primitives, outputting a unified panoptic segmentation. The key insight is representing CAD drawings as sets of **line segments** rather than points or rasterized images.

---

## Architecture

### High-Level Pipeline

```
Raw SVG/JSON  -->  Preprocess (line sampling)  -->  VecData (N x [x1,y1,x2,y2] + features)
                                                         |
                                                    VecBackbone (PTv3-based encoder/decoder)
                                                         |
                                                    Primitive features (per-primitive embeddings)
                                                         |
                                               Layer Fusion Enhancement (LFE)
                                                         |
                                                    CAD Decoder (cross-attention with learnable queries)
                                                         |
                                              Instance masks + Semantic labels + Panoptic output
```

### Line-based Transformer Backbone (`model/vecformer/vec_backbone/`)

The backbone wraps **Point Transformer V3** (PTv3) and adapts it for line-segment inputs.

**Input representation (line mode):**
- Coordinates: `(N, 4)` as `[x1, y1, x2, y2]` per line segment, scaled to `[-1000, 1000]`
- Features: `(N, 10)` as `[length, |dx|, |dy|, center_x, center_y, prim_center_x, prim_center_y, r, g, b]`
- The 7 "non-geometric" features are projected to embedding dim via a learned `Projection` layer; 3 coord-derived features are handled separately

**Positional encoding (dual scheme):**
- **APE (Absolute Position Embedding):** Sinusoidal encoding of coordinates with learnable magnitudes, added to features before transformer blocks (`modules/abs_pos_embed.py`)
- **RoPE (Rotary Position Embedding):** Applied to Q/K inside each self-attention layer (`modules/attention.py:72-113`), 4D, theta=10000.0

**Encoder** (`vec_backbone/vec_encoder.py`):
- 5 stages, depths `(2, 2, 2, 6, 2)`, channels `(32, 64, 128, 256, 512)`, heads `(2, 4, 8, 16, 32)`
- Patch (context window) size: 1024 at all stages
- Stride 2 downsampling between stages
- **Primitive Fusion** after each stage: `GroupFeatFusion` aggregates line-segment features belonging to the same primitive (max+mean pool, then broadcast back)
- **Layer Fusion** on last encoder stage: aggregates primitives sharing the same CAD layer ID

**Decoder** (`vec_backbone/vec_decoder.py`):
- 4 symmetric upsampling stages, channels `(64, 64, 128, 256)`, heads `(4, 4, 8, 16)`
- Mirrors the encoder with skip connections

**Serialization** (`point_transformer_v3/serialization/`):
- Space-filling curves order the unstructured line set into sequences for windowed attention
- 4 orderings used simultaneously: `z`, `z-trans`, `hilbert`, `hilbert-trans`
- Orders are shuffled during training for robustness

### CAD Decoder (`model/vecformer/cad_decoder/cad_decoder.py`)

A Mask2Former-style decoder that generates instance predictions:
- 6 transformer blocks (configurable), embed_dim=256, 8 heads
- **Learnable instance queries** cross-attend to primitive features from the backbone
- Each block outputs: instance class logits, mask logits (dot product with primitive features), optional objectiveness score
- Iterative prediction: losses computed at every block (not just the last)
- Semantic head: MLP on primitive features, only from last block

### Layer Fusion Enhancement (LFE) (`modules/fusion_layer_feats_module.py`)

Optional module between backbone and CAD decoder that refines primitive features by fusing information across primitives sharing the same CAD layer ID.

---

## ArchCAD-400K Data Format

**Dataset class:** `data/floorplancad/floorplancad.py`
**Data structures:** `data/floorplancad/dataclass_define.py`
**Preprocessing:** `data/floorplancad/preprocess.py`

### Raw Format (JSON)

Each sample (`SVGData`) is a JSON file containing:

| Field              | Shape/Type      | Description                           |
|--------------------|-----------------|---------------------------------------|
| `viewBox`          | `(4,)`          | Bounding box: `[minx, miny, w, h]`   |
| `coords`           | `(N, 4)`        | Line segments: `[x1, y1, x2, y2]`    |
| `colors`           | `(N, 3)`        | RGB color per primitive               |
| `widths`           | `(N,)`          | Line width per primitive              |
| `primitive_ids`    | `(N,)`          | Which primitive each segment belongs to |
| `layer_ids`        | `(N,)`          | CAD layer membership                  |
| `semantic_ids`     | `(N,)`          | Semantic class label (0-34)           |
| `instance_ids`     | `(N,)`          | Instance ID                           |
| `primitive_lengths`| `(N,)`          | Number of segments per primitive      |

### Processed Format (`VecData`)

After preprocessing and augmentation:

| Field          | Shape     | Description                                                    |
|----------------|-----------|----------------------------------------------------------------|
| `coords`       | `(N, 4)`  | Normalized to `[-0.5, 0.5]`: `[x1, y1, x2, y2]`             |
| `feats`        | `(N, 10)` | `[length, |dx|, |dy|, cx, cy, pcx, pcy, r, g, b]`           |
| `prim_ids`     | `(N,)`    | Remapped primitive IDs                                        |
| `layer_ids`    | `(N,)`    | Layer IDs                                                     |
| `sem_ids`      | `(N,)`    | Semantic labels                                               |
| `inst_ids`     | `(N,)`    | Instance labels                                               |
| `prim_lengths` | `(N,)`    | Segments per primitive                                        |

**Collation** produces batched tensors with `cu_seqlens` (cumulative sequence lengths) and `cu_numprims` (cumulative primitive counts) for variable-length batch processing.

**Augmentation** (`data/floorplancad/transform_utils.py`): random flip, rotation, scale `[0.5, 1.5]`, translation.

### 35 Semantic Classes

- **Thing classes (0-29):** Instances like doors, windows, sinks, toilets, furniture, etc.
- **Stuff classes (30-34):** Non-instance regions like walls, railings, etc.

---

## Entry Points

### Inference

```bash
# Primary entry point
python launch.py --launch_mode test \
    --resume_from_checkpoint <checkpoint_path> \
    --config_path configs/vecformer.yaml \
    --model_args_path configs/model/vecformer.yaml \
    --data_args_path configs/data/floorplancad.yaml \
    --output_dir <output_dir>
```

**Script:** `scripts/test.sh` (uses `torchrun` for distributed evaluation)

**Key code path for inference:**
1. `launch.py:74-82` - test mode: loads checkpoint, calls `trainer.evaluate()`
2. `modeling_vecformer.py` - `VecFormer.forward()` runs backbone + decoder
3. `modeling_vecformer.py` - `VecFormer.predict()` (lines 254-312) generates semantic/instance/panoptic predictions with NMS, thresholding, and filtering

### Training

```bash
scripts/train.sh      # Fresh training
scripts/resume.sh     # Resume (preserves optimizer state)
scripts/continue.sh   # Continue (resets optimizer)
```

### Configuration

| File | Purpose |
|------|---------|
| `configs/vecformer.yaml` | Training hyperparams: lr=1e-4, AdamW, 500 epochs, batch=2/GPU |
| `configs/model/vecformer.yaml` | Model variant selection (defaults used) |
| `configs/data/floorplancad.yaml` | Dataset paths, sample mode, split files |

---

## PTv3 CUDA Kernels

**There are no custom CUDA kernels (.cu/.cpp) in this repository.** The PTv3 backbone (`model/vecformer/point_transformer_v3/model.py`, 1038 lines) delegates all GPU-accelerated operations to external compiled libraries:

| Library | Import Location | Purpose |
|---------|----------------|---------|
| `spconv` | `model.py:19` | Sparse convolution operations |
| `torch_scatter` | `model.py:20` | Scatter/gather for grouping |
| `flash_attn` | `model.py:30-33` | Flash Attention for efficient self/cross-attention |

These must be installed separately (see `requirements.txt`). The serialization (Z-order, Hilbert curves) is pure Python/PyTorch in `point_transformer_v3/serialization/`.

---

## Key Modules Quick Reference

| Module | File | Purpose |
|--------|------|---------|
| `VecFormer` | `model/vecformer/modeling_vecformer.py` | Top-level model (HuggingFace PreTrainedModel) |
| `VecFormerConfig` | `model/vecformer/configuration_vecformer.py` | All hyperparameters and defaults |
| `PointTransformerV3` | `model/vecformer/point_transformer_v3/model.py` | PTv3 backbone with serialization |
| `VecBackbone` | `model/vecformer/vec_backbone/vec_backbone.py` | Line-adapted PTv3 wrapper |
| `VecEncoder` | `model/vecformer/vec_backbone/vec_encoder.py` | Encoder with primitive/layer fusion |
| `VecDecoder` | `model/vecformer/vec_backbone/vec_decoder.py` | Symmetric decoder |
| `CADDecoder` | `model/vecformer/cad_decoder/cad_decoder.py` | Query-based instance decoder |
| `GroupFeatFusion` | `model/vecformer/modules/group_feat_fusion.py` | Primitive/layer feature aggregation |
| `VarlenSelfAttentionWithRoPE` | `model/vecformer/modules/attention.py` | Core attention with Flash Attention + RoPE |
| `Criterion` | `model/vecformer/criterion/criterion.py` | Multi-task loss (instance + semantic) |
| `Evaluator` | `model/vecformer/evaluator/evaluator.py` | PQ/SQ/RQ metrics |
| `FloorPlanCAD` | `data/floorplancad/floorplancad.py` | Dataset class |

---

## Tools

### `tools/legend_matcher.py` -- One-Shot Legend Matching

Implemented CLI tool that encodes a legend crop and a full floor plan through
the VecFormer backbone, sweeps the plan with a spatial sliding window, and
outputs bounding boxes for matches above a cosine-similarity threshold.

```bash
python tools/legend_matcher.py \
    --checkpoint vecformer_archcad.pth \
    --legend legend_crop.json \
    --floor_plan floor_plan.json \
    --output matches.json \
    --threshold 0.90 --window_size 0.15 --stride 0.05
```

Key API: `get_embedding(model, proj, coords, feats, prim_ids, layer_ids, cu_seqlens) → Tensor(512,)`

### `model/vecformer/siamese_head.py` -- Siamese Cross-Attention Head

`VecFormerSiameseHead` replaces the panoptic CAD decoder for search tasks.
Legend tokens (Q, 64) act as queries into plan tokens (N, 64) via multi-head
cross-attention, producing a per-primitive similarity score in [0, 1].

### `tools/siamese_inference.py` -- Siamese Search CLI

```bash
python tools/siamese_inference.py \
    --checkpoint weights/vecformer_archcad.pth \
    --legend legend_crop.json \
    --floor_plan floor_plan.json \
    --output siamese_matches.json \
    --threshold 0.85
```

Outputs clustered match groups with bounding boxes and primitive IDs.

### `utils/vector_healer.py` -- PDF Vector Extraction & Healing

Extracts vector paths from PDFs via PyMuPDF, merges fragmented segments via
Shapely `linemerge`, simplifies to stay within 128 segments/primitive, and
outputs `SVGData` compatible with the VecFormer pipeline.

---

## Plan: One-Shot Legend Matching System

### Goal

Given a single example (legend crop or symbol template) of a CAD symbol, find all instances of that symbol in a full floorplan drawing -- without retraining.

### Approach: Leverage the Existing Encoder as a Feature Extractor

The VecBackbone already produces rich per-primitive embeddings that encode geometric and semantic information about line groups. These embeddings can be repurposed for similarity-based matching.

### Implementation Plan

#### Phase 1: Feature Extraction Pipeline

1. **Create `inference/feature_extractor.py`**
   - Load a pretrained VecFormer checkpoint (backbone only, no CAD decoder needed)
   - Accept raw SVG/JSON input, run through preprocessing (`data/floorplancad/preprocess.py`)
   - Run through VecBackbone to get per-primitive feature embeddings (output of the encoder's final stage, dim=64 after decoder)
   - Pool primitive features into **per-instance embeddings** using existing `GroupFeatFusion` (or simple mean/max over primitive IDs)
   - Output: a feature bank of shape `(num_primitives, embed_dim)` for the full drawing

2. **Create `inference/legend_encoder.py`**
   - Extract a legend region: crop the SVG to the legend bounding box, filter primitives by spatial overlap
   - Encode each legend symbol as a single embedding vector by:
     - Running the cropped primitives through the same backbone
     - Mean-pooling all primitive features within the legend symbol -> one vector per symbol class

#### Phase 2: Matching Engine

3. **Create `inference/matcher.py`**
   - **Similarity computation:** Cosine similarity between each legend embedding and every primitive embedding in the target drawing
   - **Primitive-to-instance aggregation:** Average similarity scores across primitives sharing the same `primitive_id` to get instance-level scores
   - **Thresholding + NMS:** Apply a similarity threshold (tunable), then use the existing `vector_nms` from `modeling_vecformer.py` to suppress overlapping detections
   - Output: list of matched instances with locations and confidence scores

#### Phase 3: Integration

4. **Create `inference/one_shot_pipeline.py`** -- end-to-end pipeline:
   ```python
   def one_shot_match(
       checkpoint_path: str,
       target_drawing_path: str,    # full floorplan SVG/JSON
       legend_drawing_path: str,    # legend crop or same drawing with legend bbox
       legend_bbox: tuple = None,   # optional: auto-detect legend region
       similarity_threshold: float = 0.7,
   ) -> List[MatchResult]:
       ...
   ```

5. **Evaluation script** (`scripts/one_shot_eval.sh`):
   - Run on ArchCAD-400K test split
   - For each drawing, use one instance per class as the "legend" query
   - Report per-class recall@IoU=0.5 and mAP

#### Phase 4: Refinements

6. **Multi-scale matching:** Run the backbone at multiple scales (use the existing random-scale augmentation at `[0.75, 1.0, 1.25]`) and aggregate similarity scores
7. **Spatial attention masking:** Weight similarity by spatial proximity -- legend symbols far from the query region get a distance penalty
8. **Fine-grained discrimination:** If two legend symbols are visually similar (e.g., different door types), use the **Layer Fusion** features which encode CAD-layer context to disambiguate

### Key Architectural Decisions

- **No retraining required:** The pretrained backbone features are already discriminative for symbol types (evidenced by strong panoptic segmentation performance)
- **Reuse existing modules:** `GroupFeatFusion` for pooling, `vector_nms` for suppression, `preprocess.py` for input handling
- **Backbone output dim = 64** (first decoder channel): this is the primitive feature dimension to work with
- **The CAD Decoder is NOT needed** for one-shot matching -- it is task-specific to the closed-set panoptic classes. The backbone features are general-purpose
- **Layer Fusion Enhancement (LFE)** could optionally be included since it enriches features with CAD-layer context, which may help distinguish symbols that share geometry but differ in layer assignment

### Risk Assessment

| Risk | Mitigation |
|------|------------|
| Backbone features may not generalize to unseen symbol types | Fine-tune with contrastive loss on a small set of legend pairs |
| Scale variance between legend crop and in-situ symbols | Multi-scale inference with augmentation |
| Legend symbols may contain extraneous context lines | Use tight bounding box cropping + primitive filtering by spatial overlap |
| Computational cost for large drawings | Batch primitives, use FAISS for approximate nearest-neighbor search |
