# 2026-03-14 wall penetration fix plan

## Problem

The current `heuristic` / `proposed` refiners still allow visually obvious wall penetration in some cases.

The root causes are:

1. Candidate validity uses only OBB corner containment against a simplified room polygon.
2. Wall margin uses only corner-to-edge distance.
3. Candidate validity sometimes uses a coarse room boundary instead of the actual room polygon for the object.
4. Plotting uses raster wall lines, so a geometry-side loose check becomes visible immediately.

## Goal

Reduce wall penetration without touching the frozen main protocol.

This is a spec-update candidate for refinement plausibility, not a replacement of the frozen main.

## Scope

Apply the fix to:

- `experiments/src/refine_heuristic.py`
- `experiments/src/refine_proposed_beam.py`
- shared geometry helpers in `experiments/src/layout_tools.py`

## Changes

### 1. Replace corner-only containment with footprint sampling

For each OBB candidate, sample:

- 4 corners
- edge points at fixed spacing
- center point

Require all sampled points to be inside or on the room polygon.

This is meant to catch rectangles that bridge a concave notch while their corners still remain inside.

### 2. Replace corner-only wall margin with footprint-based wall margin

Compute wall clearance from sampled footprint points to polygon edges, not only from corners.

This makes the margin react to long edges that approach or cross walls.

### 3. Use per-object room polygon

Use `room_polygon_for_object(layout, obj)` instead of the coarse `layout["room"]["boundary_poly_xy"]` wherever possible.

This avoids validating a candidate against the wrong room boundary in multi-room / notch cases.

## Non-goals in this patch

- switching candidate validity to evaluator occupancy
- full OBB-vs-OBB overlap SAT
- changing frozen `eval_v1.json`
- changing frozen `proposed_beam_v1.json`

## Expected effect

- less apparent wall penetration in `plot_with_bg`
- fewer cases where a candidate spans a concave inner notch
- stricter room-boundary validity with minimal change to the current refinement code path

## Follow-up

If this still leaves visible penetration, the next step is to use evaluator occupancy as the candidate validity truth instead of polygon approximations.
