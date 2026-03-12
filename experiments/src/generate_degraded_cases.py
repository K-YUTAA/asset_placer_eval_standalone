from __future__ import annotations

import argparse
import copy
import math
import pathlib
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from layout_tools import (
    as_float,
    load_layout_contract,
    obb_corners_xy,
    point_in_polygon,
    utc_now_iso,
    write_json,
)


Vec2 = Tuple[float, float]


FORBIDDEN_CATEGORIES = {"floor", "door", "window", "opening"}
BOTTLENECK_PREFERRED = ("table", "coffee_table", "chair", "sofa", "tv_cabinet", "cabinet")
OCCLUSION_PREFERRED = ("storage", "cabinet", "tv_cabinet", "sink", "sofa", "table")
CLUTTER_PREFERRED = ("chair", "table", "coffee_table", "sofa", "tv_cabinet")


def _room_centroid(poly: Sequence[Sequence[float]]) -> Vec2:
    if not poly:
        return (0.0, 0.0)
    sx = 0.0
    sy = 0.0
    n = 0
    for p in poly:
        if len(p) < 2:
            continue
        sx += as_float(p[0], 0.0)
        sy += as_float(p[1], 0.0)
        n += 1
    if n <= 0:
        return (0.0, 0.0)
    return (sx / n, sy / n)


def _room_bbox(poly: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    xs = [as_float(p[0], 0.0) for p in poly if len(p) >= 2]
    ys = [as_float(p[1], 0.0) for p in poly if len(p) >= 2]
    if not xs or not ys:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def _is_inside_room(obj: Dict[str, Any], room_poly: Sequence[Sequence[float]]) -> bool:
    corners = obb_corners_xy(obj)
    for x, y in corners:
        if not point_in_polygon(x, y, room_poly):
            return False
    return True


def _normalize(v: Vec2) -> Vec2:
    d = math.hypot(v[0], v[1])
    if d < 1e-9:
        return (1.0, 0.0)
    return (v[0] / d, v[1] / d)


def _line_point(a: Vec2, b: Vec2, t: float) -> Vec2:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _find_obj(layout: Dict[str, Any], object_id: str) -> Optional[Dict[str, Any]]:
    for obj in layout.get("objects", []):
        if str(obj.get("id")) == object_id:
            return obj
    return None


def _object_center(obj: Dict[str, Any]) -> Vec2:
    pose = obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]
    return (as_float(pose[0], 0.0), as_float(pose[1], 0.0))


def _set_object_pose_xy(obj: Dict[str, Any], x: float, y: float) -> None:
    pose = list(obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0])
    while len(pose) < 4:
        pose.append(0.0)
    pose[0] = x
    pose[1] = y
    obj["pose_xyz_yaw"] = pose


def _candidate_objects(layout: Dict[str, Any], preferred: Iterable[str]) -> List[Dict[str, Any]]:
    preferred_set = {str(x).strip().lower() for x in preferred}
    movable = [
        obj
        for obj in layout.get("objects", [])
        if bool(obj.get("movable", True))
        and str(obj.get("category") or "").strip().lower() not in FORBIDDEN_CATEGORIES
    ]
    ranked = []
    for obj in movable:
        cat = str(obj.get("category") or "").strip().lower()
        rank = 0 if cat in preferred_set else 1
        ranked.append((rank, cat, str(obj.get("id") or ""), obj))
    ranked.sort(key=lambda x: (x[0], x[1], x[2]))
    return [x[3] for x in ranked]


def _select_main_door_and_bed(layout: Dict[str, Any], room_poly: Sequence[Sequence[float]]) -> Tuple[Vec2, Vec2]:
    room_ctr = _room_centroid(room_poly)
    start = room_ctr
    goal = room_ctr

    doors = [obj for obj in layout.get("objects", []) if str(obj.get("category") or "").strip().lower() == "door"]
    if doors:
        def door_score(obj: Dict[str, Any]) -> float:
            size = obj.get("size_lwh_m") or [0.0, 0.0, 0.0]
            l = as_float(size[0] if len(size) > 0 else 0.0, 0.0)
            w = as_float(size[1] if len(size) > 1 else 0.0, 0.0)
            cx, cy = _object_center(obj)
            # Larger door first; tie-break toward boundary (entrance-like).
            bx, by, tx, ty = _room_bbox(room_poly)
            d_edge = min(abs(cx - bx), abs(tx - cx), abs(cy - by), abs(ty - cy))
            return max(l, w) * 100.0 - d_edge

        main_door = max(doors, key=door_score)
        start = _object_center(main_door)

    beds = [obj for obj in layout.get("objects", []) if str(obj.get("category") or "").strip().lower() == "bed"]
    if beds:
        goal = _object_center(beds[0])

    return start, goal


def _move_object_inside_room(
    layout: Dict[str, Any],
    object_id: str,
    target_xy: Vec2,
    room_poly: Sequence[Sequence[float]],
    margin_m: float = 0.05,
) -> Optional[Dict[str, Any]]:
    obj = _find_obj(layout, object_id)
    if obj is None:
        return None

    original_xy = _object_center(obj)
    room_ctr = _room_centroid(room_poly)
    tx, ty = target_xy

    # Keep target roughly inside bbox to reduce failures before polygon check.
    min_x, min_y, max_x, max_y = _room_bbox(room_poly)
    tx = min(max(tx, min_x + margin_m), max_x - margin_m)
    ty = min(max(ty, min_y + margin_m), max_y - margin_m)
    _set_object_pose_xy(obj, tx, ty)
    if _is_inside_room(obj, room_poly):
        return {"object_id": object_id, "from_xy": [original_xy[0], original_xy[1]], "to_xy": [tx, ty]}

    # Pull back toward room center until object corners are inside.
    best_xy = original_xy
    lo = 0.0
    hi = 1.0
    for _ in range(24):
        mid = (lo + hi) * 0.5
        cx = room_ctr[0] + (tx - room_ctr[0]) * mid
        cy = room_ctr[1] + (ty - room_ctr[1]) * mid
        _set_object_pose_xy(obj, cx, cy)
        if _is_inside_room(obj, room_poly):
            best_xy = (cx, cy)
            lo = mid
        else:
            hi = mid
    _set_object_pose_xy(obj, best_xy[0], best_xy[1])
    if _is_inside_room(obj, room_poly):
        return {"object_id": object_id, "from_xy": [original_xy[0], original_xy[1]], "to_xy": [best_xy[0], best_xy[1]]}

    _set_object_pose_xy(obj, original_xy[0], original_xy[1])
    return None


def _apply_bottleneck(layout: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    room_poly = layout["room"]["boundary_poly_xy"]
    start, goal = _select_main_door_and_bed(layout, room_poly)
    axis = _normalize((goal[0] - start[0], goal[1] - start[1]))
    perp = (-axis[1], axis[0])

    candidates = _candidate_objects(layout, BOTTLENECK_PREFERRED)
    if not candidates:
        return []
    chosen = candidates[0]
    oid = str(chosen.get("id"))

    target = _line_point(start, goal, 0.52)
    lateral = rng.uniform(-0.15, 0.15)
    target = (target[0] + perp[0] * lateral, target[1] + perp[1] * lateral)
    moved = _move_object_inside_room(layout, oid, target, room_poly)
    return [dict(moved, scenario="bottleneck")] if moved else []


def _apply_occlusion(layout: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    room_poly = layout["room"]["boundary_poly_xy"]
    start, goal = _select_main_door_and_bed(layout, room_poly)
    axis = _normalize((goal[0] - start[0], goal[1] - start[1]))
    perp = (-axis[1], axis[0])

    candidates = _candidate_objects(layout, OCCLUSION_PREFERRED)
    if not candidates:
        return []
    chosen = candidates[0]
    oid = str(chosen.get("id"))

    target = _line_point(start, goal, 0.18)
    lateral = rng.uniform(-0.2, 0.2)
    target = (target[0] + perp[0] * lateral, target[1] + perp[1] * lateral)
    moved = _move_object_inside_room(layout, oid, target, room_poly)
    return [dict(moved, scenario="occlusion")] if moved else []


def _apply_clutter(layout: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    room_poly = layout["room"]["boundary_poly_xy"]
    start, goal = _select_main_door_and_bed(layout, room_poly)
    axis = _normalize((goal[0] - start[0], goal[1] - start[1]))
    perp = (-axis[1], axis[0])

    candidates = _candidate_objects(layout, CLUTTER_PREFERRED)
    if not candidates:
        return []

    anchors = [
        _line_point(start, goal, 0.42),
        _line_point(start, goal, 0.50),
    ]
    offsets = [
        (-0.20, 0.18),
        (0.20, -0.18),
    ]

    changes: List[Dict[str, Any]] = []
    used_ids = set()
    idx = 0
    for cand in candidates:
        oid = str(cand.get("id"))
        if oid in used_ids:
            continue
        if idx >= 2:
            break
        base = anchors[idx]
        u, v = offsets[idx]
        jitter_u = rng.uniform(-0.05, 0.05)
        jitter_v = rng.uniform(-0.05, 0.05)
        target = (
            base[0] + axis[0] * (u + jitter_u) + perp[0] * (v + jitter_v),
            base[1] + axis[1] * (u + jitter_u) + perp[1] * (v + jitter_v),
        )
        moved = _move_object_inside_room(layout, oid, target, room_poly)
        if moved:
            changes.append(dict(moved, scenario="clutter"))
            used_ids.add(oid)
            idx += 1
    return changes


def _iter_case_dirs(source_root: pathlib.Path, layout_filename: str) -> List[pathlib.Path]:
    cases = []
    for p in sorted(source_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / layout_filename).exists():
            cases.append(p)
    return cases


def _copy_layout(layout: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(layout)


def _save_variant_layout(
    out_root: pathlib.Path,
    case_name: str,
    variant_name: str,
    layout: Dict[str, Any],
    base_layout_path: pathlib.Path,
    changes: List[Dict[str, Any]],
) -> pathlib.Path:
    out_dir = out_root / case_name / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)

    layout_out = _copy_layout(layout)
    meta = dict(layout_out.get("meta") or {})
    meta["layout_id"] = f"{case_name}__{variant_name}"
    meta["source"] = "degraded_case_generator"
    meta["source_layout_path"] = str(base_layout_path)
    meta["degradation_variant"] = variant_name
    meta["degradation_changes"] = changes
    layout_out["meta"] = meta

    out_path = out_dir / "layout_generated.json"
    write_json(out_path, layout_out)
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate degraded cases for eval optimization.")
    parser.add_argument("--source_root", required=True, help="Root directory that contains per-case layout_generated.json")
    parser.add_argument("--out_root", required=True, help="Output directory for generated cases")
    parser.add_argument("--layout_filename", default="layout_generated.json")
    parser.add_argument("--case_names", default="", help="Comma-separated base case names; empty means all in source_root")
    parser.add_argument("--seed", type=int, default=20260223)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    source_root = pathlib.Path(args.source_root).resolve()
    out_root = pathlib.Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    case_filter = {x.strip() for x in str(args.case_names).split(",") if x.strip()}
    case_dirs = _iter_case_dirs(source_root, args.layout_filename)
    if case_filter:
        case_dirs = [p for p in case_dirs if p.name in case_filter]

    all_entries: List[Dict[str, Any]] = []
    scenario_counts: Dict[str, int] = {"base": 0, "bottleneck": 0, "occlusion": 0, "clutter": 0}

    for case_dir in case_dirs:
        case_name = case_dir.name
        base_path = case_dir / args.layout_filename
        base_layout = load_layout_contract(base_path)

        base_out = _save_variant_layout(
            out_root=out_root,
            case_name=case_name,
            variant_name="base",
            layout=base_layout,
            base_layout_path=base_path,
            changes=[],
        )
        all_entries.append(
            {
                "base_case": case_name,
                "variant": "base",
                "scenario": "base",
                "layout_id": f"{case_name}__base",
                "layout_path": str(base_out),
                "source_layout_path": str(base_path),
                "changes": [],
            }
        )
        scenario_counts["base"] += 1

        for variant_name, applier in (
            ("bottleneck", _apply_bottleneck),
            ("occlusion", _apply_occlusion),
            ("clutter", _apply_clutter),
        ):
            local_rng = random.Random(hash((args.seed, case_name, variant_name)) & 0xFFFFFFFF)
            layout_variant = _copy_layout(base_layout)
            changes = applier(layout_variant, local_rng)
            out_path = _save_variant_layout(
                out_root=out_root,
                case_name=case_name,
                variant_name=variant_name,
                layout=layout_variant,
                base_layout_path=base_path,
                changes=changes,
            )
            all_entries.append(
                {
                    "base_case": case_name,
                    "variant": variant_name,
                    "scenario": variant_name,
                    "layout_id": f"{case_name}__{variant_name}",
                    "layout_path": str(out_path),
                    "source_layout_path": str(base_path),
                    "changes": changes,
                }
            )
            scenario_counts[variant_name] += 1

    manifest = {
        "generated_at": utc_now_iso(),
        "generator": "generate_degraded_cases.py",
        "source_root": str(source_root),
        "out_root": str(out_root),
        "layout_filename": args.layout_filename,
        "seed": args.seed,
        "total_cases": len(case_dirs),
        "total_variants": len(all_entries),
        "scenario_counts": scenario_counts,
        "entries": all_entries,
    }
    write_json(out_root / "degraded_cases_manifest.json", manifest)
    print(
        {
            "out_root": str(out_root),
            "manifest": str(out_root / "degraded_cases_manifest.json"),
            "total_cases": len(case_dirs),
            "total_variants": len(all_entries),
            "scenario_counts": scenario_counts,
        }
    )


if __name__ == "__main__":
    main()

