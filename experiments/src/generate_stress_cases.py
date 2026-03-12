from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pathlib
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from eval_metrics import default_eval_config, evaluate_layout, merge_eval_config
from layout_tools import as_float, load_layout_contract, obb_corners_xy, point_in_polygon, utc_now_iso, write_json
from stress_dataset_qa import build_dataset_qa_report, write_dataset_qa_report
from stress_manifest import build_sample_manifest, sha256_file
from stress_priors import (
    choose_count_from_distribution,
    choose_weighted_item,
    local_offset_to_world,
    sample_local_frame_delta,
    sample_multiple_object_ids,
)


Vec2 = Tuple[float, float]


EXCLUDE_COLLISION_CATEGORIES = {"floor", "door", "window", "opening"}


@dataclass
class CandidateResult:
    score: float
    layout: Dict[str, Any]
    metrics: Dict[str, Any]
    debug: Dict[str, Any]
    action: Dict[str, Any]


@dataclass
class NaturalCandidate:
    layout: Dict[str, Any]
    metrics: Dict[str, Any]
    debug: Dict[str, Any]
    actions: List[Dict[str, Any]]
    selection_notes: str


def _clearance_value(metrics: Dict[str, Any]) -> float:
    if "clr_feasible" in metrics:
        return as_float(metrics.get("clr_feasible"), 0.0)
    if "clr_min_astar" in metrics:
        return as_float(metrics.get("clr_min_astar"), 0.0)
    return as_float(metrics.get("clr_min"), 0.0)


def _clearance_metric_key(metrics: Dict[str, Any]) -> str:
    if "clr_feasible" in metrics:
        return "clr_feasible"
    if "clr_min_astar" in metrics:
        return "clr_min_astar"
    return "clr_min"


def _clearance_threshold(eval_cfg: Dict[str, Any]) -> float:
    tau_clr = as_float(eval_cfg.get("tau_clr"), 0.2)
    adopt_cfg = eval_cfg.get("adopt") if isinstance(eval_cfg.get("adopt"), dict) else {}
    mode = str(adopt_cfg.get("clearance_metric") or "clr_feasible").strip().lower()
    if mode in {"clr_min", "clr_min_astar", "astar", "path"}:
        return as_float(eval_cfg.get("tau_clr_astar"), tau_clr)
    return as_float(eval_cfg.get("tau_clr_feasible"), tau_clr)


def _load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(x) for x in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _room_polygon(room: Dict[str, Any]) -> List[List[float]]:
    raw = room.get("room_polygon")
    if not isinstance(raw, list):
        raw = room.get("boundary_poly_xy")
    out: List[List[float]] = []
    if not isinstance(raw, list):
        return out
    for p in raw:
        if isinstance(p, dict):
            out.append([as_float(p.get("X", p.get("x", 0.0)), 0.0), as_float(p.get("Y", p.get("y", 0.0)), 0.0)])
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append([as_float(p[0], 0.0), as_float(p[1], 0.0)])
    return out


def _polygon_area(poly: Sequence[Sequence[float]]) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    area2 = 0.0
    for i in range(n):
        x0, y0 = as_float(poly[i][0], 0.0), as_float(poly[i][1], 0.0)
        x1, y1 = as_float(poly[(i + 1) % n][0], 0.0), as_float(poly[(i + 1) % n][1], 0.0)
        area2 += x0 * y1 - x1 * y0
    return abs(0.5 * area2)


def _room_bbox(poly: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    xs = [as_float(p[0], 0.0) for p in poly if len(p) >= 2]
    ys = [as_float(p[1], 0.0) for p in poly if len(p) >= 2]
    if not xs or not ys:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def _centroid(poly: Sequence[Sequence[float]]) -> Vec2:
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


def _object_center(obj: Dict[str, Any]) -> Vec2:
    pose = obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]
    return (as_float(pose[0], 0.0), as_float(pose[1], 0.0))


def _object_yaw(obj: Dict[str, Any]) -> float:
    pose = obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]
    return as_float(pose[3] if len(pose) >= 4 else 0.0, 0.0)


def _set_object_pose(obj: Dict[str, Any], x: float, y: float, yaw_rad: float) -> None:
    pose = list(obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0])
    while len(pose) < 4:
        pose.append(0.0)
    pose[0] = x
    pose[1] = y
    pose[3] = yaw_rad
    obj["pose_xyz_yaw"] = pose


def _next_clutter_id(layout: Dict[str, Any]) -> str:
    max_idx = -1
    for obj in layout.get("objects", []):
        oid = str(obj.get("id") or "")
        if oid.startswith("clutter_"):
            try:
                max_idx = max(max_idx, int(oid.split("_")[-1]))
            except Exception:
                continue
    for item in layout.get("clutter_objects", []):
        if not isinstance(item, dict):
            continue
        oid = str(item.get("id") or "")
        if oid.startswith("clutter_"):
            try:
                max_idx = max(max_idx, int(oid.split("_")[-1]))
            except Exception:
                continue
    return f"clutter_{max_idx + 1:02d}"


def _append_clutter_object(
    layout: Dict[str, Any],
    *,
    object_id: str,
    x: float,
    y: float,
    yaw_rad: float,
    length_m: float,
    width_m: float,
    height_m: float,
) -> None:
    layout.setdefault("objects", []).append(
        {
            "id": object_id,
            "category": "clutter",
            "asset_query": "temporary clutter box",
            "asset_id": "",
            "scale": [1.0, 1.0, 1.0],
            "size_lwh_m": [length_m, width_m, height_m],
            "pose_xyz_yaw": [x, y, 0.0, yaw_rad],
            "movable": False,
        }
    )
    layout.setdefault("clutter_objects", []).append(
        {
            "id": object_id,
            "shape": "box",
            "size_xy": [length_m, width_m],
            "height_m": height_m,
            "pose_xytheta": [x, y, yaw_rad],
            "movable": False,
        }
    )


def _find_object(layout: Dict[str, Any], object_id: str) -> Optional[Dict[str, Any]]:
    for obj in layout.get("objects", []):
        if str(obj.get("id")) == object_id:
            return obj
    return None


def _normalize(v: Vec2) -> Vec2:
    d = math.hypot(v[0], v[1])
    if d < 1e-9:
        return (1.0, 0.0)
    return (v[0] / d, v[1] / d)


def _normal(v: Vec2) -> Vec2:
    return (-v[1], v[0])


def _point_on_line(a: Vec2, b: Vec2, t: float) -> Vec2:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _distance(a: Vec2, b: Vec2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _object_category(obj: Dict[str, Any]) -> str:
    return str(obj.get("category") or "").strip().lower()


def _config_mode(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("generator_mode") or "legacy_targeted").strip().lower()


def _config_variants(cfg: Dict[str, Any]) -> List[str]:
    mode = _config_mode(cfg)
    if mode in {"natural_main", "targeted_diag"}:
        raw = cfg.get("variants")
        if isinstance(raw, list) and raw:
            return [str(x) for x in raw]
    return list(cfg.get("output", {}).get("variants") or ["base", "bottleneck", "occlusion", "clutter"])


def _legacy_scenarios_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw = cfg.get("scenarios")
    return raw if isinstance(raw, dict) else {}


def _natural_family_cfg(cfg: Dict[str, Any], variant: str) -> Dict[str, Any]:
    families = cfg.get("families")
    if isinstance(families, dict):
        node = families.get(variant)
        if isinstance(node, dict):
            return node
    return {}


def _targeted_scenario_cfg(cfg: Dict[str, Any], variant: str) -> Dict[str, Any]:
    node = cfg.get(variant)
    if isinstance(node, dict):
        return node
    return {}


def _bed_object_for_goal(layout: Dict[str, Any], goal_xy: Vec2) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_dist = float("inf")
    for obj in layout.get("objects", []):
        if _object_category(obj) != "bed":
            continue
        d = _distance(_object_center(obj), goal_xy)
        if d < best_dist:
            best_dist = d
            best = obj
    return best


def _sample_band_point(
    *,
    rng: random.Random,
    band_cfg: Dict[str, Any],
    start_xy: Vec2,
    goal_xy: Vec2,
    layout: Dict[str, Any],
) -> Tuple[Vec2, str]:
    forward = _normalize((goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1]))
    normal = _normal(forward)
    band_id = str(band_cfg.get("band_id") or "unknown_band")
    anchor = str(band_cfg.get("anchor") or "")

    if anchor == "start_point":
        tangent_length = max(0.1, as_float(band_cfg.get("tangent_length_m"), 1.0))
        normal_width = max(0.1, as_float(band_cfg.get("normal_width_m"), 0.6))
        s = rng.uniform(0.0, tangent_length)
        n = rng.uniform(-0.5 * normal_width, 0.5 * normal_width)
        return ((start_xy[0] + forward[0] * s + normal[0] * n, start_xy[1] + forward[1] * s + normal[1] * n), band_id)

    if anchor == "start_goal_segment":
        t_range = band_cfg.get("segment_t_range") if isinstance(band_cfg.get("segment_t_range"), list) else [0.25, 0.75]
        t0 = min(1.0, max(0.0, as_float(t_range[0], 0.25)))
        t1 = min(1.0, max(t0, as_float(t_range[1], 0.75)))
        half_width = max(0.05, as_float(band_cfg.get("normal_half_width_m"), 0.45))
        t = rng.uniform(t0, t1)
        center = _point_on_line(start_xy, goal_xy, t)
        n = rng.uniform(-half_width, half_width)
        return ((center[0] + normal[0] * n, center[1] + normal[1] * n), band_id)

    if anchor == "bed_perimeter_goal_side":
        bed = _bed_object_for_goal(layout, goal_xy)
        bed_center = _object_center(bed) if isinstance(bed, dict) else (goal_xy[0] - forward[0] * 0.6, goal_xy[1] - forward[1] * 0.6)
        out_dir = _normalize((goal_xy[0] - bed_center[0], goal_xy[1] - bed_center[1]))
        tangent = _normal(out_dir)
        offset_range = band_cfg.get("offset_range_m") if isinstance(band_cfg.get("offset_range_m"), list) else [0.2, 0.8]
        r0 = as_float(offset_range[0], 0.2)
        r1 = max(r0, as_float(offset_range[1], 0.8))
        radial = rng.uniform(r0, r1) - 0.6
        lateral = rng.uniform(-0.35, 0.35)
        return ((goal_xy[0] + out_dir[0] * radial + tangent[0] * lateral, goal_xy[1] + out_dir[1] * radial + tangent[1] * lateral), band_id)

    return (start_xy, band_id)


def _sample_clutter_yaw(rng: random.Random, clutter_cfg: Dict[str, Any]) -> float:
    pose_jitter = clutter_cfg.get("pose_jitter") if isinstance(clutter_cfg.get("pose_jitter"), dict) else {}
    base_choices = pose_jitter.get("yaw_base_deg") if isinstance(pose_jitter.get("yaw_base_deg"), list) else [0.0, 90.0]
    base_deg = as_float(base_choices[int(rng.randrange(len(base_choices)))], 0.0) if base_choices else 0.0
    jitter = rng.uniform(-as_float(pose_jitter.get("yaw_jitter_deg"), 10.0), as_float(pose_jitter.get("yaw_jitter_deg"), 10.0))
    return math.radians(base_deg + jitter)


def _usage_shift_candidates(layout: Dict[str, Any], family_cfg: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    eligible = {str(x).strip().lower() for x in family_cfg.get("eligible_categories") or []}
    low_priority = {str(x).strip().lower() for x in family_cfg.get("low_priority_categories") or []}
    priors = family_cfg.get("priors_local_frame") if isinstance(family_cfg.get("priors_local_frame"), dict) else {}
    preferred_ids: List[str] = []
    low_ids: List[str] = []
    for obj in layout.get("objects", []):
        oid = str(obj.get("id") or "")
        cat = _object_category(obj)
        if not oid or cat in EXCLUDE_COLLISION_CATEGORIES:
            continue
        if cat in eligible and isinstance(priors.get(cat), dict):
            preferred_ids.append(oid)
        elif cat in low_priority and isinstance(priors.get(cat), dict):
            low_ids.append(oid)
    return preferred_ids, low_ids


def _pool_summary(valid_count: int, attempts: int, rejection_counts: Dict[str, int]) -> Dict[str, Any]:
    return {
        "attempted_candidates": attempts,
        "valid_candidates": valid_count,
        "rejected_candidates": max(0, attempts - valid_count),
        "rejection_counts": rejection_counts,
    }


def _stress_family_name(mode: str, variant: str) -> str:
    v = str(variant or "").strip().lower()
    if mode in {"targeted_diag", "legacy_targeted"}:
        if v in {"bottleneck", "targeted_bottleneck"}:
            return "targeted_bottleneck"
        if v in {"occlusion", "targeted_occlusion"}:
            return "targeted_occlusion"
    return v


def _build_room_registry(layout: Dict[str, Any]) -> Tuple[Dict[str, List[List[float]]], List[List[float]]]:
    main_room = list(layout.get("room", {}).get("boundary_poly_xy") or [])
    rooms: Dict[str, List[List[float]]] = {}
    for idx, room in enumerate(layout.get("rooms") or []):
        if not isinstance(room, dict):
            continue
        rid = str(room.get("room_id") or f"room_{idx+1}")
        poly = _room_polygon(room)
        if len(poly) >= 3:
            rooms[rid] = poly
    if not rooms and len(main_room) >= 3:
        rooms["main_room"] = main_room
    return rooms, main_room


def _room_for_point(rooms: Dict[str, List[List[float]]], point: Vec2) -> Optional[str]:
    candidates: List[Tuple[float, str]] = []
    for rid, poly in rooms.items():
        if point_in_polygon(point[0], point[1], poly):
            candidates.append((_polygon_area(poly), rid))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _assign_object_rooms(layout: Dict[str, Any], rooms: Dict[str, List[List[float]]], main_room: List[List[float]]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for obj in layout.get("objects", []):
        oid = str(obj.get("id") or "")
        c = _object_center(obj)
        rid = _room_for_point(rooms, c)
        if rid is None and len(main_room) >= 3 and point_in_polygon(c[0], c[1], main_room):
            rid = "main_room"
        out[oid] = rid
    return out


def _inside_polygon(obj: Dict[str, Any], poly: Sequence[Sequence[float]]) -> bool:
    if len(poly) < 3:
        return False
    corners = obb_corners_xy(obj)
    for x, y in corners:
        if not point_in_polygon(x, y, poly):
            return False
    return True


def _obb_aabb(obj: Dict[str, Any]) -> Tuple[float, float, float, float]:
    corners = obb_corners_xy(obj)
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def _aabb_intersection(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _aabb_area(a: Tuple[float, float, float, float]) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def _overlap_ratio(obj: Dict[str, Any], other: Dict[str, Any]) -> float:
    a = _obb_aabb(obj)
    b = _obb_aabb(other)
    inter = _aabb_intersection(a, b)
    if inter <= 0.0:
        return 0.0
    den = min(_aabb_area(a), _aabb_area(b))
    if den <= 1e-9:
        return 0.0
    return inter / den


def _collect_doors(layout: Dict[str, Any]) -> List[Vec2]:
    out: List[Vec2] = []
    for obj in layout.get("objects", []):
        if str(obj.get("category") or "").strip().lower() == "door":
            out.append(_object_center(obj))
    return out


def _task_start_goal(base_layout: Dict[str, Any], base_debug: Dict[str, Any], main_room: List[List[float]]) -> Tuple[Vec2, Vec2]:
    start_xy: Optional[Vec2] = None
    goal_xy: Optional[Vec2] = None

    task = base_debug.get("task_points")
    if isinstance(task, dict):
        start = (task.get("start") or {}).get("xy")
        goal = (task.get("goal") or {}).get("xy")
        if isinstance(start, (list, tuple)) and len(start) >= 2:
            start_xy = (as_float(start[0], 0.0), as_float(start[1], 0.0))
        if isinstance(goal, (list, tuple)) and len(goal) >= 2:
            goal_xy = (as_float(goal[0], 0.0), as_float(goal[1], 0.0))

    if start_xy is None:
        doors = _collect_doors(base_layout)
        if doors:
            start_xy = doors[0]

    if goal_xy is None:
        for obj in base_layout.get("objects", []):
            if str(obj.get("category") or "").strip().lower() == "bed":
                goal_xy = _object_center(obj)
                break

    ctr = _centroid(main_room)
    if start_xy is None:
        start_xy = ctr
    if goal_xy is None:
        goal_xy = ctr
    return start_xy, goal_xy


def _iter_case_dirs(source_root: pathlib.Path, layout_filename: str, case_names: Sequence[str]) -> List[pathlib.Path]:
    selected = set(x for x in case_names if x)
    out: List[pathlib.Path] = []
    for p in sorted(source_root.iterdir()):
        if not p.is_dir():
            continue
        if selected and p.name not in selected:
            continue
        if (p / layout_filename).exists():
            out.append(p)
    return out


def _movable_candidates(layout: Dict[str, Any], degrade_cfg: Dict[str, Any], preferred: Iterable[str]) -> List[str]:
    movable_set = {str(x).strip().lower() for x in (degrade_cfg.get("movable_categories") or [])}
    fixed_set = {str(x).strip().lower() for x in (degrade_cfg.get("fixed_categories") or [])}
    pref_set = {str(x).strip().lower() for x in preferred}

    ranked: List[Tuple[int, str]] = []
    for obj in layout.get("objects", []):
        oid = str(obj.get("id") or "")
        cat = str(obj.get("category") or "").strip().lower()
        if not oid:
            continue
        if not bool(obj.get("movable", True)):
            continue
        if cat in fixed_set:
            continue
        if movable_set and cat not in movable_set:
            continue
        rank = 0 if cat in pref_set else 1
        ranked.append((rank, oid))
    ranked.sort(key=lambda x: (x[0], x[1]))
    return [x[1] for x in ranked]


def _frange_symmetric(radius: float, step: float) -> List[float]:
    if step <= 0.0:
        return [0.0]
    n = int(max(0, math.floor(radius / step)))
    vals = [i * step for i in range(-n, n + 1)]
    if 0.0 not in vals:
        vals.append(0.0)
    vals = sorted(set(round(v, 6) for v in vals))
    return vals


def _position_candidates(anchor: Vec2, radius: float, step: float) -> List[Vec2]:
    out: List[Vec2] = []
    vals = _frange_symmetric(radius, step)
    for dx in vals:
        for dy in vals:
            if dx * dx + dy * dy > radius * radius + 1e-9:
                continue
            out.append((anchor[0] + dx, anchor[1] + dy))
    out.sort(key=lambda p: (math.hypot(p[0] - anchor[0], p[1] - anchor[1]), p[0], p[1]))
    return out


def _yaw_candidates(base_yaw: float, rot_step_deg: float, rot_max_deg: float) -> List[float]:
    if rot_step_deg <= 0.0 or rot_max_deg <= 0.0:
        return [base_yaw]
    step = math.radians(rot_step_deg)
    m = math.radians(rot_max_deg)
    out = [base_yaw]
    k = 1
    while k * step <= m + 1e-9:
        out.append(base_yaw + k * step)
        out.append(base_yaw - k * step)
        k += 1
    return out


def _quick_constraints_ok(
    layout: Dict[str, Any],
    moved_object_id: str,
    obj_room_map: Dict[str, Optional[str]],
    rooms: Dict[str, List[List[float]]],
    main_room: List[List[float]],
    door_centers: Sequence[Vec2],
    constraints: Dict[str, Any],
    override_room_id: Optional[str] = None,
) -> bool:
    obj = _find_object(layout, moved_object_id)
    if obj is None:
        return False

    same_room_only = bool(constraints.get("same_room_only", True))
    keepout_r = as_float(constraints.get("door_keepout_radius_m"), 0.5)
    overlap_max = as_float(constraints.get("overlap_ratio_max"), 0.05)

    allowed_poly = main_room
    if same_room_only:
        rid = override_room_id if override_room_id is not None else obj_room_map.get(moved_object_id)
        if rid is not None and rid in rooms:
            allowed_poly = rooms[rid]
    if len(allowed_poly) < 3:
        return False
    if not _inside_polygon(obj, allowed_poly):
        return False

    c = _object_center(obj)
    for d in door_centers:
        if _distance(c, d) < keepout_r - 1e-9:
            return False

    cat = str(obj.get("category") or "").strip().lower()
    if cat not in EXCLUDE_COLLISION_CATEGORIES:
        for other in layout.get("objects", []):
            oid = str(other.get("id") or "")
            if oid == moved_object_id:
                continue
            other_cat = str(other.get("category") or "").strip().lower()
            if other_cat in EXCLUDE_COLLISION_CATEGORIES:
                continue
            if _overlap_ratio(obj, other) > overlap_max + 1e-9:
                return False
    return True


def _anchor_from_start_goal(start: Vec2, goal: Vec2, t: float) -> Vec2:
    return _point_on_line(start, goal, max(0.0, min(1.0, t)))


def _copy_layout(layout: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(layout)


def _best_single_move(
    *,
    base_layout: Dict[str, Any],
    start_layout: Dict[str, Any],
    base_metrics: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    degrade_cfg: Dict[str, Any],
    rooms: Dict[str, List[List[float]]],
    main_room: List[List[float]],
    obj_room_map: Dict[str, Optional[str]],
    door_centers: Sequence[Vec2],
    object_ids: Sequence[str],
    anchor: Vec2,
    score_fn: Callable[[Dict[str, Any], Dict[str, Any]], Optional[float]],
    top_k: int,
) -> List[CandidateResult]:
    constraints = degrade_cfg.get("constraints") or {}
    search_cfg = degrade_cfg.get("search") or {}

    step_m = as_float(search_cfg.get("translation_step_m"), 0.1)
    d_max = as_float(constraints.get("translation_max_m"), 0.6)
    radius = min(d_max, as_float(search_cfg.get("top_radius_m"), d_max))
    rot_step_deg = as_float(search_cfg.get("rotation_step_deg"), 15.0)
    rot_max_deg = as_float(constraints.get("rotation_max_deg"), 30.0)
    anchor_bias_weight = as_float(search_cfg.get("anchor_bias_weight"), 0.35)

    candidates: List[CandidateResult] = []
    for oid in object_ids:
        base_obj = _find_object(start_layout, oid)
        if base_obj is None:
            continue
        orig = _object_center(base_obj)
        y0 = _object_yaw(base_obj)

        # Candidate positions are generated around the current pose (controlled perturbation).
        # The scenario anchor is used only as a soft preference in ranking.
        pos_list = _position_candidates(orig, radius=radius, step=step_m)
        yaw_list = _yaw_candidates(y0, rot_step_deg, rot_max_deg)

        tested = 0
        for px, py in pos_list:
            for yaw in yaw_list:
                tested += 1
                cand_layout = _copy_layout(start_layout)
                cand_obj = _find_object(cand_layout, oid)
                if cand_obj is None:
                    continue
                _set_object_pose(cand_obj, px, py, yaw)

                if not _quick_constraints_ok(
                    cand_layout,
                    moved_object_id=oid,
                    obj_room_map=obj_room_map,
                    rooms=rooms,
                    main_room=main_room,
                    door_centers=door_centers,
                    constraints=constraints,
                ):
                    continue

                metrics, debug = evaluate_layout(cand_layout, base_layout, eval_cfg)
                if int(metrics.get("validity", 0)) != 1:
                    continue

                raw_score = score_fn(metrics, debug)
                if raw_score is None:
                    continue
                anchor_dist = _distance((px, py), anchor)
                score = raw_score - anchor_bias_weight * anchor_dist
                dx_world = px - orig[0]
                dy_world = py - orig[1]
                dx_local = math.cos(y0) * dx_world + math.sin(y0) * dy_world
                dy_local = -math.sin(y0) * dx_world + math.cos(y0) * dy_world

                action = {
                    "type": "move_object",
                    "object_id": oid,
                    "category": _object_category(obj),
                    "from_xy": [orig[0], orig[1]],
                    "to_xy": [px, py],
                    "from_yaw": y0,
                    "to_yaw": yaw,
                    "dx_local": dx_local,
                    "dy_local": dy_local,
                    "dtheta_deg": math.degrees(yaw - y0),
                    "same_room_ok": True,
                    "delta_translation_m": _distance((px, py), orig),
                    "delta_yaw_deg": abs(math.degrees(yaw - y0)),
                    "anchor_distance_m": anchor_dist,
                    "score_raw": raw_score,
                    "score_final": score,
                }
                candidates.append(CandidateResult(score=score, layout=cand_layout, metrics=metrics, debug=debug, action=action))

                if tested >= int(as_float(search_cfg.get("top_candidates_per_object"), 80)):
                    break
            if tested >= int(as_float(search_cfg.get("top_candidates_per_object"), 80)):
                break

    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates[: max(1, top_k)]


def _clutter_sizes_xy(scenario_cfg: Dict[str, Any]) -> List[Tuple[float, float]]:
    raw = scenario_cfg.get("object_sizes_xy_m")
    out: List[Tuple[float, float]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                l = max(0.1, as_float(item[0], 0.4))
                w = max(0.1, as_float(item[1], 0.4))
                out.append((l, w))
            elif isinstance(item, dict):
                l = max(0.1, as_float(item.get("length", item.get("L", 0.4)), 0.4))
                w = max(0.1, as_float(item.get("width", item.get("W", 0.4)), 0.4))
                out.append((l, w))
    if out:
        return out
    return [(0.35, 0.35), (0.60, 0.40)]


def _best_single_clutter_add(
    *,
    base_layout: Dict[str, Any],
    start_layout: Dict[str, Any],
    base_metrics: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    degrade_cfg: Dict[str, Any],
    rooms: Dict[str, List[List[float]]],
    main_room: List[List[float]],
    obj_room_map: Dict[str, Optional[str]],
    door_centers: Sequence[Vec2],
    anchor: Vec2,
    score_fn: Callable[[Dict[str, Any], Dict[str, Any]], Optional[float]],
    top_k: int,
    clutter_index: int,
    scenario_cfg: Dict[str, Any],
) -> List[CandidateResult]:
    constraints = degrade_cfg.get("constraints") or {}
    search_cfg = degrade_cfg.get("search") or {}

    step_m = as_float(search_cfg.get("translation_step_m"), 0.1)
    d_max = as_float(constraints.get("translation_max_m"), 0.6)
    radius = min(d_max, as_float(scenario_cfg.get("search_radius_m"), d_max))
    rot_step_deg = as_float(search_cfg.get("rotation_step_deg"), 15.0)
    rot_max_deg = as_float(constraints.get("rotation_max_deg"), 30.0)
    anchor_bias_weight = as_float(search_cfg.get("anchor_bias_weight"), 0.35)
    top_candidates_limit = int(as_float(search_cfg.get("top_candidates_per_object"), 80))
    clutter_h = max(0.1, as_float(scenario_cfg.get("object_height_m"), 1.0))

    anchor_room_id = _room_for_point(rooms, anchor)
    if anchor_room_id is None and len(main_room) >= 3 and point_in_polygon(anchor[0], anchor[1], main_room):
        anchor_room_id = "main_room"

    pos_list = _position_candidates(anchor, radius=radius, step=step_m)
    yaw_list = _yaw_candidates(0.0, rot_step_deg, rot_max_deg)
    clutter_sizes = _clutter_sizes_xy(scenario_cfg)

    candidates: List[CandidateResult] = []
    tested = 0
    for length_m, width_m in clutter_sizes:
        for px, py in pos_list:
            for yaw in yaw_list:
                tested += 1
                cand_layout = _copy_layout(start_layout)
                clutter_id = _next_clutter_id(cand_layout)
                _append_clutter_object(
                    cand_layout,
                    object_id=clutter_id,
                    x=px,
                    y=py,
                    yaw_rad=yaw,
                    length_m=length_m,
                    width_m=width_m,
                    height_m=clutter_h,
                )

                if not _quick_constraints_ok(
                    cand_layout,
                    moved_object_id=clutter_id,
                    obj_room_map=obj_room_map,
                    rooms=rooms,
                    main_room=main_room,
                    door_centers=door_centers,
                    constraints=constraints,
                    override_room_id=anchor_room_id,
                ):
                    continue

                metrics, debug = evaluate_layout(cand_layout, base_layout, eval_cfg)
                if int(metrics.get("validity", 0)) != 1:
                    continue

                raw_score = score_fn(metrics, debug)
                if raw_score is None:
                    continue
                anchor_dist = _distance((px, py), anchor)
                score = raw_score - anchor_bias_weight * anchor_dist

                action = {
                    "type": "add_clutter",
                    "index": clutter_index,
                    "object_id": clutter_id,
                    "size_xy_m": [length_m, width_m],
                    "height_m": clutter_h,
                    "pose_xytheta": [px, py, yaw],
                    "anchor_distance_m": anchor_dist,
                    "score_raw": raw_score,
                    "score_final": score,
                }
                candidates.append(CandidateResult(score=score, layout=cand_layout, metrics=metrics, debug=debug, action=action))

                if tested >= top_candidates_limit:
                    break
            if tested >= top_candidates_limit:
                break
        if tested >= top_candidates_limit:
            break

    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates[: max(1, top_k)]


def _score_bottleneck(
    metrics: Dict[str, Any],
    debug: Dict[str, Any],
    base_metrics: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
) -> Optional[float]:
    if int(metrics.get("validity", 0)) != 1:
        return None
    if bool(scenario_cfg.get("require_path", True)):
        path = debug.get("path_cells") or []
        if len(path) <= 0:
            return None

    tau_clr = _clearance_threshold(eval_cfg)
    clr = _clearance_value(metrics)
    target = tau_clr - as_float(scenario_cfg.get("target_below_tau_clr_m"), 0.02)
    min_allowed = tau_clr - as_float(scenario_cfg.get("max_below_tau_clr_m"), 0.08)
    if clr < min_allowed - 1e-9:
        return None

    base_reach = as_float(base_metrics.get("R_reach"), 0.0)
    reach = as_float(metrics.get("R_reach"), 0.0)
    if reach < max(0.25, base_reach - 0.5):
        return None

    below = 1.0 if clr < tau_clr else 0.0
    score = below * 10.0 - abs(clr - target) * 8.0
    score += (_clearance_value(base_metrics) - clr) * 2.0
    score -= as_float(metrics.get("Delta_layout"), 0.0) * 1.0
    return score


def _score_occlusion(
    metrics: Dict[str, Any],
    _debug: Dict[str, Any],
    base_metrics: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
) -> Optional[float]:
    if int(metrics.get("validity", 0)) != 1:
        return None

    keep = bool(scenario_cfg.get("keep_reach_and_clr", True))
    tau_r = as_float(eval_cfg.get("tau_R"), 0.9)
    tau_clr = _clearance_threshold(eval_cfg)
    if keep:
        if as_float(metrics.get("R_reach"), 0.0) + 1e-9 < tau_r:
            return None
        if _clearance_value(metrics) + 1e-9 < tau_clr:
            return None

    primary = str(scenario_cfg.get("primary_metric") or "OOE_R_rec_entry_surf")
    base_val = as_float(base_metrics.get(primary), 0.0)
    val = as_float(metrics.get(primary), 0.0)
    drop = base_val - val
    score = drop * 12.0 - as_float(metrics.get("Delta_layout"), 0.0) * 0.8
    return score


def _score_clutter(
    metrics: Dict[str, Any],
    _debug: Dict[str, Any],
    base_metrics: Dict[str, Any],
    _eval_cfg: Dict[str, Any],
    _scenario_cfg: Dict[str, Any],
) -> Optional[float]:
    if int(metrics.get("validity", 0)) != 1:
        return None
    reach_drop = as_float(base_metrics.get("R_reach"), 0.0) - as_float(metrics.get("R_reach"), 0.0)
    entry_drop = as_float(base_metrics.get("OOE_R_rec_entry_surf"), 0.0) - as_float(metrics.get("OOE_R_rec_entry_surf"), 0.0)
    entry_only = 1.0 if int(metrics.get("Adopt_core", 0)) == 1 and int(metrics.get("Adopt_entry", 0)) == 0 else 0.0
    score = entry_only * 8.0 + reach_drop * 9.0 + entry_drop * 5.0 - as_float(metrics.get("Delta_layout"), 0.0) * 0.8
    return score


def _choose_best_variant(candidates: Sequence[CandidateResult], fallback_metric: str) -> Optional[CandidateResult]:
    if not candidates:
        return None
    preferred = [c for c in candidates if int(c.metrics.get("validity", 0)) == 1]
    if preferred:
        return preferred[0]
    candidates = sorted(candidates, key=lambda c: as_float(c.metrics.get(fallback_metric), 0.0))
    return candidates[0] if candidates else None


def _run_bottleneck(
    base_layout: Dict[str, Any],
    base_metrics: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    degrade_cfg: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
    rooms: Dict[str, List[List[float]]],
    main_room: List[List[float]],
    obj_room_map: Dict[str, Optional[str]],
    start_xy: Vec2,
    goal_xy: Vec2,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    preferred = scenario_cfg.get("preferred_categories") or []
    object_ids = _movable_candidates(base_layout, degrade_cfg, preferred=preferred)
    anchor = _anchor_from_start_goal(start_xy, goal_xy, as_float(scenario_cfg.get("anchor_t"), 0.55))

    top = _best_single_move(
        base_layout=base_layout,
        start_layout=base_layout,
        base_metrics=base_metrics,
        eval_cfg=eval_cfg,
        degrade_cfg=degrade_cfg,
        rooms=rooms,
        main_room=main_room,
        obj_room_map=obj_room_map,
        door_centers=_collect_doors(base_layout),
        object_ids=object_ids,
        anchor=anchor,
        score_fn=lambda m, d: _score_bottleneck(m, d, base_metrics, eval_cfg, scenario_cfg),
        top_k=1,
    )
    best = _choose_best_variant(top, fallback_metric=_clearance_metric_key(base_metrics))
    if best is None:
        return _copy_layout(base_layout), base_metrics, {
            "scenario": "bottleneck",
            "status": "fallback_base",
            "actions": [],
            "selected_from_pool_index": 0,
            "pool_size": 1,
            "validity_checks": {"targeted_diag": True},
            "rejection_summary": {},
            "selection_notes": "No targeted bottleneck candidate was found; fell back to base layout.",
        }
    return best.layout, best.metrics, {
        "scenario": "bottleneck",
        "status": "ok",
        "actions": [best.action],
        "score": best.score,
        "selected_from_pool_index": 0,
        "pool_size": max(1, len(top)),
        "validity_checks": {"targeted_diag": True},
        "rejection_summary": {},
        "selection_notes": "Selected highest-scoring targeted bottleneck candidate.",
    }


def _run_occlusion(
    base_layout: Dict[str, Any],
    base_metrics: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    degrade_cfg: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
    rooms: Dict[str, List[List[float]]],
    main_room: List[List[float]],
    obj_room_map: Dict[str, Optional[str]],
    start_xy: Vec2,
    goal_xy: Vec2,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    preferred = scenario_cfg.get("preferred_categories") or []
    object_ids = _movable_candidates(base_layout, degrade_cfg, preferred=preferred)
    anchor = _anchor_from_start_goal(start_xy, goal_xy, as_float(scenario_cfg.get("anchor_t"), 0.20))

    top = _best_single_move(
        base_layout=base_layout,
        start_layout=base_layout,
        base_metrics=base_metrics,
        eval_cfg=eval_cfg,
        degrade_cfg=degrade_cfg,
        rooms=rooms,
        main_room=main_room,
        obj_room_map=obj_room_map,
        door_centers=_collect_doors(base_layout),
        object_ids=object_ids,
        anchor=anchor,
        score_fn=lambda m, d: _score_occlusion(m, d, base_metrics, eval_cfg, scenario_cfg),
        top_k=1,
    )
    best = _choose_best_variant(top, fallback_metric="C_vis_start")
    if best is None:
        return _copy_layout(base_layout), base_metrics, {
            "scenario": "occlusion",
            "status": "fallback_base",
            "actions": [],
            "selected_from_pool_index": 0,
            "pool_size": 1,
            "validity_checks": {"targeted_diag": True},
            "rejection_summary": {},
            "selection_notes": "No targeted occlusion candidate was found; fell back to base layout.",
        }
    return best.layout, best.metrics, {
        "scenario": "occlusion",
        "status": "ok",
        "actions": [best.action],
        "score": best.score,
        "selected_from_pool_index": 0,
        "pool_size": max(1, len(top)),
        "validity_checks": {"targeted_diag": True},
        "rejection_summary": {},
        "selection_notes": "Selected highest-scoring targeted occlusion candidate.",
    }


def _run_clutter(
    base_layout: Dict[str, Any],
    base_metrics: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    degrade_cfg: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
    rooms: Dict[str, List[List[float]]],
    main_room: List[List[float]],
    obj_room_map: Dict[str, Optional[str]],
    start_xy: Vec2,
    goal_xy: Vec2,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    max_added = int(as_float(scenario_cfg.get("max_added_objects"), 2))
    if max_added <= 0:
        return _copy_layout(base_layout), base_metrics, {"scenario": "clutter", "status": "fallback_base", "actions": []}
    search_cfg = degrade_cfg.get("search") or {}
    beam_width = int(as_float(search_cfg.get("clutter_beam_width"), 8))

    anchor1 = _anchor_from_start_goal(start_xy, goal_xy, as_float(scenario_cfg.get("anchor_t_1"), 0.45))
    first = _best_single_clutter_add(
        base_layout=base_layout,
        start_layout=base_layout,
        base_metrics=base_metrics,
        eval_cfg=eval_cfg,
        degrade_cfg=degrade_cfg,
        rooms=rooms,
        main_room=main_room,
        obj_room_map=obj_room_map,
        door_centers=_collect_doors(base_layout),
        anchor=anchor1,
        score_fn=lambda m, d: _score_clutter(m, d, base_metrics, eval_cfg, scenario_cfg),
        top_k=max(1, beam_width),
        clutter_index=0,
        scenario_cfg=scenario_cfg,
    )
    if not first:
        return _copy_layout(base_layout), base_metrics, {
            "scenario": "clutter",
            "status": "fallback_base",
            "actions": [],
            "selected_from_pool_index": 0,
            "pool_size": 1,
            "validity_checks": {"targeted_diag": True},
            "rejection_summary": {},
            "selection_notes": "No targeted clutter candidate was found; fell back to base layout.",
        }

    candidates: List[CandidateResult] = list(first)
    if max_added >= 2:
        anchor2 = _anchor_from_start_goal(start_xy, goal_xy, as_float(scenario_cfg.get("anchor_t_2"), 0.60))
        for base_c in first:
            second = _best_single_clutter_add(
                base_layout=base_layout,
                start_layout=base_c.layout,
                base_metrics=base_metrics,
                eval_cfg=eval_cfg,
                degrade_cfg=degrade_cfg,
                rooms=rooms,
                main_room=main_room,
                obj_room_map=obj_room_map,
                door_centers=_collect_doors(base_layout),
                anchor=anchor2,
                score_fn=lambda m, d: _score_clutter(m, d, base_metrics, eval_cfg, scenario_cfg),
                top_k=1,
                clutter_index=1,
                scenario_cfg=scenario_cfg,
            )
            if not second:
                continue
            s2 = second[0]
            combined_action = {"steps": [base_c.action, s2.action]}
            combined = CandidateResult(
                score=s2.score,
                layout=s2.layout,
                metrics=s2.metrics,
                debug=s2.debug,
                action=combined_action,
            )
            candidates.append(combined)

    target_mode = str(scenario_cfg.get("target_mode") or "entry_only_or_reach_drop")
    if target_mode == "entry_only_or_reach_drop":
        entry_only = [c for c in candidates if int(c.metrics.get("Adopt_core", 0)) == 1 and int(c.metrics.get("Adopt_entry", 0)) == 0]
        if entry_only:
            entry_only.sort(key=lambda c: c.score, reverse=True)
            best = entry_only[0]
            return best.layout, best.metrics, {
                "scenario": "clutter",
                "status": "ok_entry_only",
                "actions": best.action.get("steps") if isinstance(best.action.get("steps"), list) else [best.action],
                "score": best.score,
                "selected_from_pool_index": 0,
                "pool_size": max(1, len(candidates)),
                "validity_checks": {"targeted_diag": True},
                "rejection_summary": {},
                "selection_notes": "Selected targeted clutter candidate that preserves core adopt while dropping entry adopt.",
            }

    candidates.sort(key=lambda c: c.score, reverse=True)
    best = candidates[0]
    actions = best.action.get("steps") if isinstance(best.action.get("steps"), list) else [best.action]
    return best.layout, best.metrics, {
        "scenario": "clutter",
        "status": "ok",
        "actions": actions,
        "score": best.score,
        "selected_from_pool_index": 0,
        "pool_size": max(1, len(candidates)),
        "validity_checks": {"targeted_diag": True},
        "rejection_summary": {},
        "selection_notes": "Selected highest-scoring targeted clutter candidate.",
    }


def _generate_usage_shift_pool(
    *,
    base_layout: Dict[str, Any],
    base_metrics: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    stress_cfg: Dict[str, Any],
    family_cfg: Dict[str, Any],
    rooms: Dict[str, List[List[float]]],
    main_room: List[List[float]],
    obj_room_map: Dict[str, Optional[str]],
    rng: random.Random,
    fixed_object_count: Optional[int] = None,
) -> Tuple[List[NaturalCandidate], Dict[str, Any], Dict[str, Any]]:
    constraints = stress_cfg.get("constraints") or {}
    preferred_ids, low_ids = _usage_shift_candidates(base_layout, family_cfg)
    priors = family_cfg.get("priors_local_frame") if isinstance(family_cfg.get("priors_local_frame"), dict) else {}
    object_count = int(fixed_object_count) if fixed_object_count is not None else choose_count_from_distribution(
        rng,
        family_cfg.get("selection_distribution") if isinstance(family_cfg.get("selection_distribution"), list) else [],
        count_key="moved_object_count",
        default=1,
    )
    available_candidate_count = len(set(preferred_ids + low_ids))
    requested_object_count = object_count
    if available_candidate_count > 0:
        object_count = max(1, min(object_count, available_candidate_count))
    pool_target = 12
    max_attempts = 48
    rejection_counts: Dict[str, int] = {}
    pool: List[NaturalCandidate] = []
    attempts = 0
    door_centers = _collect_doors(base_layout)

    while len(pool) < pool_target and attempts < max_attempts:
        attempts += 1
        object_ids = sample_multiple_object_ids(rng, preferred_ids, low_ids, object_count)
        if len(object_ids) != object_count:
            rejection_counts["insufficient_objects"] = rejection_counts.get("insufficient_objects", 0) + 1
            continue
        cand_layout = _copy_layout(base_layout)
        actions: List[Dict[str, Any]] = []
        candidate_ok = True
        for oid in object_ids:
            obj = _find_object(cand_layout, oid)
            if not isinstance(obj, dict):
                candidate_ok = False
                rejection_counts["missing_object"] = rejection_counts.get("missing_object", 0) + 1
                break
            category = _object_category(obj)
            x0, y0 = _object_center(obj)
            yaw0 = _object_yaw(obj)
            placed = False
            for _ in range(12):
                dx_local, dy_local, dtheta_deg = sample_local_frame_delta(rng, priors, category)
                if abs(dx_local) + abs(dy_local) + abs(dtheta_deg) < 1e-6:
                    continue
                dx_world, dy_world = local_offset_to_world(dx_local, dy_local, yaw0)
                x1 = x0 + dx_world
                y1 = y0 + dy_world
                yaw1 = yaw0 + math.radians(dtheta_deg)
                _set_object_pose(obj, x1, y1, yaw1)
                if _quick_constraints_ok(
                    cand_layout,
                    moved_object_id=oid,
                    obj_room_map=obj_room_map,
                    rooms=rooms,
                    main_room=main_room,
                    door_centers=door_centers,
                    constraints=constraints,
                ):
                    actions.append(
                        {
                            "type": "move_object",
                            "object_id": oid,
                            "category": category,
                            "from_xy": [x0, y0],
                            "to_xy": [x1, y1],
                            "from_yaw": yaw0,
                            "to_yaw": yaw1,
                            "dx_local": dx_local,
                            "dy_local": dy_local,
                            "dtheta_deg": dtheta_deg,
                            "same_room_ok": True,
                        }
                    )
                    placed = True
                    break
                _set_object_pose(obj, x0, y0, yaw0)
            if not placed:
                candidate_ok = False
                rejection_counts["constraint_failure"] = rejection_counts.get("constraint_failure", 0) + 1
                break
        if not candidate_ok:
            continue
        metrics, debug = evaluate_layout(cand_layout, base_layout, eval_cfg)
        if int(metrics.get("validity", 0)) != 1:
            rejection_counts["eval_invalid"] = rejection_counts.get("eval_invalid", 0) + 1
            continue
        pool.append(
            NaturalCandidate(
                layout=cand_layout,
                metrics=metrics,
                debug=debug,
                actions=actions,
                selection_notes=(
                    f"Randomly sampled local-frame usage drift for {len(actions)} movable object(s). "
                    f"requested_count={requested_object_count}, available_candidates={available_candidate_count}, "
                    f"effective_count={object_count}."
                ),
            )
        )

    validity_checks = {
        "require_validity": bool(constraints.get("require_validity", True)),
        "same_room_only": bool(constraints.get("same_room_only", False)),
        "door_keepout_radius_m": as_float(constraints.get("door_keepout_radius_m"), 0.0),
        "overlap_ratio_max": as_float(constraints.get("overlap_ratio_max"), 0.0),
    }
    return pool, validity_checks, _pool_summary(len(pool), attempts, rejection_counts)


def _select_natural_candidate(pool: Sequence[NaturalCandidate], rng: random.Random) -> Tuple[Optional[NaturalCandidate], int]:
    if not pool:
        return None, 0
    idx = int(rng.randrange(len(pool)))
    return pool[idx], idx


def _run_usage_shift(
    *,
    base_layout: Dict[str, Any],
    base_metrics: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    stress_cfg: Dict[str, Any],
    family_cfg: Dict[str, Any],
    rooms: Dict[str, List[List[float]]],
    main_room: List[List[float]],
    obj_room_map: Dict[str, Optional[str]],
    rng: random.Random,
    fixed_object_count: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    pool, validity_checks, rejection_summary = _generate_usage_shift_pool(
        base_layout=base_layout,
        base_metrics=base_metrics,
        eval_cfg=eval_cfg,
        stress_cfg=stress_cfg,
        family_cfg=family_cfg,
        rooms=rooms,
        main_room=main_room,
        obj_room_map=obj_room_map,
        rng=rng,
        fixed_object_count=fixed_object_count,
    )
    selected, selected_idx = _select_natural_candidate(pool, rng)
    if selected is None:
        return _copy_layout(base_layout), base_metrics, {
            "scenario": "usage_shift",
            "status": "fallback_base",
            "actions": [],
            "selected_from_pool_index": 0,
            "pool_size": 1,
            "validity_checks": validity_checks,
            "rejection_summary": rejection_summary,
            "selection_notes": "No valid usage_shift candidate was found; fell back to base layout.",
        }
    return selected.layout, selected.metrics, {
        "scenario": "usage_shift",
        "status": "ok",
        "actions": selected.actions,
        "selected_from_pool_index": selected_idx,
        "pool_size": len(pool),
        "validity_checks": validity_checks,
        "rejection_summary": rejection_summary,
        "selection_notes": selected.selection_notes,
    }


def _generate_clutter_pool(
    *,
    base_layout: Dict[str, Any],
    start_layout: Dict[str, Any],
    base_metrics: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    stress_cfg: Dict[str, Any],
    family_cfg: Dict[str, Any],
    rooms: Dict[str, List[List[float]]],
    main_room: List[List[float]],
    obj_room_map: Dict[str, Optional[str]],
    start_xy: Vec2,
    goal_xy: Vec2,
    rng: random.Random,
    fixed_object_count: Optional[int] = None,
) -> Tuple[List[NaturalCandidate], Dict[str, Any], Dict[str, Any]]:
    constraints = stress_cfg.get("constraints") or {}
    count = int(fixed_object_count) if fixed_object_count is not None else choose_count_from_distribution(
        rng,
        family_cfg.get("count_distribution") if isinstance(family_cfg.get("count_distribution"), list) else [],
        count_key="added_object_count",
        default=1,
    )
    pool_target = 12
    max_attempts = 48
    door_centers = _collect_doors(base_layout)
    rejection_counts: Dict[str, int] = {}
    pool: List[NaturalCandidate] = []
    attempts = 0
    catalog = family_cfg.get("object_catalog") if isinstance(family_cfg.get("object_catalog"), list) else []
    placement_bands = family_cfg.get("placement_bands") if isinstance(family_cfg.get("placement_bands"), list) else []
    pose_jitter = family_cfg.get("pose_jitter") if isinstance(family_cfg.get("pose_jitter"), dict) else {}
    second_rule = str(pose_jitter.get("second_object_rule") or "")

    while len(pool) < pool_target and attempts < max_attempts:
        attempts += 1
        cand_layout = _copy_layout(start_layout)
        actions: List[Dict[str, Any]] = []
        used_band_ids: List[str] = []
        candidate_ok = True
        for clutter_idx in range(count):
            band_candidates = placement_bands
            if clutter_idx >= 1 and "prefer_different_band" in second_rule and used_band_ids:
                filtered = [b for b in placement_bands if str(b.get("band_id") or "") not in used_band_ids]
                if filtered:
                    band_candidates = filtered
            band_cfg = choose_weighted_item(rng, band_candidates, weight_key="selection_probability")
            catalog_item = choose_weighted_item(rng, catalog, weight_key="weight")
            if not isinstance(band_cfg, dict) or not isinstance(catalog_item, dict):
                candidate_ok = False
                rejection_counts["invalid_band_or_catalog"] = rejection_counts.get("invalid_band_or_catalog", 0) + 1
                break
            (px, py), band_id = _sample_band_point(
                rng=rng,
                band_cfg=band_cfg,
                start_xy=start_xy,
                goal_xy=goal_xy,
                layout=cand_layout,
            )
            yaw = _sample_clutter_yaw(rng, family_cfg)
            size_xy = catalog_item.get("size_xy_m") if isinstance(catalog_item.get("size_xy_m"), list) else [0.35, 0.35]
            length_m = max(0.1, as_float(size_xy[0], 0.35))
            width_m = max(0.1, as_float(size_xy[1], 0.35))
            height_m = max(0.1, as_float(catalog_item.get("height_m"), 1.0))
            clutter_id = _next_clutter_id(cand_layout)
            _append_clutter_object(
                cand_layout,
                object_id=clutter_id,
                x=px,
                y=py,
                yaw_rad=yaw,
                length_m=length_m,
                width_m=width_m,
                height_m=height_m,
            )
            override_room_id = _room_for_point(rooms, (px, py))
            if override_room_id is None and len(main_room) >= 3 and point_in_polygon(px, py, main_room):
                override_room_id = "main_room"
            if not _quick_constraints_ok(
                cand_layout,
                moved_object_id=clutter_id,
                obj_room_map=obj_room_map,
                rooms=rooms,
                main_room=main_room,
                door_centers=door_centers,
                constraints=constraints,
                override_room_id=override_room_id,
            ):
                candidate_ok = False
                rejection_counts["constraint_failure"] = rejection_counts.get("constraint_failure", 0) + 1
                break
            actions.append(
                {
                    "type": "add_clutter",
                    "index": clutter_idx,
                    "object_id": clutter_id,
                    "size_xy_m": [length_m, width_m],
                    "height_m": height_m,
                    "pose_xytheta": [px, py, yaw],
                    "band_id": band_id,
                }
            )
            used_band_ids.append(band_id)
        if not candidate_ok:
            continue
        metrics, debug = evaluate_layout(cand_layout, base_layout, eval_cfg)
        if int(metrics.get("validity", 0)) != 1:
            rejection_counts["eval_invalid"] = rejection_counts.get("eval_invalid", 0) + 1
            continue
        pool.append(
            NaturalCandidate(
                layout=cand_layout,
                metrics=metrics,
                debug=debug,
                actions=actions,
                selection_notes=f"Placed {len(actions)} clutter object(s) from geometric placement bands.",
            )
        )

    validity_checks = {
        "require_validity": bool(constraints.get("require_validity", True)),
        "same_room_only": bool(constraints.get("same_room_only", False)),
        "door_keepout_radius_m": as_float(constraints.get("door_keepout_radius_m"), 0.0),
        "overlap_ratio_max": as_float(constraints.get("overlap_ratio_max"), 0.0),
    }
    return pool, validity_checks, _pool_summary(len(pool), attempts, rejection_counts)


def _run_clutter_natural(
    *,
    base_layout: Dict[str, Any],
    base_metrics: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    stress_cfg: Dict[str, Any],
    family_cfg: Dict[str, Any],
    rooms: Dict[str, List[List[float]]],
    main_room: List[List[float]],
    obj_room_map: Dict[str, Optional[str]],
    start_xy: Vec2,
    goal_xy: Vec2,
    rng: random.Random,
    fixed_object_count: Optional[int] = None,
    start_layout: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    seed_layout = _copy_layout(start_layout if isinstance(start_layout, dict) else base_layout)
    pool, validity_checks, rejection_summary = _generate_clutter_pool(
        base_layout=base_layout,
        start_layout=seed_layout,
        base_metrics=base_metrics,
        eval_cfg=eval_cfg,
        stress_cfg=stress_cfg,
        family_cfg=family_cfg,
        rooms=rooms,
        main_room=main_room,
        obj_room_map=obj_room_map,
        start_xy=start_xy,
        goal_xy=goal_xy,
        rng=rng,
        fixed_object_count=fixed_object_count,
    )
    selected, selected_idx = _select_natural_candidate(pool, rng)
    if selected is None:
        fallback_layout = _copy_layout(seed_layout)
        fallback_metrics, _ = evaluate_layout(fallback_layout, base_layout, eval_cfg)
        return fallback_layout, fallback_metrics, {
            "scenario": "clutter",
            "status": "fallback_seed_layout" if start_layout is not None else "fallback_base",
            "actions": [],
            "selected_from_pool_index": 0,
            "pool_size": 1,
            "validity_checks": validity_checks,
            "rejection_summary": rejection_summary,
            "selection_notes": "No valid clutter candidate was found; fell back without added clutter.",
        }
    return selected.layout, selected.metrics, {
        "scenario": "clutter",
        "status": "ok",
        "actions": selected.actions,
        "selected_from_pool_index": selected_idx,
        "pool_size": len(pool),
        "validity_checks": validity_checks,
        "rejection_summary": rejection_summary,
        "selection_notes": selected.selection_notes,
    }


def _run_compound(
    *,
    base_layout: Dict[str, Any],
    base_metrics: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    stress_cfg: Dict[str, Any],
    family_cfg: Dict[str, Any],
    rooms: Dict[str, List[List[float]]],
    main_room: List[List[float]],
    obj_room_map: Dict[str, Optional[str]],
    start_xy: Vec2,
    goal_xy: Vec2,
    rng: random.Random,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    families = stress_cfg.get("families") if isinstance(stress_cfg.get("families"), dict) else {}
    usage_cfg = families.get("usage_shift") if isinstance(families.get("usage_shift"), dict) else {}
    clutter_cfg = families.get("clutter") if isinstance(families.get("clutter"), dict) else {}
    usage_count = int(as_float(family_cfg.get("usage_shift_object_count"), 1))
    clutter_count = int(as_float(family_cfg.get("clutter_object_count"), 1))

    usage_layout, _usage_metrics, usage_disturb = _run_usage_shift(
        base_layout=base_layout,
        base_metrics=base_metrics,
        eval_cfg=eval_cfg,
        stress_cfg=stress_cfg,
        family_cfg=usage_cfg,
        rooms=rooms,
        main_room=main_room,
        obj_room_map=obj_room_map,
        rng=random.Random(rng.randint(0, 2**31 - 1)),
        fixed_object_count=usage_count,
    )
    clutter_layout, clutter_metrics, clutter_disturb = _run_clutter_natural(
        base_layout=base_layout,
        base_metrics=base_metrics,
        eval_cfg=eval_cfg,
        stress_cfg=stress_cfg,
        family_cfg=clutter_cfg,
        rooms=rooms,
        main_room=main_room,
        obj_room_map=obj_room_map,
        start_xy=start_xy,
        goal_xy=goal_xy,
        rng=random.Random(rng.randint(0, 2**31 - 1)),
        fixed_object_count=clutter_count,
        start_layout=usage_layout,
    )

    combined_actions = []
    combined_actions.extend(usage_disturb.get("actions") if isinstance(usage_disturb.get("actions"), list) else [])
    combined_actions.extend(clutter_disturb.get("actions") if isinstance(clutter_disturb.get("actions"), list) else [])
    return clutter_layout, clutter_metrics, {
        "scenario": "compound",
        "status": "ok" if combined_actions else "fallback_base",
        "actions": combined_actions,
        "selected_from_pool_index": 0,
        "pool_size": max(int(usage_disturb.get("pool_size", 1) or 1), 1) * max(int(clutter_disturb.get("pool_size", 1) or 1), 1),
        "validity_checks": {
            "usage_shift": usage_disturb.get("validity_checks", {}),
            "clutter": clutter_disturb.get("validity_checks", {}),
        },
        "rejection_summary": {
            "usage_shift": usage_disturb.get("rejection_summary", {}),
            "clutter": clutter_disturb.get("rejection_summary", {}),
        },
        "selection_notes": "Compound sample created by applying one usage_shift candidate and one clutter candidate sequentially.",
    }


def _delta_disturb_for_variant(
    *,
    variant: str,
    variant_metrics: Dict[str, Any],
    disturb: Dict[str, Any],
    stress_cfg: Dict[str, Any],
) -> float:
    base = as_float(variant_metrics.get("Delta_layout"), 0.0)
    mode = _config_mode(stress_cfg)
    if mode == "natural_main":
        return base
    if variant not in {"clutter", "targeted_clutter"}:
        return base
    degrade_cfg = stress_cfg
    scfg = (degrade_cfg.get("scenarios") or {}).get("clutter") or {}
    per_added = as_float(scfg.get("delta_disturb_per_added_object"), 0.02)
    actions = disturb.get("actions") if isinstance(disturb.get("actions"), list) else []
    added = 0
    for a in actions:
        if not isinstance(a, dict):
            continue
        if str(a.get("type") or "").strip().lower() == "add_clutter":
            added += 1
    return base + per_added * float(added)


def _target_metric_for_variant(variant: str, scenario_cfg: Dict[str, Any], stress_cfg: Dict[str, Any]) -> str:
    v = str(variant or "").strip().lower()
    if _config_mode(stress_cfg) == "natural_main":
        if v in {"usage_shift", "compound"}:
            return "post_generation_only"
        if v == "clutter":
            return "post_generation_only"
        return "none"
    if v == "targeted_bottleneck":
        return "clr_feasible"
    if v == "targeted_occlusion":
        return str(scenario_cfg.get("primary_metric") or "OOE_R_rec_entry_surf")
    if v == "bottleneck":
        return "clr_feasible"
    if v == "occlusion":
        return str(scenario_cfg.get("primary_metric") or "OOE_R_rec_entry_surf")
    if v == "clutter":
        return "OOE_R_rec_entry_surf_or_R_reach"
    return "none"


def _delta_metrics(base_metrics: Dict[str, Any], variant_metrics: Dict[str, Any]) -> Dict[str, float]:
    keys = [
        "C_vis",
        "C_vis_start",
        "R_reach",
        "clr_min",
        "clr_min_astar",
        "clr_feasible",
        "Delta_layout",
        "OOE_C_obj_entry_hit",
        "OOE_R_rec_entry_hit",
        "OOE_C_obj_entry_surf",
        "OOE_R_rec_entry_surf",
        "Adopt_core",
        "Adopt_entry",
        "validity",
    ]
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = as_float(variant_metrics.get(k), 0.0) - as_float(base_metrics.get(k), 0.0)
    return out


def _save_variant_layout(
    out_root: pathlib.Path,
    case_name: str,
    variant: str,
    stress_family: str,
    layout: Dict[str, Any],
    source_layout_path: pathlib.Path,
    base_metrics: Dict[str, Any],
    variant_metrics: Dict[str, Any],
    disturb: Dict[str, Any],
    delta_disturb: float,
    target_metric: str,
    eval_cfg: Dict[str, Any],
    stress_cfg: Dict[str, Any],
    stress_config_path: pathlib.Path,
    stress_config_hash: str,
    repo_root: pathlib.Path,
    sample_seed: int,
) -> pathlib.Path:
    out_dir = out_root / case_name / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = _copy_layout(layout)
    for obj in saved.get("objects", []):
        if not isinstance(obj, dict):
            continue
        cat = str(obj.get("category") or "").strip().lower()
        obj["origin"] = "stress_added" if cat == "clutter" else "original"
        obj["stress_kind"] = stress_family
        obj["source_case_id"] = case_name
        obj["stress_case_id"] = f"{case_name}__{variant}"
        if cat == "clutter":
            obj["object_role"] = "clutter"
            obj["movable_in_main"] = False
            obj["movable_in_clutter_recovery"] = True
        else:
            obj["object_role"] = "furniture"
            obj["movable_in_main"] = bool(obj.get("movable", True))
            obj["movable_in_clutter_recovery"] = False
    meta = dict(saved.get("meta") or {})
    meta["layout_id"] = f"{case_name}__{variant}"
    meta["source"] = "generate_stress_cases.py"
    meta["source_layout_path"] = str(source_layout_path)
    meta["stress_variant"] = variant
    meta["stress_manifest"] = disturb
    saved["meta"] = meta

    out_layout = out_dir / "layout_generated.json"
    write_json(out_layout, saved)
    write_json(
        out_dir / "disturb_manifest.json",
        {
            "generated_at": utc_now_iso(),
            "case_name": case_name,
            "variant": variant,
            "source_layout_path": str(source_layout_path),
            "base_metrics": base_metrics,
            "variant_metrics": variant_metrics,
            "Delta_disturb": delta_disturb,
            "disturb": disturb,
        },
    )
    sample_manifest = build_sample_manifest(
        stress_cfg=stress_cfg,
        stress_config_path=stress_config_path,
        stress_config_hash=stress_config_hash,
        repo_root=repo_root,
        base_case_id=case_name,
        scene_id=f"{case_name}__{variant}",
        stress_family=stress_family,
        seed=sample_seed,
        base_metrics=base_metrics,
        variant_metrics=variant_metrics,
        disturb={**(disturb if isinstance(disturb, dict) else {}), "Delta_disturb": delta_disturb},
        eval_cfg=eval_cfg,
    )
    sample_manifest["generated_at"] = utc_now_iso()
    sample_manifest["case_id"] = f"{case_name}__{variant}"
    sample_manifest["variant"] = variant
    sample_manifest["stress_type"] = stress_family
    sample_manifest["target_metric"] = str(target_metric or "none")
    sample_manifest["source_layout_path"] = str(source_layout_path)
    sample_manifest["status"] = str(disturb.get("status") or "unknown") if isinstance(disturb, dict) else "unknown"
    sample_manifest["score"] = as_float(disturb.get("score"), 0.0) if isinstance(disturb, dict) else 0.0
    write_json(out_dir / "stress_manifest.json", sample_manifest)
    return out_layout


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate controlled-perturbation stress-test layouts.")
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--layout_filename", default="layout_generated.json")
    parser.add_argument("--eval_config", default="experiments/configs/eval/eval_v1.json")
    parser.add_argument("--degrade_config", default="experiments/configs/stress/stress_v2_natural.json")
    parser.add_argument("--case_names", default="")
    parser.add_argument("--seed", type=int, default=-1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = pathlib.Path(__file__).resolve().parents[2]

    source_root = pathlib.Path(args.source_root)
    if not source_root.is_absolute():
        source_root = (repo_root / source_root).resolve()
    out_root = pathlib.Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (repo_root / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    eval_cfg_path = pathlib.Path(args.eval_config)
    if not eval_cfg_path.is_absolute():
        eval_cfg_path = (repo_root / eval_cfg_path).resolve()
    degrade_cfg_path = pathlib.Path(args.degrade_config)
    if not degrade_cfg_path.is_absolute():
        degrade_cfg_path = (repo_root / degrade_cfg_path).resolve()

    eval_cfg = merge_eval_config(default_eval_config(), _load_json(eval_cfg_path))
    degrade_cfg = _load_json(degrade_cfg_path)
    stress_mode = _config_mode(degrade_cfg)
    reference_cfg = degrade_cfg
    if stress_mode == "targeted_diag":
        ref_path_raw = degrade_cfg.get("reference_config_path")
        if ref_path_raw:
            ref_path = pathlib.Path(str(ref_path_raw))
            if not ref_path.is_absolute():
                ref_path = (repo_root / ref_path).resolve()
            reference_cfg = _load_json(ref_path)

    seed_base = int(as_float(degrade_cfg.get("seed"), 20260225))
    if args.seed >= 0:
        seed_base = int(args.seed)

    case_names = [x.strip() for x in str(args.case_names).split(",") if x.strip()]
    case_dirs = _iter_case_dirs(source_root, args.layout_filename, case_names)

    variants = _config_variants(degrade_cfg)
    scenarios_cfg = _legacy_scenarios_cfg(degrade_cfg)
    stress_config_hash = sha256_file(degrade_cfg_path)
    sample_schema_path = (repo_root / "experiments/configs/stress/stress_sample_manifest_schema.json").resolve()
    sample_schema = _load_json(sample_schema_path)

    entries: List[Dict[str, Any]] = []
    scenario_counts: Dict[str, int] = {v: 0 for v in variants}

    for case_idx, case_dir in enumerate(case_dirs):
        case_name = case_dir.name
        source_layout_path = case_dir / args.layout_filename
        base_layout = load_layout_contract(source_layout_path)
        base_metrics, base_debug = evaluate_layout(base_layout, base_layout, eval_cfg)

        rooms, main_room = _build_room_registry(base_layout)
        room_map = _assign_object_rooms(base_layout, rooms, main_room)
        start_xy, goal_xy = _task_start_goal(base_layout, base_debug, main_room)

        base_seed = _stable_seed(seed_base, case_idx, case_name, "base")
        base_disturb = {
            "scenario": "base",
            "status": "base",
            "actions": [],
            "selected_from_pool_index": 0,
            "pool_size": 1,
            "validity_checks": {"base_layout": True},
            "rejection_summary": {},
            "selection_notes": "No perturbation applied.",
        }
        base_out = _save_variant_layout(
            out_root=out_root,
            case_name=case_name,
            variant="base",
            stress_family="base",
            layout=base_layout,
            source_layout_path=source_layout_path,
            base_metrics=base_metrics,
            variant_metrics=base_metrics,
            disturb=base_disturb,
            delta_disturb=as_float(base_metrics.get("Delta_layout"), 0.0),
            target_metric="none",
            eval_cfg=eval_cfg,
            stress_cfg=degrade_cfg,
            stress_config_path=degrade_cfg_path,
            stress_config_hash=stress_config_hash,
            repo_root=repo_root,
            sample_seed=base_seed,
        )
        entries.append(
            {
                "base_case": case_name,
                "variant": "base",
                "scenario": "base",
                "layout_id": f"{case_name}__base",
                "layout_path": str(base_out),
                "stress_manifest_path": str(base_out.parent / "stress_manifest.json"),
                "source_layout_path": str(source_layout_path),
                "base_metrics": base_metrics,
                "variant_metrics": base_metrics,
                "Delta_disturb": as_float(base_metrics.get("Delta_layout"), 0.0),
                "disturb": base_disturb,
            }
        )
        scenario_counts["base"] = scenario_counts.get("base", 0) + 1

        for variant in variants:
            if variant == "base":
                continue
            local_seed = _stable_seed(seed_base, case_idx, case_name, variant)
            rng = random.Random(local_seed)

            if stress_mode == "natural_main":
                scfg = _natural_family_cfg(degrade_cfg, variant)
            elif stress_mode == "targeted_diag":
                scfg = _targeted_scenario_cfg(degrade_cfg, variant)
            else:
                scfg = scenarios_cfg.get(variant) if isinstance(scenarios_cfg.get(variant), dict) else {}
            if not scfg and variant != "base":
                continue

            if stress_mode == "natural_main" and variant == "usage_shift":
                var_layout, var_metrics, disturb = _run_usage_shift(
                    base_layout=base_layout,
                    base_metrics=base_metrics,
                    eval_cfg=eval_cfg,
                    stress_cfg=degrade_cfg,
                    family_cfg=scfg,
                    rooms=rooms,
                    main_room=main_room,
                    obj_room_map=room_map,
                    rng=rng,
                )
            elif stress_mode == "natural_main" and variant == "clutter":
                var_layout, var_metrics, disturb = _run_clutter_natural(
                    base_layout=base_layout,
                    base_metrics=base_metrics,
                    eval_cfg=eval_cfg,
                    stress_cfg=degrade_cfg,
                    family_cfg=scfg,
                    rooms=rooms,
                    main_room=main_room,
                    obj_room_map=room_map,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                    rng=rng,
                )
            elif stress_mode == "natural_main" and variant == "compound":
                var_layout, var_metrics, disturb = _run_compound(
                    base_layout=base_layout,
                    base_metrics=base_metrics,
                    eval_cfg=eval_cfg,
                    stress_cfg=degrade_cfg,
                    family_cfg=scfg,
                    rooms=rooms,
                    main_room=main_room,
                    obj_room_map=room_map,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                    rng=rng,
                )
            elif variant in {"bottleneck", "targeted_bottleneck"}:
                var_layout, var_metrics, disturb = _run_bottleneck(
                    base_layout=base_layout,
                    base_metrics=base_metrics,
                    eval_cfg=eval_cfg,
                    degrade_cfg=reference_cfg,
                    scenario_cfg=scfg,
                    rooms=rooms,
                    main_room=main_room,
                    obj_room_map=room_map,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                )
            elif variant in {"occlusion", "targeted_occlusion"}:
                var_layout, var_metrics, disturb = _run_occlusion(
                    base_layout=base_layout,
                    base_metrics=base_metrics,
                    eval_cfg=eval_cfg,
                    degrade_cfg=reference_cfg,
                    scenario_cfg=scfg,
                    rooms=rooms,
                    main_room=main_room,
                    obj_room_map=room_map,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                )
            elif variant == "clutter":
                var_layout, var_metrics, disturb = _run_clutter(
                    base_layout=base_layout,
                    base_metrics=base_metrics,
                    eval_cfg=eval_cfg,
                    degrade_cfg=reference_cfg,
                    scenario_cfg=scfg,
                    rooms=rooms,
                    main_room=main_room,
                    obj_room_map=room_map,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                )
            else:
                continue

            stress_family = _stress_family_name(stress_mode, variant)
            delta_disturb = _delta_disturb_for_variant(
                variant=variant,
                variant_metrics=var_metrics,
                disturb=disturb,
                stress_cfg=reference_cfg if stress_mode == "legacy_targeted" else degrade_cfg,
            )
            out_path = _save_variant_layout(
                out_root=out_root,
                case_name=case_name,
                variant=variant,
                stress_family=stress_family,
                layout=var_layout,
                source_layout_path=source_layout_path,
                base_metrics=base_metrics,
                variant_metrics=var_metrics,
                disturb=disturb,
                delta_disturb=delta_disturb,
                target_metric=_target_metric_for_variant(variant, scfg, degrade_cfg),
                eval_cfg=eval_cfg,
                stress_cfg=degrade_cfg,
                stress_config_path=degrade_cfg_path,
                stress_config_hash=stress_config_hash,
                repo_root=repo_root,
                sample_seed=local_seed,
            )
            entries.append(
                {
                    "base_case": case_name,
                    "variant": variant,
                    "scenario": stress_family,
                    "layout_id": f"{case_name}__{variant}",
                    "layout_path": str(out_path),
                    "stress_manifest_path": str(out_path.parent / "stress_manifest.json"),
                    "source_layout_path": str(source_layout_path),
                    "base_metrics": base_metrics,
                    "variant_metrics": var_metrics,
                    "Delta_disturb": delta_disturb,
                    "disturb": disturb,
                }
            )
            scenario_counts[variant] = scenario_counts.get(variant, 0) + 1

    dataset_qa_report = build_dataset_qa_report(
        out_root=out_root,
        stress_cfg=degrade_cfg,
        stress_config_path=degrade_cfg_path,
        sample_schema=sample_schema,
        entries=entries,
        repo_root=repo_root,
    )
    dataset_qa_path = out_root / "stress_dataset_qa_report.json"
    write_dataset_qa_report(dataset_qa_path, dataset_qa_report)

    manifest = {
        "generated_at": utc_now_iso(),
        "generator": "generate_stress_cases.py",
        "source_root": str(source_root),
        "out_root": str(out_root),
        "layout_filename": args.layout_filename,
        "eval_config": str(eval_cfg_path),
        "degrade_config": str(degrade_cfg_path),
        "stress_mode": stress_mode,
        "stress_version": str(degrade_cfg.get("stress_version") or "legacy_targeted"),
        "stress_family": str(degrade_cfg.get("stress_family") or stress_mode),
        "stress_config_hash": stress_config_hash,
        "seed": seed_base,
        "variants": variants,
        "total_base_cases": len(case_dirs),
        "total_variants": len(entries),
        "scenario_counts": scenario_counts,
        "dataset_qa_report": str(dataset_qa_path),
        "entries": entries,
    }
    write_json(out_root / "stress_cases_manifest.json", manifest)
    print(
        {
            "out_root": str(out_root),
            "manifest": str(out_root / "stress_cases_manifest.json"),
            "total_base_cases": len(case_dirs),
            "total_variants": len(entries),
            "scenario_counts": scenario_counts,
        }
    )


if __name__ == "__main__":
    main()
