from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from eval_metrics import default_eval_config, evaluate_layout, merge_eval_config
from layout_tools import (
    angle_diff_rad,
    as_float,
    load_layout_contract,
    obb_corners_xy,
    obb_sample_points_xy,
    orthogonal_deviation_rad,
    orthogonal_yaws,
    point_in_polygon,
    room_polygon_for_object,
    room_axis_for_object,
    write_json,
)


Vec2 = Tuple[float, float]


@dataclass
class BeamNode:
    layout: Dict[str, Any]
    metrics: Dict[str, Any]
    debug: Dict[str, Any]
    changed_ids: Set[str]
    steps: List[Dict[str, Any]]
    score_tuple: Tuple[int, int, int, float]


def _recovery_protocol(config: Dict[str, Any]) -> str:
    return str((config or {}).get("recovery_protocol", "layout_only") or "layout_only").strip().lower()


def _find_object(layout: Dict[str, Any], object_id: str) -> Optional[Dict[str, Any]]:
    for obj in layout.get("objects", []):
        if str(obj.get("id") or "") == object_id:
            return obj
    return None


def _object_center(obj: Dict[str, Any]) -> Vec2:
    pose = obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]
    return as_float(pose[0], 0.0), as_float(pose[1], 0.0)


def _object_inside_room(obj: Dict[str, Any], room_poly: Sequence[Sequence[float]]) -> bool:
    for x, y in obb_sample_points_xy(obj, spacing_m=0.05, include_center=True):
        if point_in_polygon(x, y, room_poly):
            continue
        on_edge = False
        n = len(room_poly)
        for i in range(n):
            ax, ay = room_poly[i]
            bx, by = room_poly[(i + 1) % n]
            if _point_to_segment_distance(x, y, ax, ay, bx, by) <= 1e-4:
                on_edge = True
                break
        if not on_edge:
            return False
    return True


def _room_edge_clearance(obj: Dict[str, Any], room_poly: Sequence[Sequence[float]]) -> float:
    if not room_poly or len(room_poly) < 2:
        return float("inf")
    corners = obb_sample_points_xy(obj, spacing_m=0.05, include_center=True)
    best = float("inf")
    n = len(room_poly)
    for x, y in corners:
        for i in range(n):
            ax, ay = room_poly[i]
            bx, by = room_poly[(i + 1) % n]
            d = _point_to_segment_distance(x, y, ax, ay, bx, by)
            if d < best:
                best = d
    return best


def _collect_door_centers(layout: Dict[str, Any]) -> List[Vec2]:
    out: List[Vec2] = []
    for obj in layout.get("objects", []):
        if str(obj.get("category") or "").strip().lower() == "door":
            out.append(_object_center(obj))
    return out


def _distance(a: Vec2, b: Vec2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


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


def _has_excess_overlap(layout: Dict[str, Any], moved_object_id: str, max_ratio: float) -> bool:
    obj = _find_object(layout, moved_object_id)
    if obj is None:
        return True
    cat = str(obj.get("category") or "").strip().lower()
    if cat in {"floor", "door", "window", "opening"}:
        return False
    a = _obb_aabb(obj)
    for other in layout.get("objects", []):
        oid = str(other.get("id") or "")
        if oid == moved_object_id:
            continue
        ocat = str(other.get("category") or "").strip().lower()
        if ocat in {"floor", "door", "window", "opening"}:
            continue
        b = _obb_aabb(other)
        inter = _aabb_intersection(a, b)
        if inter <= 0.0:
            continue
        den = min(_aabb_area(a), _aabb_area(b))
        if den > 1e-9 and inter / den > max_ratio + 1e-9:
            return True
    return False


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _margin(value: float, tau: float) -> float:
    den = abs(tau) if abs(tau) > 1e-9 else 1.0
    return _clip((value - tau) / den, -1.0, 1.0)


def _clearance_metric_name(config: Dict[str, Any]) -> str:
    adopt_cfg = config.get("adopt") if isinstance(config.get("adopt"), dict) else {}
    raw = str(adopt_cfg.get("clearance_metric") or "clr_feasible").strip().lower()
    if raw in {"clr_min", "clr_min_astar", "astar", "path"}:
        return "clr_min_astar"
    return "clr_feasible"


def _clearance_threshold(config: Dict[str, Any], metric_name: str) -> float:
    tau_clr = as_float(config.get("tau_clr"), 0.2)
    if metric_name == "clr_min_astar":
        return as_float(config.get("tau_clr_astar"), tau_clr)
    return as_float(config.get("tau_clr_feasible"), tau_clr)


def _clearance_value(metrics: Dict[str, Any], metric_name: str) -> float:
    if metric_name == "clr_min_astar":
        return as_float(metrics.get("clr_min_astar", metrics.get("clr_min")), 0.0)
    if "clr_feasible" in metrics:
        return as_float(metrics.get("clr_feasible"), 0.0)
    return as_float(metrics.get("clr_min_astar", metrics.get("clr_min")), 0.0)


def _continuous_score(
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    *,
    ooe_primary: str,
    delta_weight: float,
) -> float:
    clearance_metric = _clearance_metric_name(config)
    tau_clr = _clearance_threshold(config, clearance_metric)
    tau_r = as_float(config.get("tau_R"), 0.9)
    tau_v = as_float(config.get("tau_V"), 0.4)

    adopt_cfg = config.get("adopt") if isinstance(config.get("adopt"), dict) else {}
    entry_gate = adopt_cfg.get("entry_gate") if isinstance(adopt_cfg.get("entry_gate"), dict) else {}
    tau_ooe = as_float(entry_gate.get("min_value"), 0.8)

    m_clr = _margin(_clearance_value(metrics, clearance_metric), tau_clr)
    m_r = _margin(as_float(metrics.get("R_reach"), 0.0), tau_r)
    m_vis = _margin(as_float(metrics.get("C_vis"), 0.0), tau_v)
    m_start = _margin(as_float(metrics.get("C_vis_start"), 0.0), tau_v)
    m_ooe = _margin(as_float(metrics.get(ooe_primary), 0.0), tau_ooe)
    p_delta = as_float(metrics.get("Delta_layout"), 0.0)

    # Conservative defaults; lexicographic ordering is primary.
    return (
        1.0 * m_clr
        + 1.0 * m_r
        + 0.2 * m_vis
        + 1.0 * m_start
        + 1.0 * m_ooe
        - delta_weight * p_delta
    )


def _alignment_prior_cfg(config: Dict[str, Any], protocol: str) -> Optional[Dict[str, Any]]:
    if protocol == "clutter_assisted":
        return None
    raw = config.get("layout_axis_alignment_prior")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return None
    protocols = raw.get("apply_to_protocols")
    if isinstance(protocols, list) and protocols:
        allowed = {str(x).strip().lower() for x in protocols}
        if protocol not in allowed:
            return None
    return raw


def _is_rectangular_table(obj: Dict[str, Any]) -> bool:
    cat = str(obj.get("category") or "").strip().lower()
    if cat != "table":
        return False
    size = obj.get("size_lwh_m") or [1.0, 1.0, 1.0]
    return abs(as_float(size[0], 1.0) - as_float(size[1], 1.0)) > 0.1


def _use_alignment_prior(obj: Dict[str, Any], prior_cfg: Optional[Dict[str, Any]]) -> bool:
    if prior_cfg is None:
        return False
    cat = str(obj.get("category") or "").strip().lower()
    if cat in {"clutter", "door", "window", "opening", "toilet", "floor"}:
        return False
    if cat == "chair":
        return bool(prior_cfg.get("apply_to_chair", True))
    if cat == "table":
        return _is_rectangular_table(obj)
    target_cats = {
        "bed",
        "sofa",
        "storage",
        "tv_cabinet",
        "sink",
        "cabinet",
        "coffee_table",
        "small_storage",
    }
    return cat in target_cats


def _strict_orthogonal_category(obj: Dict[str, Any], prior_cfg: Optional[Dict[str, Any]]) -> bool:
    if prior_cfg is None:
        return False
    cat = str(obj.get("category") or "").strip().lower()
    strict = prior_cfg.get("strict_orthogonal_categories")
    if isinstance(strict, list):
        return cat in {str(x).strip().lower() for x in strict}
    return cat in {"sink", "storage", "tv_cabinet", "cabinet"}


def _candidate_yaws(layout: Dict[str, Any], obj: Dict[str, Any], config: Dict[str, Any], protocol: str, rot_deg: float) -> List[float]:
    pose = obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]
    current_yaw = as_float(pose[3], 0.0)
    prior_cfg = _alignment_prior_cfg(config, protocol)
    if not _use_alignment_prior(obj, prior_cfg):
        rot_rad = math.radians(rot_deg)
        return [current_yaw - rot_rad, current_yaw, current_yaw + rot_rad]

    axis = room_axis_for_object(layout, obj)
    candidates = orthogonal_yaws(axis)
    if _strict_orthogonal_category(obj, prior_cfg):
        return candidates
    max_off_axis_deg = as_float(prior_cfg.get("max_off_axis_deg"), 15.0)
    off_axis_degrees = prior_cfg.get("off_axis_degrees") or [10.0, 15.0]
    nearest = min(candidates, key=lambda c: angle_diff_rad(current_yaw, c))
    for deg in off_axis_degrees:
        d = as_float(deg, 0.0)
        if d <= 1e-9 or d > max_off_axis_deg + 1e-9:
            continue
        dr = math.radians(d)
        candidates.append(nearest - dr)
        candidates.append(nearest + dr)

    dedup_deg = as_float(prior_cfg.get("dedup_deg"), 5.0)
    dedup_rad = math.radians(dedup_deg)
    out: List[float] = []
    for cand in candidates:
        wrapped = ((cand + math.pi) % (2.0 * math.pi)) - math.pi
        if any(angle_diff_rad(wrapped, existing) < dedup_rad for existing in out):
            continue
        out.append(wrapped)
    return out


def _major_gain_ok(current_metrics: Dict[str, Any], candidate_metrics: Dict[str, Any], prior_cfg: Dict[str, Any]) -> bool:
    if int(candidate_metrics.get("validity", 0)) != 1:
        return False
    if int(current_metrics.get("Adopt_core", 0)) == 1 and int(candidate_metrics.get("Adopt_core", 0)) == 0:
        return False
    if int(current_metrics.get("Adopt_entry", 0)) == 1 and int(candidate_metrics.get("Adopt_entry", 0)) == 0:
        return False
    if int(current_metrics.get("Adopt_core", 0)) == 0 and int(candidate_metrics.get("Adopt_core", 0)) == 1:
        return True
    if int(current_metrics.get("Adopt_entry", 0)) == 0 and int(candidate_metrics.get("Adopt_entry", 0)) == 1:
        return True

    major = prior_cfg.get("major_gain_thresholds") if isinstance(prior_cfg.get("major_gain_thresholds"), dict) else {}
    if as_float(candidate_metrics.get("clr_feasible"), 0.0) - as_float(current_metrics.get("clr_feasible"), 0.0) >= as_float(major.get("clr_feasible"), 0.03):
        return True
    if as_float(candidate_metrics.get("C_vis_start"), 0.0) - as_float(current_metrics.get("C_vis_start"), 0.0) >= as_float(major.get("C_vis_start"), 0.05):
        return True
    if as_float(candidate_metrics.get("OOE_R_rec_entry_surf"), 0.0) - as_float(current_metrics.get("OOE_R_rec_entry_surf"), 0.0) >= as_float(major.get("OOE_R_rec_entry_surf"), 0.10):
        return True
    return False


def _rotation_penalty(
    layout: Dict[str, Any],
    obj: Dict[str, Any],
    yaw_rad: float,
    current_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    config: Dict[str, Any],
    protocol: str,
) -> float:
    prior_cfg = _alignment_prior_cfg(config, protocol)
    if not _use_alignment_prior(obj, prior_cfg):
        return 0.0
    axis = room_axis_for_object(layout, obj)
    deviation = orthogonal_deviation_rad(yaw_rad, axis)
    if deviation <= 1e-9:
        return 0.0

    max_off_axis_deg = as_float(prior_cfg.get("max_off_axis_deg"), 15.0)
    max_off_axis_rad = math.radians(max_off_axis_deg)
    if deviation > max_off_axis_rad + 1e-9:
        return float("inf")
    if not _major_gain_ok(current_metrics, candidate_metrics, prior_cfg):
        return float("inf")
    weight = as_float(prior_cfg.get("rotation_penalty_weight"), 0.5)
    return weight * (deviation / max_off_axis_rad) ** 2


def _score_tuple(
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    *,
    ooe_primary: str,
    delta_weight: float,
    use_lexicographic: bool,
) -> Tuple[int, int, int, float]:
    valid = 1 if int(metrics.get("validity", 0)) == 1 else 0
    adopt_entry = int(metrics.get("Adopt_entry", 0))
    adopt_core = int(metrics.get("Adopt_core", 0))
    cont = _continuous_score(metrics, config, ooe_primary=ooe_primary, delta_weight=delta_weight)
    if use_lexicographic:
        return (valid, adopt_entry, adopt_core, cont)
    # Fallback: use only continuous score but keep tuple shape.
    return (valid, 0, 0, cont)


def _state_key(layout: Dict[str, Any], *, protocol: str = "layout_only") -> Tuple[Tuple[str, int, int, int], ...]:
    rec: List[Tuple[str, int, int, int]] = []
    for obj in layout.get("objects", []):
        oid = str(obj.get("id") or "")
        cat = str(obj.get("category") or "").strip().lower()
        if protocol == "clutter_assisted":
            if not bool(obj.get("movable_in_clutter_recovery", cat == "clutter")) or cat != "clutter":
                continue
        elif not bool(obj.get("movable", True)):
            continue
        pose = obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]
        x = int(round(as_float(pose[0], 0.0) * 100.0))
        y = int(round(as_float(pose[1], 0.0) * 100.0))
        th = int(round(as_float(pose[3], 0.0) * 100.0))
        rec.append((oid, x, y, th))
    rec.sort(key=lambda t: t[0])
    return tuple(rec)


def _resolve_start_goal(debug: Dict[str, Any], config: Dict[str, Any]) -> Tuple[Vec2, Vec2]:
    task_points = debug.get("task_points")
    if isinstance(task_points, dict):
        s = (task_points.get("start") or {}).get("xy")
        g = (task_points.get("goal") or {}).get("xy")
        if isinstance(s, (list, tuple)) and len(s) >= 2 and isinstance(g, (list, tuple)) and len(g) >= 2:
            return (as_float(s[0], 0.0), as_float(s[1], 0.0)), (as_float(g[0], 0.0), as_float(g[1], 0.0))
    start_xy = config.get("start_xy") or [0.8, 0.8]
    goal_xy = config.get("goal_xy") or [5.0, 5.0]
    return (as_float(start_xy[0], 0.8), as_float(start_xy[1], 0.8)), (as_float(goal_xy[0], 5.0), as_float(goal_xy[1], 5.0))


def _resolve_bottleneck_xy(debug: Dict[str, Any], config: Dict[str, Any]) -> Optional[Vec2]:
    bottleneck = debug.get("bottleneck_cell")
    bounds = debug.get("bounds")
    resolution = as_float(config.get("grid_resolution_m"), 0.1)
    if bottleneck is None or not isinstance(bounds, (list, tuple)) or len(bounds) < 4:
        return None
    bx = as_float(bounds[0], 0.0) + (int(bottleneck[0]) + 0.5) * resolution
    by = as_float(bounds[1], 0.0) + (int(bottleneck[1]) + 0.5) * resolution
    return (bx, by)


def _object_priority(
    obj: Dict[str, Any],
    *,
    start_xy: Vec2,
    goal_xy: Vec2,
    bottleneck_xy: Optional[Vec2],
) -> float:
    x, y = _object_center(obj)
    d_path = _point_to_segment_distance(x, y, start_xy[0], start_xy[1], goal_xy[0], goal_xy[1])
    d_entry = _distance((x, y), start_xy)
    d_bneck = _distance((x, y), bottleneck_xy) if bottleneck_xy is not None else 1.0
    eps = 1e-6
    return (1.0 / (d_path + eps)) + 0.7 * (1.0 / (d_entry + eps)) + 0.8 * (1.0 / (d_bneck + eps))


def _select_candidate_objects(
    layout: Dict[str, Any],
    debug: Dict[str, Any],
    config: Dict[str, Any],
    *,
    changed_ids: Set[str],
    max_changed_objects: int,
    top_k: int,
) -> List[str]:
    protocol = _recovery_protocol(config)
    start_xy, goal_xy = _resolve_start_goal(debug, config)
    bottleneck_xy = _resolve_bottleneck_xy(debug, config)

    objs: List[Tuple[float, str]] = []
    for obj in layout.get("objects", []):
        oid = str(obj.get("id") or "")
        cat = str(obj.get("category") or "").strip().lower()
        if not oid:
            continue
        if protocol == "clutter_assisted":
            if not bool(obj.get("movable_in_clutter_recovery", cat == "clutter")) or cat != "clutter":
                continue
        else:
            if not bool(obj.get("movable", True)):
                continue
            if cat in {"floor", "door", "window", "opening", "clutter"}:
                continue
        if len(changed_ids) >= max_changed_objects and oid not in changed_ids:
            continue
        score = _object_priority(obj, start_xy=start_xy, goal_xy=goal_xy, bottleneck_xy=bottleneck_xy)
        objs.append((score, oid))
    objs.sort(key=lambda t: t[0], reverse=True)
    return [oid for _, oid in objs[: max(1, top_k)]]


def _top_k_unique(nodes: Sequence[BeamNode], k: int, *, protocol: str = "layout_only") -> List[BeamNode]:
    seen: Set[Tuple[Tuple[str, int, int, int], ...]] = set()
    out: List[BeamNode] = []
    for n in sorted(nodes, key=lambda x: x.score_tuple, reverse=True):
        key = _state_key(n.layout, protocol=protocol)
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
        if len(out) >= max(1, k):
            break
    return out


def run_refinement(
    layout: Dict[str, Any],
    baseline_layout: Dict[str, Any],
    config: Dict[str, Any],
    *,
    step_m: float,
    rot_deg: float,
    max_changed_objects: int,
    beam_width: int = 5,
    depth: int = 3,
    candidate_objects_per_state: int = 2,
    max_eval_calls: int = 0,
    ooe_primary: str = "OOE_R_rec_entry_surf",
    use_lexicographic: bool = True,
    allow_intermediate_regression: bool = True,
    door_keepout_radius_m: float = 0.0,
    overlap_ratio_max: float = 0.05,
    delta_weight: float = 0.3,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    protocol = _recovery_protocol(config)
    prior_cfg = _alignment_prior_cfg(config, protocol)
    if prior_cfg is not None:
        overlap_ratio_max = min(overlap_ratio_max, as_float(prior_cfg.get("overlap_ratio_max"), overlap_ratio_max))
        door_keepout_radius_m = max(door_keepout_radius_m, as_float(prior_cfg.get("door_keepout_radius_m"), door_keepout_radius_m))
        wall_margin_m = as_float(prior_cfg.get("wall_margin_m"), 0.0)
    else:
        wall_margin_m = 0.0
    if beam_width <= 0:
        beam_width = 5
    if depth <= 0:
        depth = 3
    if candidate_objects_per_state <= 0:
        candidate_objects_per_state = 2
    if max_changed_objects <= 0:
        max_changed_objects = 3

    actions: List[Tuple[float, float]] = []
    for dx in (-step_m, 0.0, step_m):
        for dy in (-step_m, 0.0, step_m):
            actions.append((dx, dy))

    if max_eval_calls <= 0:
        max_eval_calls = beam_width * candidate_objects_per_state * max(1, len(actions)) * depth

    initial_layout = copy.deepcopy(layout)
    initial_metrics, initial_debug = evaluate_layout(initial_layout, baseline_layout, config)
    initial_tuple = _score_tuple(
        initial_metrics,
        config,
        ooe_primary=ooe_primary,
        delta_weight=delta_weight,
        use_lexicographic=use_lexicographic,
    )
    initial_node = BeamNode(
        layout=initial_layout,
        metrics=initial_metrics,
        debug=initial_debug,
        changed_ids=set(),
        steps=[],
        score_tuple=initial_tuple,
    )

    best = initial_node
    beam: List[BeamNode] = [initial_node]
    visited: Set[Tuple[Tuple[str, int, int, int], ...]] = {_state_key(initial_layout, protocol=protocol)}
    eval_calls = 1
    logs: List[Dict[str, Any]] = []

    for d in range(depth):
        layer_candidates: List[BeamNode] = []
        door_centers = _collect_door_centers(best.layout)
        clearance_metric = _clearance_metric_name(config)
        for node in beam:
            object_ids = _select_candidate_objects(
                node.layout,
                node.debug,
                config,
                changed_ids=node.changed_ids,
                max_changed_objects=max_changed_objects,
                top_k=candidate_objects_per_state,
            )
            for oid in object_ids:
                obj0 = _find_object(node.layout, oid)
                if obj0 is None:
                    continue
                room_poly = room_polygon_for_object(node.layout, obj0) or ((node.layout.get("room") or {}).get("boundary_poly_xy") or [])
                yaw_candidates = _candidate_yaws(node.layout, obj0, config, protocol, rot_deg)
                current_yaw = as_float((obj0.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0])[3], 0.0)
                for dx, dy in actions:
                    for yaw_candidate in yaw_candidates:
                        if abs(dx) < 1e-12 and abs(dy) < 1e-12 and angle_diff_rad(yaw_candidate, current_yaw) < 1e-12:
                            continue
                        if eval_calls >= max_eval_calls:
                            break
                        cand_layout = copy.deepcopy(node.layout)
                        obj = _find_object(cand_layout, oid)
                        if obj is None:
                            continue
                        pose = list(obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0])
                        while len(pose) < 4:
                            pose.append(0.0)
                        pose[0] = as_float(pose[0], 0.0) + dx
                        pose[1] = as_float(pose[1], 0.0) + dy
                        pose[3] = yaw_candidate
                        obj["pose_xyz_yaw"] = pose

                        if room_poly and not _object_inside_room(obj, room_poly):
                            continue
                        if wall_margin_m > 0.0 and _room_edge_clearance(obj, room_poly) + 1e-9 < wall_margin_m:
                            continue
                        if door_keepout_radius_m > 0.0:
                            c = _object_center(obj)
                            if any(_distance(c, dc) < door_keepout_radius_m - 1e-9 for dc in door_centers):
                                continue
                        if overlap_ratio_max > 0.0 and _has_excess_overlap(cand_layout, oid, overlap_ratio_max):
                            continue

                        cand_metrics, cand_debug = evaluate_layout(cand_layout, baseline_layout, config)
                        eval_calls += 1
                        if int(cand_metrics.get("validity", 0)) != 1:
                            continue
                        if not allow_intermediate_regression:
                            if as_float(cand_metrics.get("R_reach"), 0.0) + 1e-9 < as_float(node.metrics.get("R_reach"), 0.0):
                                continue
                            if _clearance_value(cand_metrics, clearance_metric) + 1e-9 < _clearance_value(node.metrics, clearance_metric):
                                continue

                        rotation_penalty = _rotation_penalty(cand_layout, obj, yaw_candidate, node.metrics, cand_metrics, config, protocol)
                        if not math.isfinite(rotation_penalty):
                            continue

                        changed_ids = set(node.changed_ids)
                        changed_ids.add(oid)
                        if len(changed_ids) > max_changed_objects:
                            continue

                        key = _state_key(cand_layout, protocol=protocol)
                        if key in visited:
                            continue
                        visited.add(key)

                        st = _score_tuple(
                            cand_metrics,
                            config,
                            ooe_primary=ooe_primary,
                            delta_weight=delta_weight,
                            use_lexicographic=use_lexicographic,
                        )
                        st = (st[0], st[1], st[2], st[3] - rotation_penalty)
                        step = {
                            "depth": d + 1,
                            "object_id": oid,
                            "dx": dx,
                            "dy": dy,
                            "yaw_deg": math.degrees(yaw_candidate),
                            "rotation_penalty": rotation_penalty,
                            "score_tuple": list(st),
                        }
                        new_node = BeamNode(
                            layout=cand_layout,
                            metrics=cand_metrics,
                            debug=cand_debug,
                            changed_ids=changed_ids,
                            steps=[*node.steps, step],
                            score_tuple=st,
                        )
                        layer_candidates.append(new_node)
                        if new_node.score_tuple > best.score_tuple:
                            best = new_node
                    if eval_calls >= max_eval_calls:
                        break
            if eval_calls >= max_eval_calls:
                break

        if not layer_candidates:
            break
        beam = _top_k_unique(layer_candidates, beam_width, protocol=protocol)
        logs.append(
            {
                "depth": d + 1,
                "beam_size": len(beam),
                "candidates_generated": len(layer_candidates),
                "best_score_tuple": list(best.score_tuple),
                "eval_calls": eval_calls,
            }
        )
        if eval_calls >= max_eval_calls:
            break

    return best.layout, best.metrics, [{"search": logs, "best_steps": best.steps, "eval_calls": eval_calls, "budget": max_eval_calls}]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refine layout with proposed beam-search method")
    parser.add_argument("--layout_in", required=True)
    parser.add_argument("--layout_out", required=True)
    parser.add_argument("--metrics_out", required=True)
    parser.add_argument("--log_out", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--step_m", type=float, default=0.10)
    parser.add_argument("--rot_deg", type=float, default=15.0)
    parser.add_argument("--max_changed_objects", type=int, default=3)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--candidate_objects_per_state", type=int, default=2)
    parser.add_argument("--max_eval_calls", type=int, default=0)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = default_eval_config()
    if args.config:
        config = merge_eval_config(config, json.loads(pathlib.Path(args.config).read_text(encoding="utf-8-sig")))

    layout = load_layout_contract(pathlib.Path(args.layout_in))
    baseline = copy.deepcopy(layout)

    refined, metrics, logs = run_refinement(
        layout=layout,
        baseline_layout=baseline,
        config=config,
        step_m=args.step_m,
        rot_deg=args.rot_deg,
        max_changed_objects=args.max_changed_objects,
        beam_width=args.beam_width,
        depth=args.depth,
        candidate_objects_per_state=args.candidate_objects_per_state,
        max_eval_calls=args.max_eval_calls,
    )
    write_json(pathlib.Path(args.layout_out), refined)
    write_json(pathlib.Path(args.metrics_out), metrics)
    write_json(pathlib.Path(args.log_out), {"steps": logs})
    print(json.dumps({"metrics": metrics, "log_steps": len(logs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
