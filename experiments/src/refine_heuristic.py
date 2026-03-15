from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
from typing import Any, Dict, List, Optional, Set, Tuple

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


def _clearance_value(metrics: Dict[str, Any]) -> float:
    if "clr_feasible" in metrics:
        return as_float(metrics.get("clr_feasible"), 0.0)
    if "clr_min_astar" in metrics:
        return as_float(metrics.get("clr_min_astar"), 0.0)
    return as_float(metrics.get("clr_min"), 0.0)


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


def _object_center(obj: Dict[str, Any]) -> Tuple[float, float]:
    pose = obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]
    return as_float(pose[0], 0.0), as_float(pose[1], 0.0)


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _collect_door_centers(layout: Dict[str, Any]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for obj in layout.get("objects", []):
        if str(obj.get("category") or "").strip().lower() == "door":
            out.append(_object_center(obj))
    return out


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


def _score(
    metrics: Dict[str, Any],
    alpha: float = 1.0,
    beta: float = 1.0,
    eta: float = 1.0,
    gamma: float = 0.5,
    extra_penalty: float = 0.0,
) -> float:
    cmax = 1.0
    penalty = 0.0
    if metrics.get("validity", 0) == 0:
        penalty += 5.0
    if metrics.get("R_reach", 0.0) <= 0.0:
        penalty += 2.0

    return (
        alpha * as_float(metrics.get("C_vis"), 0.0)
        + beta * as_float(metrics.get("R_reach"), 0.0)
        + eta * max(0.0, min(cmax, _clearance_value(metrics)))
        - gamma * as_float(metrics.get("Delta_layout"), 0.0)
        - extra_penalty
        - penalty
    )


def _find_object(layout: Dict[str, Any], object_id: str) -> Optional[Dict[str, Any]]:
    for obj in layout.get("objects", []):
        if obj.get("id") == object_id:
            return obj
    return None


def _recovery_protocol(config: Dict[str, Any]) -> str:
    return str((config or {}).get("recovery_protocol", "layout_only") or "layout_only").strip().lower()


def _candidate_objects(layout: Dict[str, Any], protocol: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for obj in layout.get("objects", []):
        oid = str(obj.get("id") or "")
        if not oid:
            continue
        cat = str(obj.get("category") or "").strip().lower()
        if protocol == "clutter_assisted":
            if bool(obj.get("movable_in_clutter_recovery", cat == "clutter")) and cat == "clutter":
                out.append(obj)
            continue
        if bool(obj.get("movable", True)):
            out.append(obj)
    return out


def _object_inside_room(obj: Dict[str, Any], room_poly: List[List[float]]) -> bool:
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


def _room_edge_clearance(obj: Dict[str, Any], room_poly: List[List[float]]) -> float:
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
    length = as_float(size[0], 1.0)
    width = as_float(size[1], 1.0)
    return abs(length - width) > 0.1


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


def _select_target_object(
    layout: Dict[str, Any],
    debug: Dict[str, Any],
    config: Dict[str, Any],
    changed_ids: Set[str],
    max_changed_objects: int,
) -> Optional[str]:
    protocol = _recovery_protocol(config)
    movable = _candidate_objects(layout, protocol)
    if not movable:
        return None

    if len(changed_ids) >= max_changed_objects:
        movable = [obj for obj in movable if obj.get("id") in changed_ids]
        if not movable:
            return None
    else:
        movable = [obj for obj in movable if obj.get("id") not in changed_ids]

    resolution = as_float(config.get("grid_resolution_m"), 0.1)
    bounds = debug["bounds"]
    bottleneck = debug.get("bottleneck_cell")

    if bottleneck is not None:
        bx = bounds[0] + (bottleneck[0] + 0.5) * resolution
        by = bounds[1] + (bottleneck[1] + 0.5) * resolution

        best = None
        best_dist = float("inf")
        for obj in movable:
            pose = obj.get("pose_xyz_yaw", [0.0, 0.0, 0.0, 0.0])
            dist = math.hypot(as_float(pose[0], 0.0) - bx, as_float(pose[1], 0.0) - by)
            if dist < best_dist:
                best_dist = dist
                best = obj.get("id")
        return best

    start_xy = None
    goal_xy = None
    task_points = debug.get("task_points")
    if isinstance(task_points, dict):
        start = (task_points.get("start") or {}).get("xy")
        goal = (task_points.get("goal") or {}).get("xy")
        if isinstance(start, (list, tuple)) and len(start) >= 2 and isinstance(goal, (list, tuple)) and len(goal) >= 2:
            start_xy = start
            goal_xy = goal

    if start_xy is None:
        start_xy = config.get("start_xy") or [0.8, 0.8]
    if goal_xy is None:
        goal_xy = config.get("goal_xy") or [5.0, 5.0]
    ax, ay = as_float(start_xy[0], 0.8), as_float(start_xy[1], 0.8)
    bx, by = as_float(goal_xy[0], 5.0), as_float(goal_xy[1], 5.0)

    best = None
    best_dist = float("inf")
    for obj in movable:
        pose = obj.get("pose_xyz_yaw", [0.0, 0.0, 0.0, 0.0])
        px, py = as_float(pose[0], 0.0), as_float(pose[1], 0.0)
        dist = _point_to_segment_distance(px, py, ax, ay, bx, by)
        if dist < best_dist:
            best_dist = dist
            best = obj.get("id")
    return best


def run_refinement(
    layout: Dict[str, Any],
    baseline_layout: Dict[str, Any],
    config: Dict[str, Any],
    max_iterations: int,
    step_m: float,
    rot_deg: float,
    max_changed_objects: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    current = copy.deepcopy(layout)
    current_metrics, current_debug = evaluate_layout(current, baseline_layout, config)
    current_score = _score(current_metrics)
    protocol = _recovery_protocol(config)
    prior_cfg = _alignment_prior_cfg(config, protocol)
    overlap_ratio_max = 0.0 if prior_cfg is None else as_float(prior_cfg.get("overlap_ratio_max"), 0.0)
    wall_margin_m = 0.0 if prior_cfg is None else as_float(prior_cfg.get("wall_margin_m"), 0.0)
    door_keepout_radius_m = 0.0 if prior_cfg is None else as_float(prior_cfg.get("door_keepout_radius_m"), 0.0)

    door_centers = _collect_door_centers(current)
    changed_ids: Set[str] = set()
    logs: List[Dict[str, Any]] = []

    for iteration in range(1, max_iterations + 1):
        target_id = _select_target_object(current, current_debug, config, changed_ids, max_changed_objects)
        if target_id is None:
            break

        base_obj = _find_object(current, target_id)
        if base_obj is None:
            break
        room_poly = room_polygon_for_object(current, base_obj) or current["room"]["boundary_poly_xy"]

        yaw_candidates = _candidate_yaws(current, base_obj, config, protocol, rot_deg)

        best_layout = None
        best_metrics = None
        best_debug = None
        best_score = current_score
        best_move = None

        for dx in (-step_m, 0.0, step_m):
            for dy in (-step_m, 0.0, step_m):
                for yaw_candidate in yaw_candidates:
                    current_yaw = as_float((base_obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0])[3], 0.0)
                    if abs(dx) < 1e-12 and abs(dy) < 1e-12 and angle_diff_rad(yaw_candidate, current_yaw) < 1e-12:
                        continue

                    candidate = copy.deepcopy(current)
                    obj = _find_object(candidate, target_id)
                    if obj is None:
                        continue

                    pose = list(obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0])
                    pose[0] = as_float(pose[0], 0.0) + dx
                    pose[1] = as_float(pose[1], 0.0) + dy
                    pose[3] = yaw_candidate
                    obj["pose_xyz_yaw"] = pose

                    if not _object_inside_room(obj, room_poly):
                        continue
                    if wall_margin_m > 0.0 and _room_edge_clearance(obj, room_poly) + 1e-9 < wall_margin_m:
                        continue
                    if door_keepout_radius_m > 0.0:
                        c = _object_center(obj)
                        if any(_distance(c, dc) < door_keepout_radius_m - 1e-9 for dc in door_centers):
                            continue
                    if overlap_ratio_max > 0.0 and _has_excess_overlap(candidate, target_id, overlap_ratio_max):
                        continue

                    metrics, debug = evaluate_layout(candidate, baseline_layout, config)
                    if metrics.get("validity", 0) == 0:
                        continue

                    # Non-regression guard on reachability/clearance.
                    if as_float(metrics.get("R_reach"), 0.0) + 1e-9 < as_float(current_metrics.get("R_reach"), 0.0):
                        continue
                    if _clearance_value(metrics) + 1e-9 < _clearance_value(current_metrics):
                        continue

                    rotation_penalty = _rotation_penalty(candidate, obj, yaw_candidate, current_metrics, metrics, config, protocol)
                    if not math.isfinite(rotation_penalty):
                        continue

                    score = _score(metrics, extra_penalty=rotation_penalty)
                    if score > best_score + 1e-9:
                        best_score = score
                        best_layout = candidate
                        best_metrics = metrics
                        best_debug = debug
                        best_move = {
                            "object_id": target_id,
                            "dx": dx,
                            "dy": dy,
                            "yaw_deg": math.degrees(yaw_candidate),
                            "rotation_penalty": rotation_penalty,
                        }

        if best_layout is None:
            break

        logs.append(
            {
                "iteration": iteration,
                "target_id": target_id,
                "move": best_move,
                "before": current_metrics,
                "after": best_metrics,
                "score_before": current_score,
                "score_after": best_score,
            }
        )

        changed_ids.add(target_id)
        current = best_layout
        current_metrics = best_metrics
        current_debug = best_debug
        current_score = best_score

    return current, current_metrics, logs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refine layout with heuristic local search")
    parser.add_argument("--layout_in", required=True)
    parser.add_argument("--layout_out", required=True)
    parser.add_argument("--metrics_out", required=True)
    parser.add_argument("--log_out", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--max_iterations", type=int, default=30)
    parser.add_argument("--step_m", type=float, default=0.10)
    parser.add_argument("--rot_deg", type=float, default=15.0)
    parser.add_argument("--max_changed_objects", type=int, default=3)
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
        max_iterations=args.max_iterations,
        step_m=args.step_m,
        rot_deg=args.rot_deg,
        max_changed_objects=args.max_changed_objects,
    )

    write_json(pathlib.Path(args.layout_out), refined)
    write_json(pathlib.Path(args.metrics_out), metrics)
    write_json(pathlib.Path(args.log_out), {"steps": logs})

    print(json.dumps({"metrics": metrics, "log_steps": len(logs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

