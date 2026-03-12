from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from eval_metrics import evaluate_layout
from layout_tools import as_float, obb_corners_xy, point_in_polygon


Vec2 = Tuple[float, float]


@dataclass
class SearchNode:
    layout: Dict[str, Any]
    metrics: Dict[str, Any]
    debug: Dict[str, Any]
    changed_ids: Set[str]
    steps: List[Dict[str, Any]]
    score_tuple: Tuple[int, int, int, float]


def _find_object(layout: Dict[str, Any], object_id: str) -> Optional[Dict[str, Any]]:
    for obj in layout.get("objects", []):
        if str(obj.get("id") or "") == object_id:
            return obj
    return None


def _object_center(obj: Dict[str, Any]) -> Vec2:
    pose = obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]
    return as_float(pose[0], 0.0), as_float(pose[1], 0.0)


def _object_inside_room(obj: Dict[str, Any], room_poly: Sequence[Sequence[float]]) -> bool:
    corners = obb_corners_xy(obj)
    for x, y in corners:
        if not point_in_polygon(x, y, room_poly):
            return False
    return True


def _room_bounds(room_poly: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    xs = [as_float(p[0], 0.0) for p in room_poly]
    ys = [as_float(p[1], 0.0) for p in room_poly]
    return min(xs), min(ys), max(xs), max(ys)


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
    return min(xs), min(ys), max(xs), max(ys)


def _aabb_area(a: Tuple[float, float, float, float]) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def _aabb_intersection(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _has_excess_overlap(layout: Dict[str, Any], moved_object_id: str, max_ratio: float) -> bool:
    obj = _find_object(layout, moved_object_id)
    if obj is None:
        return True
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


def _resolve_start_goal(debug: Dict[str, Any], config: Dict[str, Any]) -> Tuple[Vec2, Vec2]:
    task_points = debug.get("task_points")
    if isinstance(task_points, dict):
        start = (task_points.get("start") or {}).get("xy")
        goal = (task_points.get("goal") or {}).get("xy")
        if isinstance(start, (list, tuple)) and len(start) >= 2 and isinstance(goal, (list, tuple)) and len(goal) >= 2:
            return (as_float(start[0], 0.0), as_float(start[1], 0.0)), (as_float(goal[0], 0.0), as_float(goal[1], 0.0))
    start_xy = config.get("start_xy") or [0.8, 0.8]
    goal_xy = config.get("goal_xy") or [5.0, 5.0]
    return (as_float(start_xy[0], 0.8), as_float(start_xy[1], 0.8)), (as_float(goal_xy[0], 5.0), as_float(goal_xy[1], 5.0))


def _resolve_bottleneck(debug: Dict[str, Any], config: Dict[str, Any]) -> Optional[Vec2]:
    cell = debug.get("bottleneck_cell")
    bounds = debug.get("bounds")
    resolution = as_float(config.get("grid_resolution_m"), 0.1)
    if cell is None or not isinstance(bounds, (list, tuple)) or len(bounds) < 4:
        return None
    bx = as_float(bounds[0], 0.0) + (int(cell[0]) + 0.5) * resolution
    by = as_float(bounds[1], 0.0) + (int(cell[1]) + 0.5) * resolution
    return (bx, by)


def _collect_door_centers(layout: Dict[str, Any]) -> List[Vec2]:
    out: List[Vec2] = []
    for obj in layout.get("objects", []):
        if str(obj.get("category") or "").strip().lower() == "door":
            out.append(_object_center(obj))
    return out


def _candidate_clutter_objects(layout: Dict[str, Any], changed_ids: Set[str], max_changed_objects: int) -> List[str]:
    out: List[str] = []
    for obj in layout.get("objects", []):
        oid = str(obj.get("id") or "")
        cat = str(obj.get("category") or "").strip().lower()
        if not oid or cat != "clutter":
            continue
        if not bool(obj.get("movable_in_clutter_recovery", True)):
            continue
        if len(changed_ids) >= max_changed_objects and oid not in changed_ids:
            continue
        out.append(oid)
    return out


def _boundary_score(x: float, y: float, bounds: Tuple[float, float, float, float]) -> float:
    min_x, min_y, max_x, max_y = bounds
    d = min(x - min_x, max_x - x, y - min_y, max_y - y)
    return 1.0 / max(0.05, d)


def _quick_position_score(
    *,
    x: float,
    y: float,
    start_xy: Vec2,
    goal_xy: Vec2,
    bottleneck_xy: Optional[Vec2],
    room_bounds: Tuple[float, float, float, float],
) -> float:
    dist_path = _point_to_segment_distance(x, y, start_xy[0], start_xy[1], goal_xy[0], goal_xy[1])
    dist_start = _distance((x, y), start_xy)
    dist_goal = _distance((x, y), goal_xy)
    dist_bneck = _distance((x, y), bottleneck_xy) if bottleneck_xy is not None else 0.0
    return (
        2.5 * dist_path
        + 1.2 * dist_start
        + 0.4 * dist_goal
        + 1.0 * dist_bneck
        + 1.0 * _boundary_score(x, y, room_bounds)
    )


def _generate_candidate_positions(
    layout: Dict[str, Any],
    object_id: str,
    *,
    debug: Dict[str, Any],
    grid_step_m: float,
    candidate_limit: int,
    door_keepout_radius_m: float,
    overlap_ratio_max: float,
    config: Dict[str, Any],
) -> List[Tuple[float, float]]:
    room_poly = (layout.get("room") or {}).get("boundary_poly_xy") or []
    if not room_poly:
        return []
    room_bounds = _room_bounds(room_poly)
    obj = _find_object(layout, object_id)
    if obj is None:
        return []
    start_xy, goal_xy = _resolve_start_goal(debug, config)
    bottleneck_xy = _resolve_bottleneck(debug, config)
    door_centers = _collect_door_centers(layout)
    current_pose = list(obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0])
    while len(current_pose) < 4:
        current_pose.append(0.0)
    current_x, current_y = as_float(current_pose[0], 0.0), as_float(current_pose[1], 0.0)

    candidates: List[Tuple[float, Tuple[float, float]]] = []
    seen: Set[Tuple[int, int]] = set()
    min_x, min_y, max_x, max_y = room_bounds
    y = min_y
    while y <= max_y + 1e-9:
        x = min_x
        while x <= max_x + 1e-9:
            key = (int(round(x * 100.0)), int(round(y * 100.0)))
            if key in seen:
                x += grid_step_m
                continue
            seen.add(key)
            cand_layout = copy.deepcopy(layout)
            moved = _find_object(cand_layout, object_id)
            if moved is None:
                x += grid_step_m
                continue
            pose = list(moved.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0])
            while len(pose) < 4:
                pose.append(0.0)
            pose[0] = x
            pose[1] = y
            moved["pose_xyz_yaw"] = pose
            if not _object_inside_room(moved, room_poly):
                x += grid_step_m
                continue
            center = (x, y)
            if any(_distance(center, dc) < door_keepout_radius_m - 1e-9 for dc in door_centers):
                x += grid_step_m
                continue
            if overlap_ratio_max > 0.0 and _has_excess_overlap(cand_layout, object_id, overlap_ratio_max):
                x += grid_step_m
                continue
            score = _quick_position_score(
                x=x,
                y=y,
                start_xy=start_xy,
                goal_xy=goal_xy,
                bottleneck_xy=bottleneck_xy,
                room_bounds=room_bounds,
            )
            candidates.append((score, center))
            x += grid_step_m
        y += grid_step_m

    current_key = (int(round(current_x * 100.0)), int(round(current_y * 100.0)))
    if current_key not in seen:
        candidates.append(
            (
                _quick_position_score(
                    x=current_x,
                    y=current_y,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                    bottleneck_xy=bottleneck_xy,
                    room_bounds=room_bounds,
                ),
                (current_x, current_y),
            )
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [center for _, center in candidates[: max(1, candidate_limit)]]


def _score_tuple(metrics: Dict[str, Any], *, delta_weight: float, ooe_primary: str) -> Tuple[int, int, int, float]:
    valid = 1 if int(metrics.get("validity", 0)) == 1 else 0
    adopt_entry = int(metrics.get("Adopt_entry", 0))
    adopt_core = int(metrics.get("Adopt_core", 0))
    cont = (
        3.0 * as_float(metrics.get("clr_feasible"), 0.0)
        + 2.0 * as_float(metrics.get("C_vis_start"), 0.0)
        + 2.0 * as_float(metrics.get(ooe_primary), 0.0)
        + 1.0 * as_float(metrics.get("R_reach"), 0.0)
        + 0.5 * as_float(metrics.get("C_vis"), 0.0)
        - delta_weight * as_float(metrics.get("Delta_layout"), 0.0)
    )
    return (valid, adopt_entry, adopt_core, cont)


def _state_key(layout: Dict[str, Any]) -> Tuple[Tuple[str, int, int], ...]:
    rec: List[Tuple[str, int, int]] = []
    for obj in layout.get("objects", []):
        oid = str(obj.get("id") or "")
        cat = str(obj.get("category") or "").strip().lower()
        if not oid or cat != "clutter":
            continue
        if not bool(obj.get("movable_in_clutter_recovery", True)):
            continue
        x, y = _object_center(obj)
        rec.append((oid, int(round(x * 100.0)), int(round(y * 100.0))))
    rec.sort(key=lambda item: item[0])
    return tuple(rec)


def _run_heuristic(
    layout: Dict[str, Any],
    baseline_layout: Dict[str, Any],
    config: Dict[str, Any],
    *,
    max_iterations: int,
    max_changed_objects: int,
    grid_step_m: float,
    candidate_limit: int,
    door_keepout_radius_m: float,
    overlap_ratio_max: float,
    delta_weight: float,
    ooe_primary: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    current = copy.deepcopy(layout)
    current_metrics, current_debug = evaluate_layout(current, baseline_layout, config)
    current_score = _score_tuple(current_metrics, delta_weight=delta_weight, ooe_primary=ooe_primary)
    changed_ids: Set[str] = set()
    logs: List[Dict[str, Any]] = []

    for iteration in range(1, max_iterations + 1):
        object_ids = _candidate_clutter_objects(current, changed_ids, max_changed_objects)
        if not object_ids:
            break
        best_layout = None
        best_metrics = None
        best_debug = None
        best_score = current_score
        best_move = None

        for oid in object_ids:
            positions = _generate_candidate_positions(
                current,
                oid,
                debug=current_debug,
                grid_step_m=grid_step_m,
                candidate_limit=candidate_limit,
                door_keepout_radius_m=door_keepout_radius_m,
                overlap_ratio_max=overlap_ratio_max,
                config=config,
            )
            for px, py in positions:
                cand = copy.deepcopy(current)
                obj = _find_object(cand, oid)
                if obj is None:
                    continue
                pose = list(obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0])
                while len(pose) < 4:
                    pose.append(0.0)
                before_x, before_y = as_float(pose[0], 0.0), as_float(pose[1], 0.0)
                if abs(before_x - px) < 1e-9 and abs(before_y - py) < 1e-9:
                    continue
                pose[0] = px
                pose[1] = py
                obj["pose_xyz_yaw"] = pose
                metrics, debug = evaluate_layout(cand, baseline_layout, config)
                if int(metrics.get("validity", 0)) != 1:
                    continue
                score = _score_tuple(metrics, delta_weight=delta_weight, ooe_primary=ooe_primary)
                if score > best_score:
                    best_layout = cand
                    best_metrics = metrics
                    best_debug = debug
                    best_score = score
                    best_move = {
                        "object_id": oid,
                        "from_xy": [before_x, before_y],
                        "to_xy": [px, py],
                        "disp_m": math.hypot(px - before_x, py - before_y),
                    }

        if best_layout is None or best_metrics is None or best_debug is None:
            break

        logs.append(
            {
                "iteration": iteration,
                "move": best_move,
                "before": current_metrics,
                "after": best_metrics,
                "score_before": list(current_score),
                "score_after": list(best_score),
            }
        )
        current = best_layout
        current_metrics = best_metrics
        current_debug = best_debug
        current_score = best_score
        if best_move is not None:
            changed_ids.add(str(best_move["object_id"]))

    return current, current_metrics, logs


def _run_proposed(
    layout: Dict[str, Any],
    baseline_layout: Dict[str, Any],
    config: Dict[str, Any],
    *,
    beam_width: int,
    depth: int,
    max_changed_objects: int,
    grid_step_m: float,
    candidate_limit: int,
    door_keepout_radius_m: float,
    overlap_ratio_max: float,
    delta_weight: float,
    ooe_primary: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    initial_layout = copy.deepcopy(layout)
    initial_metrics, initial_debug = evaluate_layout(initial_layout, baseline_layout, config)
    initial_score = _score_tuple(initial_metrics, delta_weight=delta_weight, ooe_primary=ooe_primary)
    initial_node = SearchNode(
        layout=initial_layout,
        metrics=initial_metrics,
        debug=initial_debug,
        changed_ids=set(),
        steps=[],
        score_tuple=initial_score,
    )
    best = initial_node
    beam: List[SearchNode] = [initial_node]
    visited: Set[Tuple[Tuple[str, int, int], ...]] = {_state_key(initial_layout)}

    for _ in range(depth):
        layer_nodes: List[SearchNode] = []
        for node in beam:
            object_ids = _candidate_clutter_objects(node.layout, node.changed_ids, max_changed_objects)
            for oid in object_ids:
                positions = _generate_candidate_positions(
                    node.layout,
                    oid,
                    debug=node.debug,
                    grid_step_m=grid_step_m,
                    candidate_limit=candidate_limit,
                    door_keepout_radius_m=door_keepout_radius_m,
                    overlap_ratio_max=overlap_ratio_max,
                    config=config,
                )
                for px, py in positions:
                    cand = copy.deepcopy(node.layout)
                    obj = _find_object(cand, oid)
                    if obj is None:
                        continue
                    pose = list(obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0])
                    while len(pose) < 4:
                        pose.append(0.0)
                    before_x, before_y = as_float(pose[0], 0.0), as_float(pose[1], 0.0)
                    if abs(before_x - px) < 1e-9 and abs(before_y - py) < 1e-9:
                        continue
                    pose[0] = px
                    pose[1] = py
                    obj["pose_xyz_yaw"] = pose
                    state_key = _state_key(cand)
                    if state_key in visited:
                        continue
                    metrics, debug = evaluate_layout(cand, baseline_layout, config)
                    if int(metrics.get("validity", 0)) != 1:
                        continue
                    visited.add(state_key)
                    score = _score_tuple(metrics, delta_weight=delta_weight, ooe_primary=ooe_primary)
                    steps = list(node.steps)
                    steps.append(
                        {
                            "object_id": oid,
                            "from_xy": [before_x, before_y],
                            "to_xy": [px, py],
                            "disp_m": math.hypot(px - before_x, py - before_y),
                        }
                    )
                    changed = set(node.changed_ids)
                    changed.add(oid)
                    child = SearchNode(
                        layout=cand,
                        metrics=metrics,
                        debug=debug,
                        changed_ids=changed,
                        steps=steps,
                        score_tuple=score,
                    )
                    layer_nodes.append(child)
                    if child.score_tuple > best.score_tuple:
                        best = child
        if not layer_nodes:
            break
        layer_nodes.sort(key=lambda node: node.score_tuple, reverse=True)
        beam = layer_nodes[: max(1, beam_width)]

    return best.layout, best.metrics, best.steps


def run_refinement(
    layout: Dict[str, Any],
    baseline_layout: Dict[str, Any],
    config: Dict[str, Any],
    *,
    method: str,
    max_iterations: int,
    max_changed_objects: int,
    grid_step_m: float,
    candidate_limit: int,
    door_keepout_radius_m: float,
    overlap_ratio_max: float,
    delta_weight: float,
    beam_width: int,
    depth: int,
    ooe_primary: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    clutter_ids = _candidate_clutter_objects(copy.deepcopy(layout), set(), max_changed_objects=max_changed_objects)
    if not clutter_ids:
        metrics, _ = evaluate_layout(layout, baseline_layout, config)
        return copy.deepcopy(layout), metrics, []
    if str(method).strip().lower() == "proposed":
        return _run_proposed(
            layout,
            baseline_layout,
            config,
            beam_width=beam_width,
            depth=depth,
            max_changed_objects=max_changed_objects,
            grid_step_m=grid_step_m,
            candidate_limit=candidate_limit,
            door_keepout_radius_m=door_keepout_radius_m,
            overlap_ratio_max=overlap_ratio_max,
            delta_weight=delta_weight,
            ooe_primary=ooe_primary,
        )
    return _run_heuristic(
        layout,
        baseline_layout,
        config,
        max_iterations=max_iterations,
        max_changed_objects=max_changed_objects,
        grid_step_m=grid_step_m,
        candidate_limit=candidate_limit,
        door_keepout_radius_m=door_keepout_radius_m,
        overlap_ratio_max=overlap_ratio_max,
        delta_weight=delta_weight,
        ooe_primary=ooe_primary,
    )
