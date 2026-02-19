from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


ARCH_CATEGORIES = {
    "floor",
    "wall",
    "door",
    "sliding_door",
    "window",
    "opening",
    "room",
    "room_inner_frame",
    "subroom_inner_frame",
}

DEFAULT_OBJECT_HEIGHT = {
    "bed": 0.55,
    "chair": 0.9,
    "sink": 0.9,
    "toilet": 0.8,
    "storage": 2.0,
    "cabinet": 1.2,
    "tv_cabinet": 0.6,
    "tv": 0.6,
    "sofa": 0.8,
    "table": 0.75,
    "floor": 0.05,
    "door": 2.0,
}

DEFAULT_SEARCH_PROMPT = {
    "bed": "single bed with simple frame",
    "chair": "compact chair, simple modern",
    "sink": "compact sink with small vanity",
    "toilet": "standard ceramic toilet",
    "storage": "compact storage cabinet / wardrobe",
    "cabinet": "small storage cabinet",
    "tv_cabinet": "slim TV cabinet / media console",
    "tv": "tv cabinet with flat-screen TV",
    "sofa": "compact sofa, modern",
    "table": "small table, simple modern",
    "floor": "flat floor plane, neutral material",
    "door": "interior sliding door, simple modern",
}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _r3(v: float) -> float:
    return round(float(v), 3)


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def _canonical_category(raw: Any) -> str:
    text = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text == "slidingdoor":
        return "sliding_door"
    if text == "tvcabinet":
        return "tv_cabinet"
    return text or "unknown"


def _normalize_angle(angle: Any) -> Optional[int]:
    try:
        a = float(angle) % 360.0
    except (TypeError, ValueError):
        return None
    candidates = [0, 90, 180, 270]
    best = min(candidates, key=lambda c: min(abs(a - c), 360.0 - abs(a - c)))
    return int(best)


def _polygon_area(poly: List[Dict[str, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = _to_float(poly[i].get("X")), _to_float(poly[i].get("Y"))
        x2, y2 = _to_float(poly[(i + 1) % n].get("X")), _to_float(poly[(i + 1) % n].get("Y"))
        area += (x1 * y2) - (x2 * y1)
    return abs(area) * 0.5


def _clean_polygon(poly: List[Dict[str, float]]) -> List[Dict[str, float]]:
    if not poly:
        return poly
    cleaned: List[Dict[str, float]] = []
    for p in poly:
        x = _r3(_to_float(p.get("X")))
        y = _r3(_to_float(p.get("Y")))
        if not cleaned or (cleaned[-1]["X"] != x or cleaned[-1]["Y"] != y):
            cleaned.append({"X": x, "Y": y})
    if len(cleaned) >= 2 and cleaned[0]["X"] == cleaned[-1]["X"] and cleaned[0]["Y"] == cleaned[-1]["Y"]:
        cleaned.pop()
    return cleaned


def _point_in_polygon(x: float, y: float, polygon: List[Dict[str, float]]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = _to_float(polygon[i].get("X")), _to_float(polygon[i].get("Y"))
        xj, yj = _to_float(polygon[j].get("X")), _to_float(polygon[j].get("Y"))
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) if abs(yj - yi) > 1e-9 else 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _extract_base_area(step1_json: Dict[str, Any]) -> Tuple[float, float]:
    area_x = _to_float(step1_json.get("area_size_X"), 0.0)
    area_y = _to_float(step1_json.get("area_size_Y"), 0.0)
    if area_x > 0 and area_y > 0:
        return area_x, area_y

    outer = step1_json.get("outer_polygon")
    if isinstance(outer, list) and outer:
        xs = [_to_float(p.get("X")) for p in outer if isinstance(p, dict)]
        ys = [_to_float(p.get("Y")) for p in outer if isinstance(p, dict)]
        if xs and ys:
            return max(xs) - min(xs), max(ys) - min(ys)
    return 1.0, 1.0


def _extract_inner_transform(
    main_room_inner_boundary_hint: Optional[Dict[str, Any]],
    area_x: float,
    area_y: float,
) -> Dict[str, float]:
    if not isinstance(main_room_inner_boundary_hint, dict):
        return {"offset_x": 0.0, "offset_y": 0.0, "size_x": area_x, "size_y": area_y}

    box = main_room_inner_boundary_hint.get("box_world")
    if not isinstance(box, list) or len(box) < 4:
        return {"offset_x": 0.0, "offset_y": 0.0, "size_x": area_x, "size_y": area_y}

    xmin = _clamp(_to_float(box[0]), 0.0, area_x)
    ymin = _clamp(_to_float(box[1]), 0.0, area_y)
    xmax = _clamp(_to_float(box[2]), 0.0, area_x)
    ymax = _clamp(_to_float(box[3]), 0.0, area_y)
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin

    sx = xmax - xmin
    sy = ymax - ymin
    if sx <= 0.05 or sy <= 0.05:
        return {"offset_x": 0.0, "offset_y": 0.0, "size_x": area_x, "size_y": area_y}

    return {"offset_x": xmin, "offset_y": ymin, "size_x": sx, "size_y": sy}


def _to_local_xy(x: float, y: float, tx: Dict[str, float]) -> Tuple[float, float]:
    lx = _clamp(x - tx["offset_x"], 0.0, tx["size_x"])
    ly = _clamp(y - tx["offset_y"], 0.0, tx["size_y"])
    return lx, ly


def _bbox_norm_to_world(box_norm: List[Any], area_x: float, area_y: float) -> Optional[Dict[str, float]]:
    if not isinstance(box_norm, list) or len(box_norm) < 4:
        return None
    try:
        nymin = float(box_norm[0])
        nxmin = float(box_norm[1])
        nymax = float(box_norm[2])
        nxmax = float(box_norm[3])
    except (TypeError, ValueError):
        return None

    xmin = _clamp(nxmin, 0.0, 1.0) * area_x
    xmax = _clamp(nxmax, 0.0, 1.0) * area_x
    ymax = (1.0 - _clamp(nymin, 0.0, 1.0)) * area_y
    ymin = (1.0 - _clamp(nymax, 0.0, 1.0)) * area_y

    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin

    dx = xmax - xmin
    dy = ymax - ymin
    if dx <= 0 or dy <= 0:
        return None

    return {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "cx": (xmin + xmax) * 0.5,
        "cy": (ymin + ymax) * 0.5,
        "dx": dx,
        "dy": dy,
    }


def _localize_bbox(bbox_world: Dict[str, float], tx: Dict[str, float]) -> Dict[str, float]:
    xmin = _clamp(bbox_world["xmin"] - tx["offset_x"], 0.0, tx["size_x"])
    xmax = _clamp(bbox_world["xmax"] - tx["offset_x"], 0.0, tx["size_x"])
    ymin = _clamp(bbox_world["ymin"] - tx["offset_y"], 0.0, tx["size_y"])
    ymax = _clamp(bbox_world["ymax"] - tx["offset_y"], 0.0, tx["size_y"])
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    return {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "cx": (xmin + xmax) * 0.5,
        "cy": (ymin + ymax) * 0.5,
        "dx": max(0.0, xmax - xmin),
        "dy": max(0.0, ymax - ymin),
    }


def _gemini_candidates_by_category(
    gemini_spatial_json: Optional[Dict[str, Any]],
    area_x: float,
    area_y: float,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if not isinstance(gemini_spatial_json, dict):
        return out
    objects = gemini_spatial_json.get("furniture_objects")
    if not isinstance(objects, list):
        return out

    for idx, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        category = _canonical_category(obj.get("category") or obj.get("label"))
        if not category or category in ARCH_CATEGORIES:
            continue
        box_world = _bbox_norm_to_world(obj.get("box_2d_norm"), area_x, area_y)
        if box_world is None:
            continue
        out[category].append({"uid": f"{category}:{idx}", "box_world": box_world, "raw": obj})
    return out


def _opening_candidates(
    opening_objects: Optional[List[Dict[str, Any]]],
    area_x: float,
    area_y: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(opening_objects, list):
        return out
    for idx, obj in enumerate(opening_objects):
        if not isinstance(obj, dict):
            continue
        category = _canonical_category(obj.get("category") or obj.get("label"))
        if "window" in category:
            kind = "window"
        elif "door" in category:
            kind = "door"
        else:
            continue
        box_world = _bbox_norm_to_world(obj.get("box_2d_norm"), area_x, area_y)
        if box_world is None:
            continue
        out.append({"uid": f"open:{idx}", "kind": kind, "category": category, "box_world": box_world, "raw": obj})
    return out


def _category_aliases_for_match(category: str) -> List[str]:
    cat = _canonical_category(category)
    if cat == "storage":
        return ["storage", "cabinet", "closet", "wardrobe"]
    if cat == "cabinet":
        return ["cabinet", "storage"]
    if cat == "tv_cabinet":
        return ["tv_cabinet", "tv", "cabinet", "storage"]
    if cat == "tv":
        return ["tv", "tv_cabinet"]
    return [cat]


def _pick_best_candidate(
    step1_obj: Dict[str, Any],
    category_candidates: Dict[str, List[Dict[str, Any]]],
    used_uids: set[str],
) -> Optional[Dict[str, Any]]:
    cx = _to_float(step1_obj.get("cx"), 0.0)
    cy = _to_float(step1_obj.get("cy"), 0.0)
    dx = abs(_to_float(step1_obj.get("dx"), 0.0))
    dy = abs(_to_float(step1_obj.get("dy"), 0.0))
    category = _canonical_category(step1_obj.get("category"))

    pool: List[Dict[str, Any]] = []
    for alias in _category_aliases_for_match(category):
        pool.extend(category_candidates.get(alias, []))

    best = None
    best_score = math.inf
    for cand in pool:
        if cand["uid"] in used_uids:
            continue
        bw = cand["box_world"]
        dc = math.hypot(bw["cx"] - cx, bw["cy"] - cy)
        ds = abs(bw["dx"] - dx) + abs(bw["dy"] - dy)
        score = dc + (0.35 * ds)
        if score < best_score:
            best_score = score
            best = cand
    if best is not None:
        used_uids.add(best["uid"])
    return best


def _opening_orientation(opening: Dict[str, Any], area_x: float, area_y: float) -> str:
    wall = str(opening.get("wall") or "").lower()
    if "partition_x" in wall or wall.startswith("x_") or "_x_" in wall or "east" in wall or "west" in wall:
        return "vertical"
    if "north" in wall or "south" in wall or "partition_y" in wall or wall.startswith("y_"):
        return "horizontal"

    cx = _to_float(opening.get("cx"), 0.0)
    cy = _to_float(opening.get("cy"), 0.0)
    d_left = abs(cx - 0.0)
    d_right = abs(cx - area_x)
    d_bottom = abs(cy - 0.0)
    d_top = abs(cy - area_y)
    return "vertical" if min(d_left, d_right) <= min(d_bottom, d_top) else "horizontal"


def _pick_opening_candidate(
    opening: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    used_uids: set[str],
) -> Optional[Dict[str, Any]]:
    want_type = str(opening.get("type") or "").lower()
    want_kind = "window" if want_type == "window" else "door"
    cx = _to_float(opening.get("cx"), 0.0)
    cy = _to_float(opening.get("cy"), 0.0)

    best = None
    best_score = math.inf
    for cand in candidates:
        if cand["uid"] in used_uids or cand["kind"] != want_kind:
            continue
        bw = cand["box_world"]
        score = math.hypot(bw["cx"] - cx, bw["cy"] - cy)
        if score < best_score:
            best_score = score
            best = cand
    if best is not None:
        used_uids.add(best["uid"])
    return best


def _infer_wall_facing_rotation(x: float, y: float, area_x: float, area_y: float) -> int:
    distances = {
        0: abs(x - 0.0),
        180: abs(x - area_x),
        90: abs(y - 0.0),
        270: abs(y - area_y),
    }
    return int(min(distances, key=distances.get))


def _extract_search_prompt(step1_obj: Dict[str, Any], category: str) -> str:
    for key in ("search_prompt", "searchPrompt", "asset_prompt", "prompt"):
        value = step1_obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return DEFAULT_SEARCH_PROMPT.get(category, f"{category} for indoor room")


def _rotation_for_object(step1_obj: Dict[str, Any], category: str, lx: float, ly: float, area_x: float, area_y: float) -> int:
    hinted = _normalize_angle(step1_obj.get("front_hint"))
    if hinted is not None:
        return hinted

    if category in {"sink", "storage", "cabinet", "tv_cabinet", "tv", "toilet"}:
        return _infer_wall_facing_rotation(lx, ly, area_x, area_y)

    dx = abs(_to_float(step1_obj.get("dx"), 0.0))
    dy = abs(_to_float(step1_obj.get("dy"), 0.0))
    return 90 if dy > dx else 0


def _ensure_room_id(
    room_id: Optional[str],
    cx: float,
    cy: float,
    rooms_out: List[Dict[str, Any]],
    default_room_id: str,
) -> str:
    if isinstance(room_id, str) and room_id.strip():
        return room_id
    for room in rooms_out:
        if _point_in_polygon(cx, cy, room["room_polygon"]):
            return room["room_id"]
    return default_room_id


def build_layout_rule_based(
    *,
    step1_json: Dict[str, Any],
    gemini_spatial_json: Optional[Dict[str, Any]] = None,
    room_inner_frame_objects: Optional[List[Dict[str, Any]]] = None,
    opening_objects: Optional[List[Dict[str, Any]]] = None,
    main_room_inner_boundary_hint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    del room_inner_frame_objects

    base_area_x, base_area_y = _extract_base_area(step1_json)
    tx = _extract_inner_transform(main_room_inner_boundary_hint, base_area_x, base_area_y)
    area_x = tx["size_x"]
    area_y = tx["size_y"]

    outer_polygon = [
        {"X": _r3(0.0), "Y": _r3(0.0)},
        {"X": _r3(area_x), "Y": _r3(0.0)},
        {"X": _r3(area_x), "Y": _r3(area_y)},
        {"X": _r3(0.0), "Y": _r3(area_y)},
    ]

    rooms_step1 = step1_json.get("rooms")
    rooms_out: List[Dict[str, Any]] = []
    if isinstance(rooms_step1, list):
        for idx, room in enumerate(rooms_step1):
            if not isinstance(room, dict):
                continue
            room_id = str(room.get("room_id") or f"room_{idx + 1}")
            room_name = str(room.get("room_name") or room_id)
            poly_src = room.get("room_polygon")
            poly_local: List[Dict[str, float]] = []
            if isinstance(poly_src, list):
                for p in poly_src:
                    if not isinstance(p, dict):
                        continue
                    x, y = _to_local_xy(_to_float(p.get("X")), _to_float(p.get("Y")), tx)
                    poly_local.append({"X": _r3(x), "Y": _r3(y)})
            poly_local = _clean_polygon(poly_local)
            if len(poly_local) < 3:
                poly_local = outer_polygon.copy()
            rooms_out.append({"room_id": room_id, "room_name": room_name, "room_polygon": poly_local, "openings": []})

    if not rooms_out:
        rooms_out = [{"room_id": "room_1", "room_name": "room", "room_polygon": outer_polygon.copy(), "openings": []}]

    primary_room_id = max(rooms_out, key=lambda r: _polygon_area(r["room_polygon"]))["room_id"]

    opening_cands = _opening_candidates(opening_objects, base_area_x, base_area_y)
    used_opening_uids: set[str] = set()
    openings_step1 = step1_json.get("openings")
    openings_final: List[Dict[str, Any]] = []
    if isinstance(openings_step1, list):
        for idx, opening in enumerate(openings_step1):
            if not isinstance(opening, dict):
                continue
            opening_id = str(opening.get("opening_id") or f"opening_{idx + 1}")
            opening_type = "window" if str(opening.get("type") or "").lower() == "window" else "door"
            base_cx = _to_float(opening.get("cx"), 0.0)
            base_cy = _to_float(opening.get("cy"), 0.0)
            local_cx, local_cy = _to_local_xy(base_cx, base_cy, tx)
            width = abs(_to_float(opening.get("w"), 0.0))
            height = abs(_to_float(opening.get("h"), DEFAULT_OBJECT_HEIGHT["door"]))
            sill = _to_float(opening.get("sill"), 0.0)
            orientation = _opening_orientation(opening, base_area_x, base_area_y)
            cand = _pick_opening_candidate(opening, opening_cands, used_opening_uids)
            if cand is not None:
                b_local = _localize_bbox(cand["box_world"], tx)
                if orientation == "vertical":
                    local_cx = _clamp(local_cx, 0.0, area_x)
                    local_cy = _clamp(b_local["cy"], 0.0, area_y)
                    width = max(0.05, b_local["dy"])
                else:
                    local_cx = _clamp(b_local["cx"], 0.0, area_x)
                    local_cy = _clamp(local_cy, 0.0, area_y)
                    width = max(0.05, b_local["dx"])
            room_ids_raw = opening.get("room_ids")
            room_ids = [str(r) for r in room_ids_raw if isinstance(r, (str, int, float))] if isinstance(room_ids_raw, list) else []
            openings_final.append(
                {
                    "opening_id": opening_id,
                    "type": opening_type,
                    "X": _r3(local_cx),
                    "Y": _r3(local_cy),
                    "Width": _r3(width),
                    "Height": _r3(height if height > 0 else DEFAULT_OBJECT_HEIGHT["door"]),
                    "SillHeight": _r3(max(0.0, sill)),
                    "_room_ids": room_ids,
                    "_orientation": orientation,
                }
            )

    for room in rooms_out:
        room_id = room["room_id"]
        room_poly = room["room_polygon"]
        room_openings = []
        for op in openings_final:
            room_ids = op.get("_room_ids") or []
            include = room_id in room_ids if room_ids else _point_in_polygon(op["X"], op["Y"], room_poly)
            if include:
                room_openings.append(
                    {
                        "type": op["type"],
                        "X": op["X"],
                        "Y": op["Y"],
                        "Width": op["Width"],
                        "Height": op["Height"],
                        "SillHeight": op["SillHeight"],
                    }
                )
        room["openings"] = room_openings

    windows: List[Dict[str, float]] = []
    seen_windows: set[Tuple[float, float, float, float, float]] = set()
    for op in openings_final:
        if op["type"] != "window":
            continue
        key = (op["X"], op["Y"], op["Width"], op["Height"], op["SillHeight"])
        if key in seen_windows:
            continue
        seen_windows.add(key)
        windows.append(
            {
                "X": op["X"],
                "Y": op["Y"],
                "Width": op["Width"],
                "Height": op["Height"],
                "SillHeight": op["SillHeight"],
            }
        )

    area_objects_list: List[Dict[str, Any]] = [
        {
            "object_name": "floor_1",
            "category": "floor",
            "search_prompt": DEFAULT_SEARCH_PROMPT["floor"],
            "room_id": primary_room_id,
            "X": _r3(area_x * 0.5),
            "Y": _r3(area_y * 0.5),
            "Length": _r3(area_x),
            "Width": _r3(area_y),
            "Height": _r3(DEFAULT_OBJECT_HEIGHT["floor"]),
            "rotationZ": 0,
        }
    ]

    category_candidates = _gemini_candidates_by_category(gemini_spatial_json, base_area_x, base_area_y)
    used_object_uids: set[str] = set()
    step1_objects = step1_json.get("objects")
    name_counters: Dict[str, int] = defaultdict(int)
    if isinstance(step1_objects, list):
        for obj in step1_objects:
            if not isinstance(obj, dict):
                continue
            category = _canonical_category(obj.get("category"))
            if not category or category in ARCH_CATEGORIES:
                continue
            object_name = str(obj.get("object_id") or "")
            if not object_name:
                name_counters[category] += 1
                object_name = f"{category}_{name_counters[category]}"

            room_id = obj.get("room_id") if isinstance(obj.get("room_id"), str) else None
            cand = _pick_best_candidate(obj, category_candidates, used_object_uids)

            if cand is not None:
                b_local = _localize_bbox(cand["box_world"], tx)
                cx, cy = b_local["cx"], b_local["cy"]
                length, width = b_local["dx"], b_local["dy"]
            else:
                cx_base = _to_float(obj.get("cx"), 0.0)
                cy_base = _to_float(obj.get("cy"), 0.0)
                cx, cy = _to_local_xy(cx_base, cy_base, tx)
                length = abs(_to_float(obj.get("dx"), 0.0))
                width = abs(_to_float(obj.get("dy"), 0.0))
                if (length <= 0 or width <= 0) and isinstance(obj.get("bbox"), list) and len(obj["bbox"]) >= 4:
                    bx0 = _to_float(obj["bbox"][0])
                    by0 = _to_float(obj["bbox"][1])
                    bx1 = _to_float(obj["bbox"][2])
                    by1 = _to_float(obj["bbox"][3])
                    length = abs(bx1 - bx0)
                    width = abs(by1 - by0)

            length = max(0.05, length)
            width = max(0.05, width)
            room_id_final = _ensure_room_id(room_id, cx, cy, rooms_out, primary_room_id)
            rotation = _rotation_for_object(obj, category, cx, cy, area_x, area_y)
            search_prompt = _extract_search_prompt(obj, category)
            height = _r3(DEFAULT_OBJECT_HEIGHT.get(category, 1.0))

            area_objects_list.append(
                {
                    "object_name": object_name,
                    "category": category,
                    "search_prompt": search_prompt,
                    "room_id": room_id_final,
                    "X": _r3(cx),
                    "Y": _r3(cy),
                    "Length": _r3(length),
                    "Width": _r3(width),
                    "Height": height,
                    "rotationZ": int(rotation),
                }
            )

    door_index = 0
    for op in openings_final:
        if op["type"] != "door":
            continue
        door_index += 1
        room_ids = [r for r in op.get("_room_ids", []) if str(r).lower() != "outside"]
        room_id = str(room_ids[0]) if room_ids else primary_room_id
        orient = op.get("_orientation")
        if orient == "vertical":
            rotation = 90
            door_length = 0.05
            door_width = op["Width"]
        else:
            rotation = 180 if op["Y"] >= (area_y * 0.5) else 0
            door_length = op["Width"]
            door_width = 0.05
        area_objects_list.append(
            {
                "object_name": f"door_{door_index}",
                "category": "door",
                "search_prompt": DEFAULT_SEARCH_PROMPT["door"],
                "room_id": room_id,
                "X": op["X"],
                "Y": op["Y"],
                "Length": _r3(max(0.05, door_length)),
                "Width": _r3(max(0.05, door_width)),
                "Height": op["Height"],
                "rotationZ": int(rotation),
            }
        )

    return {
        "area_name": str(step1_json.get("area_name") or "unit_plan"),
        "area_size_X": _r3(area_x),
        "area_size_Y": _r3(area_y),
        "size_mode": "world",
        "outer_polygon": outer_polygon,
        "rooms": rooms_out,
        "windows": windows,
        "area_objects_list": area_objects_list,
    }
