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

ROOM_LIKE_CATEGORY_TOKENS = {
    "room",
    "main_room",
    "subroom",
    "bedroom",
    "living_room",
    "dining_room",
    "kitchen",
    "bathroom",
    "washroom",
    "restroom",
    "toilet_room",
    "wc_room",
    "walk_in_closet",
    "closet_room",
    "corridor",
    "hallway",
    "entry",
    "entrance",
    "genkan",
    "studio",
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


def _is_room_like_detection_token(raw: Any) -> bool:
    token = _canonical_category(raw)
    if not token or token == "toilet":
        return False
    if token in ROOM_LIKE_CATEGORY_TOKENS:
        return True
    if "walk_in_closet" in token:
        return True
    if token.startswith("room_") or token.endswith("_room"):
        return True
    return False


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


def _rect_polygon_from_local_bbox(bbox_local: Dict[str, float]) -> List[Dict[str, float]]:
    return _clean_polygon(
        [
            {"X": _r3(bbox_local["xmin"]), "Y": _r3(bbox_local["ymin"])},
            {"X": _r3(bbox_local["xmax"]), "Y": _r3(bbox_local["ymin"])},
            {"X": _r3(bbox_local["xmax"]), "Y": _r3(bbox_local["ymax"])},
            {"X": _r3(bbox_local["xmin"]), "Y": _r3(bbox_local["ymax"])},
        ]
    )


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


def _extract_step1_rooms_local(
    step1_json: Dict[str, Any],
    tx: Dict[str, float],
    outer_polygon: List[Dict[str, float]],
) -> List[Dict[str, Any]]:
    rooms_step1 = step1_json.get("rooms")
    rooms_out: List[Dict[str, Any]] = []
    if not isinstance(rooms_step1, list):
        return rooms_out

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
    return rooms_out


def _build_rooms_from_gemini_inner_frames(
    room_inner_frame_objects: Optional[List[Dict[str, Any]]],
    step1_rooms_local: List[Dict[str, Any]],
    tx: Dict[str, float],
    base_area_x: float,
    base_area_y: float,
    area_x: float,
    area_y: float,
    obstacle_boxes_local: Optional[List[Dict[str, float]]],
    outer_polygon: List[Dict[str, float]],
) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(room_inner_frame_objects, list):
        return None

    parsed: List[Dict[str, Any]] = []
    for obj in room_inner_frame_objects:
        if not isinstance(obj, dict):
            continue
        category = _canonical_category(obj.get("category") or obj.get("label"))
        if "inner_frame" not in category:
            continue
        bw = _bbox_norm_to_world(obj.get("box_2d_norm"), base_area_x, base_area_y)
        if bw is None:
            continue
        bl = _localize_bbox(bw, tx)
        if bl["dx"] <= 0.05 or bl["dy"] <= 0.05:
            continue
        parsed.append(
            {
                "category": category,
                "bbox_local": bl,
                "is_subroom": ("subroom" in category or "sub_room" in category),
            }
        )

    if not parsed:
        return None

    main_template = None
    if step1_rooms_local:
        main_template = max(step1_rooms_local, key=lambda r: _polygon_area(r["room_polygon"]))

    main_room_id = str(main_template["room_id"]) if isinstance(main_template, dict) else "room_1"
    main_room_name = str(main_template["room_name"]) if isinstance(main_template, dict) else "room"

    rooms_out: List[Dict[str, Any]] = [
        {
            "room_id": main_room_id,
            "room_name": main_room_name,
            "room_polygon": outer_polygon.copy(),
            "openings": [],
        }
    ]
    used_room_ids = {main_room_id}

    subrooms = [p for p in parsed if bool(p.get("is_subroom"))]
    subrooms.sort(key=lambda x: float(x["bbox_local"]["dx"] * x["bbox_local"]["dy"]), reverse=True)

    for idx, sub in enumerate(subrooms):
        b = dict(sub["bbox_local"])
        b = _snap_subroom_bbox_to_main_if_near(
            bbox_local=b,
            area_x=area_x,
            area_y=area_y,
            obstacle_boxes_local=(obstacle_boxes_local or []),
            snap_threshold=0.8,
            max_snap_edges=2,
        )
        poly = _rect_polygon_from_local_bbox(b)
        if len(poly) < 3:
            continue

        cx = float(b["cx"])
        cy = float(b["cy"])
        matched: Optional[Dict[str, Any]] = None
        best_area = math.inf
        for r in step1_rooms_local:
            rid = str(r.get("room_id") or "")
            if not rid or rid == main_room_id:
                continue
            if _point_in_polygon(cx, cy, r["room_polygon"]):
                area = _polygon_area(r["room_polygon"])
                if area < best_area:
                    best_area = area
                    matched = r

        if matched is not None:
            room_id = str(matched.get("room_id") or f"subroom_{idx + 1}")
            room_name = str(matched.get("room_name") or room_id)
        else:
            room_id = f"subroom_{idx + 1}"
            room_name = "subroom"

        if room_id in used_room_ids:
            room_id = f"{room_id}_{idx + 1}"
        used_room_ids.add(room_id)

        rooms_out.append(
            {
                "room_id": room_id,
                "room_name": room_name,
                "room_polygon": poly,
                "openings": [],
            }
        )

    return rooms_out if rooms_out else None


def _collect_obstacle_boxes_local(
    gemini_spatial_json: Optional[Dict[str, Any]],
    tx: Dict[str, float],
    base_area_x: float,
    base_area_y: float,
) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    if not isinstance(gemini_spatial_json, dict):
        return out
    objects = gemini_spatial_json.get("furniture_objects")
    if not isinstance(objects, list):
        return out

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        category = _canonical_category(obj.get("category") or obj.get("label"))
        label_category = _canonical_category(obj.get("label"))
        if not category or category in ARCH_CATEGORIES:
            continue
        if _is_room_like_detection_token(category) or _is_room_like_detection_token(label_category):
            continue
        bw = _bbox_norm_to_world(obj.get("box_2d_norm"), base_area_x, base_area_y)
        if bw is None:
            continue
        bl = _localize_bbox(bw, tx)
        if bl["dx"] <= 0.03 or bl["dy"] <= 0.03:
            continue
        out.append(bl)
    return out


def _rects_intersect(a: Dict[str, float], b: Dict[str, float], eps: float = 1e-6) -> bool:
    return (
        min(a["xmax"], b["xmax"]) - max(a["xmin"], b["xmin"]) > eps
        and min(a["ymax"], b["ymax"]) - max(a["ymin"], b["ymin"]) > eps
    )


def _subroom_strip_blocked(
    edge: str,
    bbox_local: Dict[str, float],
    area_x: float,
    area_y: float,
    obstacle_boxes_local: List[Dict[str, float]],
) -> bool:
    if edge == "left":
        strip = {"xmin": 0.0, "xmax": bbox_local["xmin"], "ymin": bbox_local["ymin"], "ymax": bbox_local["ymax"]}
    elif edge == "right":
        strip = {"xmin": bbox_local["xmax"], "xmax": area_x, "ymin": bbox_local["ymin"], "ymax": bbox_local["ymax"]}
    elif edge == "bottom":
        strip = {"xmin": bbox_local["xmin"], "xmax": bbox_local["xmax"], "ymin": 0.0, "ymax": bbox_local["ymin"]}
    elif edge == "top":
        strip = {"xmin": bbox_local["xmin"], "xmax": bbox_local["xmax"], "ymin": bbox_local["ymax"], "ymax": area_y}
    else:
        return True

    if (strip["xmax"] - strip["xmin"]) <= 0.015 or (strip["ymax"] - strip["ymin"]) <= 0.015:
        return False

    for ob in obstacle_boxes_local:
        if _rects_intersect(strip, ob):
            return True
    return False


def _snap_subroom_bbox_to_main_if_near(
    bbox_local: Dict[str, float],
    area_x: float,
    area_y: float,
    obstacle_boxes_local: List[Dict[str, float]],
    snap_threshold: float = 0.14,
    max_snap_edges: int = 2,
) -> Dict[str, float]:
    b = dict(bbox_local)
    distances = [
        ("left", float(b["xmin"])),
        ("right", float(area_x - b["xmax"])),
        ("bottom", float(b["ymin"])),
        ("top", float(area_y - b["ymax"])),
    ]
    distances.sort(key=lambda kv: kv[1])

    snapped = 0
    for edge, dist in distances:
        if snapped >= max_snap_edges:
            break
        if dist < 0.0 or dist > float(snap_threshold):
            continue
        if _subroom_strip_blocked(edge, b, area_x, area_y, obstacle_boxes_local):
            continue
        if edge == "left":
            b["xmin"] = 0.0
        elif edge == "right":
            b["xmax"] = area_x
        elif edge == "bottom":
            b["ymin"] = 0.0
        elif edge == "top":
            b["ymax"] = area_y
        snapped += 1

    b["xmin"] = _clamp(float(b["xmin"]), 0.0, area_x)
    b["xmax"] = _clamp(float(b["xmax"]), 0.0, area_x)
    b["ymin"] = _clamp(float(b["ymin"]), 0.0, area_y)
    b["ymax"] = _clamp(float(b["ymax"]), 0.0, area_y)
    if b["xmax"] < b["xmin"]:
        b["xmin"], b["xmax"] = b["xmax"], b["xmin"]
    if b["ymax"] < b["ymin"]:
        b["ymin"], b["ymax"] = b["ymax"], b["ymin"]
    b["dx"] = max(0.0, b["xmax"] - b["xmin"])
    b["dy"] = max(0.0, b["ymax"] - b["ymin"])
    b["cx"] = (b["xmin"] + b["xmax"]) * 0.5
    b["cy"] = (b["ymin"] + b["ymax"]) * 0.5
    return b


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
        label_category = _canonical_category(obj.get("label"))
        if not category or category in ARCH_CATEGORIES:
            continue
        if _is_room_like_detection_token(category) or _is_room_like_detection_token(label_category):
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


def _opening_front_hint(
    opening: Dict[str, Any],
    orientation: str,
    x: float,
    y: float,
    area_x: float,
    area_y: float,
) -> int:
    hinted = _normalize_angle(opening.get("front_hint"))
    if hinted is None:
        for key in ("door_front_hint", "rotationZ", "rotation"):
            hinted = _normalize_angle(opening.get(key))
            if hinted is not None:
                break

    if orientation == "vertical":
        allowed = {90, 270}
        fallback = 90 if x <= (area_x * 0.5) else 270
    else:
        allowed = {0, 180}
        fallback = 0 if y <= (area_y * 0.5) else 180

    if hinted in allowed:
        return int(hinted)
    return int(fallback)


def _opening_wall_role(opening: Dict[str, Any]) -> str:
    room_ids_raw = opening.get("room_ids")
    if isinstance(room_ids_raw, list):
        room_ids = [str(r).strip().lower() for r in room_ids_raw if isinstance(r, (str, int, float))]
        room_ids = [r for r in room_ids if r]
        if any(r in {"outside", "outdoor", "exterior"} for r in room_ids):
            return "outer"
        uniq = {r for r in room_ids if r not in {"outside", "outdoor", "exterior"}}
        if len(uniq) >= 2:
            return "interior"

    wall = str(opening.get("wall") or "").strip().lower()
    if not wall:
        return "unknown"
    if any(tok in wall for tok in ("outer", "outside", "exterior", "north", "south", "east", "west")):
        return "outer"
    if any(tok in wall for tok in ("_left_wall", "_right_wall", "_top_wall", "_bottom_wall")):
        return "interior"
    if any(tok in wall for tok in ("interior", "partition", "shared")):
        return "interior"
    return "unknown"


def _pick_opening_candidate(
    opening: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    used_uids: set[str],
    orientation: str,
    base_width: float,
    area_x: float,
    area_y: float,
) -> Optional[Dict[str, Any]]:
    want_type = str(opening.get("type") or "").lower()
    want_kind = "window" if want_type == "window" else "door"
    cx = _to_float(opening.get("cx"), 0.0)
    cy = _to_float(opening.get("cy"), 0.0)
    base_w = max(0.05, abs(base_width))
    wall_role = _opening_wall_role(opening)
    outer_band = max(0.25, 0.08 * min(max(area_x, 0.1), max(area_y, 0.1)))

    best = None
    best_score = math.inf
    best_any = None
    best_any_score = math.inf
    for cand in candidates:
        if cand["uid"] in used_uids or cand["kind"] != want_kind:
            continue
        bw = cand["box_world"]
        if orientation == "vertical":
            cand_width = max(0.05, bw["dy"])
            d_along = abs(bw["cy"] - cy)
            d_cross = abs(bw["cx"] - cx)
            cand_orient = "vertical" if bw["dy"] >= bw["dx"] else "horizontal"
        else:
            cand_width = max(0.05, bw["dx"])
            d_along = abs(bw["cx"] - cx)
            d_cross = abs(bw["cy"] - cy)
            cand_orient = "horizontal" if bw["dx"] >= bw["dy"] else "vertical"

        width_ratio = cand_width / base_w
        width_penalty = 0.0
        if width_ratio < 0.45 or width_ratio > 2.0:
            width_penalty += 2.0
        elif width_ratio < 0.65 or width_ratio > 1.6:
            width_penalty += 0.8
        if "sliding" in str(cand.get("category") or "").lower() and width_ratio > 1.35:
            width_penalty += 1.2
        orient_penalty = 0.0 if cand_orient == orientation else 0.9
        cand_outer_dist = min(
            abs(bw["cx"] - 0.0),
            abs(area_x - bw["cx"]),
            abs(bw["cy"] - 0.0),
            abs(area_y - bw["cy"]),
        )
        cand_is_outer = cand_outer_dist <= outer_band
        # Cross-wall mismatch is strongly penalized.
        score = (
            d_along
            + (2.8 * d_cross)
            + (0.35 * abs(cand_width - base_w))
            + width_penalty
            + orient_penalty
        )
        if score < best_any_score:
            best_any_score = score
            best_any = cand

        wall_role_match = (
            wall_role == "unknown"
            or (wall_role == "outer" and cand_is_outer)
            or (wall_role == "interior" and not cand_is_outer)
        )
        if not wall_role_match:
            continue

        if score < best_score:
            best_score = score
            best = cand

    chosen = best if best is not None else best_any
    if chosen is not None:
        used_uids.add(chosen["uid"])
    return chosen


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


def _room_bbox_from_polygon(room_polygon: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not isinstance(room_polygon, list) or len(room_polygon) < 3:
        return None
    xs: List[float] = []
    ys: List[float] = []
    for p in room_polygon:
        if not isinstance(p, dict):
            continue
        xs.append(_to_float(p.get("X"), 0.0))
        ys.append(_to_float(p.get("Y"), 0.0))
    if not xs or not ys:
        return None
    return {
        "xmin": min(xs),
        "xmax": max(xs),
        "ymin": min(ys),
        "ymax": max(ys),
    }


def _project_opening_to_room_edge(
    opening: Dict[str, Any],
    room_polygon: List[Dict[str, float]],
) -> Tuple[float, float]:
    rb = _room_bbox_from_polygon(room_polygon)
    if rb is None:
        return _to_float(opening.get("X"), 0.0), _to_float(opening.get("Y"), 0.0)

    x = _to_float(opening.get("X"), 0.0)
    y = _to_float(opening.get("Y"), 0.0)
    orientation = str(opening.get("_orientation") or "").lower()

    if orientation == "vertical":
        x = rb["xmin"] if abs(x - rb["xmin"]) <= abs(x - rb["xmax"]) else rb["xmax"]
        y = _clamp(y, rb["ymin"], rb["ymax"])
    elif orientation == "horizontal":
        y = rb["ymin"] if abs(y - rb["ymin"]) <= abs(y - rb["ymax"]) else rb["ymax"]
        x = _clamp(x, rb["xmin"], rb["xmax"])
    else:
        dists = [
            ("left", abs(x - rb["xmin"])),
            ("right", abs(x - rb["xmax"])),
            ("bottom", abs(y - rb["ymin"])),
            ("top", abs(y - rb["ymax"])),
        ]
        edge = min(dists, key=lambda kv: kv[1])[0]
        if edge == "left":
            x = rb["xmin"]
            y = _clamp(y, rb["ymin"], rb["ymax"])
        elif edge == "right":
            x = rb["xmax"]
            y = _clamp(y, rb["ymin"], rb["ymax"])
        elif edge == "bottom":
            y = rb["ymin"]
            x = _clamp(x, rb["xmin"], rb["xmax"])
        else:
            y = rb["ymax"]
            x = _clamp(x, rb["xmin"], rb["xmax"])
    return _r3(x), _r3(y)


def build_layout_rule_based(
    *,
    step1_json: Dict[str, Any],
    gemini_spatial_json: Optional[Dict[str, Any]] = None,
    room_inner_frame_objects: Optional[List[Dict[str, Any]]] = None,
    opening_objects: Optional[List[Dict[str, Any]]] = None,
    main_room_inner_boundary_hint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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

    step1_rooms_local = _extract_step1_rooms_local(step1_json, tx, outer_polygon)
    obstacle_boxes_local = _collect_obstacle_boxes_local(
        gemini_spatial_json=gemini_spatial_json,
        tx=tx,
        base_area_x=base_area_x,
        base_area_y=base_area_y,
    )
    rooms_out = _build_rooms_from_gemini_inner_frames(
        room_inner_frame_objects=room_inner_frame_objects,
        step1_rooms_local=step1_rooms_local,
        tx=tx,
        base_area_x=base_area_x,
        base_area_y=base_area_y,
        area_x=area_x,
        area_y=area_y,
        obstacle_boxes_local=obstacle_boxes_local,
        outer_polygon=outer_polygon,
    )
    if rooms_out is None:
        rooms_out = step1_rooms_local

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
            cand = _pick_opening_candidate(
                opening,
                opening_cands,
                used_opening_uids,
                orientation,
                width,
                base_area_x,
                base_area_y,
            )
            if cand is not None:
                b_local = _localize_bbox(cand["box_world"], tx)
                # Geometry responsibility: prefer Gemini for openings.
                orientation = "vertical" if b_local["dy"] >= b_local["dx"] else "horizontal"
                local_cx = _clamp(b_local["cx"], 0.0, area_x)
                local_cy = _clamp(b_local["cy"], 0.0, area_y)
                width = max(0.05, b_local["dy"] if orientation == "vertical" else b_local["dx"])
            room_ids_raw = opening.get("room_ids")
            room_ids = [str(r) for r in room_ids_raw if isinstance(r, (str, int, float))] if isinstance(room_ids_raw, list) else []
            front_hint = _opening_front_hint(opening, orientation, local_cx, local_cy, area_x, area_y)
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
                    "_front_hint": int(front_hint),
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
                ox, oy = op["X"], op["Y"]
                if op.get("type") == "door" and room_id != primary_room_id:
                    ox, oy = _project_opening_to_room_edge(op, room_poly)
                room_openings.append(
                    {
                        "type": op["type"],
                        "X": ox,
                        "Y": oy,
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
        door_x, door_y = op["X"], op["Y"]
        if room_id != primary_room_id:
            target_room = next((r for r in rooms_out if str(r.get("room_id")) == room_id), None)
            if isinstance(target_room, dict):
                door_x, door_y = _project_opening_to_room_edge(op, target_room.get("room_polygon") or [])
        orient = op.get("_orientation")
        rotation = _normalize_angle(op.get("_front_hint"))
        if rotation is None:
            if orient == "vertical":
                rotation = 90 if door_x <= (area_x * 0.5) else 270
            else:
                rotation = 0 if door_y <= (area_y * 0.5) else 180
        if orient == "vertical":
            door_length = 0.05
            door_width = op["Width"]
        else:
            door_length = op["Width"]
            door_width = 0.05
        area_objects_list.append(
            {
                "object_name": f"door_{door_index}",
                "category": "door",
                "search_prompt": DEFAULT_SEARCH_PROMPT["door"],
                "room_id": room_id,
                "X": door_x,
                "Y": door_y,
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
