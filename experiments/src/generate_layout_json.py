from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except Exception:  # noqa: BLE001
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False
from openai import OpenAI

from layout_tools import extract_json_payload, read_text, write_json
from step2_rule_based import build_layout_rule_based

DEFAULT_MODEL = "gpt-5.2"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_TEXT_VERBOSITY = "high"
DEFAULT_MAX_OUTPUT_TOKENS = 32000
DEFAULT_IMAGE_DETAIL = "high"

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_TASK = "boxes"
DEFAULT_GEMINI_LABEL_LANGUAGE = "English"
DEFAULT_GEMINI_TEMPERATURE = 0.6
DEFAULT_GEMINI_THINKING_BUDGET = 0
DEFAULT_GEMINI_MAX_ITEMS = 20
DEFAULT_GEMINI_RESIZE_MAX = 640
DEFAULT_GEMINI_ROOM_INNER_FRAME_PROMPT = (
    "This image is a floor plan. Detect only room inner boundary frames (inside face of walls). "
    "Return a JSON list where each entry has keys \"label\", \"box_2d\", and \"mask\". "
    "Use label \"room_inner_frame\" for exactly one global main interior envelope and \"subroom_inner_frame\" for each enclosed sub-room. "
    "The main room_inner_frame must be the largest axis-aligned rectangle that stays inside inner wall lines of the whole unit. "
    "Do NOT shrink the main room to avoid subrooms; subroom_inner_frame may be inside/overlap the main room rectangle. "
    "Each subroom_inner_frame must also be the largest axis-aligned rectangle inside that enclosed subroom, touching enclosing inner walls whenever possible. "
    "Each frame must be tight to inside wall lines with zero intentional padding (no inset and no expansion), and expanded maximally until any side would cross outside inner walls. "
    "Do not shrink any frame to avoid text labels; prioritize wall lines over text glyphs. "
    "The mask must represent interior floor area for that same rectangle. "
    "Do not include wall thickness, exterior space, furniture, text labels, doors, or windows. "
    "box_2d must be the tight rectangle of the returned mask."
)
DEFAULT_GEMINI_OPENINGS_PROMPT = (
    "This image is a floor plan. Detect only architectural openings that are explicitly labeled as "
    "\"Door\", \"Sliding Door\", or \"Window\" in the room/wall context. "
    "Return tight bounding boxes around ONLY the clear opening (the gap where wall is absent and people pass through), "
    "not around text labels. "
    "For hinged doors: box only the doorway gap at the wall break (swing arc may be included only if it lies inside the gap). "
    "For sliding doors: box only the wall-break opening segment; DO NOT include the sliding panel storage/pocket region, "
    "door leaf parked area, or rail extension beyond the wall-break gap. "
    "Do NOT detect furniture doors (e.g., storage/cabinet/closet doors). "
    "Return a JSON list where each entry has keys \"label\" and \"box_2d\". "
    "Use labels from: \"door\", \"sliding_door\", \"window\". "
    "Do not include furniture, room areas, walls, or other non-opening elements."
)
DEFAULT_GEMINI_FURNITURE_PROMPT = (
    "This image is a floor plan. Detect the furniture/equipment shapes (outline lines) and return tight bounding boxes "
    "around the outer contour of each item. Do not include walls, windows, doors, or other architectural lines."
)


def _extract_dimensions_value(dimensions_text: str, key: str) -> Optional[str]:
    target = f"{key.strip().upper()}="
    for raw in str(dimensions_text or "").splitlines():
        line = raw.strip().lstrip("\ufeff")
        if not line:
            continue
        if line.upper().startswith(target):
            return line.split("=", 1)[1].strip()
    return None


def _parse_dimensions_rooms(dimensions_text: str) -> Dict[str, int]:
    raw_rooms = _extract_dimensions_value(dimensions_text, "ROOMS")
    if not raw_rooms:
        return {}

    out: Dict[str, int] = {}
    for token in raw_rooms.split(","):
        part = token.strip()
        if not part or ":" not in part:
            continue
        name_raw, count_raw = part.split(":", 1)
        name = str(name_raw or "").strip().lower().replace(" ", "_").replace("-", "_")
        if not name:
            continue
        try:
            count = int(float(str(count_raw).strip()))
        except (TypeError, ValueError):
            continue
        if count <= 0:
            continue
        out[name] = out.get(name, 0) + count
    return out


def _parse_dimensions_float(dimensions_text: str, key: str) -> Optional[float]:
    raw = _extract_dimensions_value(dimensions_text, key)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _build_room_inner_frame_constraints(dimensions_text: str) -> Tuple[str, Optional[str]]:
    rooms = _parse_dimensions_rooms(dimensions_text)
    if not rooms:
        return "items", None

    total_rooms = int(sum(rooms.values()))
    if total_rooms <= 0:
        return "items", None

    # Current inner-frame integration is rectangle-based; one main room + subrooms.
    main_count = 1
    subroom_count = max(0, total_rooms - main_count)

    main_like = {
        "ldk",
        "living",
        "dining",
        "kitchen",
        "bedroom",
        "kids_room",
        "study",
        "office",
        "guest_room",
    }
    subroom_types: List[str] = []
    for name, count in sorted(rooms.items()):
        if name in main_like:
            continue
        subroom_types.append(f"{name}:{count}")

    sx = _parse_dimensions_float(dimensions_text, "SCALE_WIDTH_M")
    sy = _parse_dimensions_float(dimensions_text, "SCALE_HEIGHT_M")
    area_line = (
        f"- AREA_SIZE_M: {sx:.3f} x {sy:.3f}"
        if isinstance(sx, float) and isinstance(sy, float) and sx > 0.0 and sy > 0.0
        else None
    )

    lines: List[str] = [
        "- LABEL_SET: room_inner_frame | subroom_inner_frame",
        f"- EXPECTED_TOTAL_ROOMS: {total_rooms}",
        f"- EXPECTED_MAIN_ROOM_COUNT: {main_count}",
        f"- EXPECTED_SUBROOM_COUNT: {subroom_count}",
        "- MAIN_ROOM_RULE: room_inner_frame must be the global maximum inner rectangle of the whole unit.",
        "- MAIN_ROOM_COVERAGE_RULE: room_inner_frame should span most of the apartment interior footprint unless impossible.",
        "- MAIN_SUBROOM_RELATION_RULE: do not shrink room_inner_frame to avoid subrooms; subroom boxes may overlap/nest inside it.",
        "- BOX_OVERLAP_RULE: subroom_inner_frame boxes may overlap room_inner_frame (rectangular approximation).",
        "- MAX_RECT_RULE: for each frame, expand rectangle until any side would cross outside inner wall lines.",
        "- SUBROOM_MAX_RECT_RULE: each subroom_inner_frame must be maximal in its enclosed room and should touch enclosing inner walls.",
        "- TIGHTNESS_RULE: boundaries must follow inside wall lines with no margin/inset padding.",
        "- TEXT_PRIORITY_RULE: never shrink frame to avoid printed text; wall lines have priority.",
        "- MASK_BOX_CONSISTENCY_RULE: box_2d must be the tight envelope of mask.",
    ]
    if area_line:
        lines.insert(0, area_line)
    if subroom_types:
        lines.append(f"- EXPECTED_SUBROOM_TYPES: {', '.join(subroom_types)}")
    lines.append("- OUTPUT_COUNT_RULE: if visible, return exactly these counts.")

    if subroom_count > 0:
        target_prompt = f"room_inner_frame x1, subroom_inner_frame x{subroom_count}"
    else:
        target_prompt = "room_inner_frame x1"
    return target_prompt, "\n".join(lines)


def _build_gemini_furniture_prompt(
    user_prompt_text: Optional[str],
    dimensions_text: str,
    step1_category_counts: Optional[Dict[str, int]] = None,
) -> str:
    base = (
        str(user_prompt_text).strip()
        if isinstance(user_prompt_text, str) and str(user_prompt_text).strip()
        else DEFAULT_GEMINI_FURNITURE_PROMPT
    )

    rooms = _parse_dimensions_rooms(dimensions_text)
    room_names = [name.replace("_", " ") for name, count in sorted(rooms.items()) if int(count) > 0]
    room_line = ", ".join(room_names) if room_names else "unknown"
    inventory_line = ""
    if isinstance(step1_category_counts, dict):
        pairs = [f"{k} x{int(v)}" for k, v in sorted(step1_category_counts.items()) if int(v) > 0]
        if pairs:
            inventory_line = ", ".join(pairs)

    guard_lines = [
        "ADDITIONAL_CONSTRAINTS:",
        f"- Rooms present may include: {room_line}.",
    ]
    if inventory_line:
        guard_lines.append(f"- Expected furniture inventory: {inventory_line}.")
    guard_lines.extend(
        [
            "- Do NOT return room areas/room labels as furniture objects.",
            "- In particular, labels such as walk-in closet / closet room are room labels, not furniture objects.",
            "- IMPORTANT EXCEPTION: \"toilet\" may be both a room name and a fixture category. "
            "Return category \"toilet\" only when the toilet fixture outline is visible; never for room text/area.",
            "- If sink and storage are adjacent in one niche/column, return separate boxes for each item; do not merge.",
            "- For sink/storage, ignore printed text and box only the fixture/cabinet outer contour.",
            "- Prefer fixture contour edges/dividers over text labels when two fixtures are stacked vertically.",
        ]
    )
    guard = "\n".join(guard_lines)
    return f"{base}\n\n{guard}"


def _get_usage_value(usage: Any, key: str, default: Any = None) -> Any:
    if usage is None:
        return default
    if isinstance(usage, dict):
        return usage.get(key, default)
    return getattr(usage, key, default)


def _usage_tokens(usage: Any) -> Dict[str, int]:
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    input_tokens = _get_usage_value(usage, "input_tokens", None)
    output_tokens = _get_usage_value(usage, "output_tokens", None)
    total_tokens = _get_usage_value(usage, "total_tokens", None)

    if input_tokens is None and output_tokens is None:
        input_tokens = _get_usage_value(usage, "prompt_tokens", 0)
        output_tokens = _get_usage_value(usage, "completion_tokens", 0)
        total_tokens = _get_usage_value(usage, "total_tokens", int(input_tokens) + int(output_tokens))

    if total_tokens is None:
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)

    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    output = getattr(response, "output", None)
    if isinstance(output, list):
        chunks: List[str] = []
        for item in output:
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for c in content:
                ctype = c.get("type") if isinstance(c, dict) else getattr(c, "type", None)
                if ctype in {"output_text", "text"}:
                    ctext = c.get("text") if isinstance(c, dict) else getattr(c, "text", None)
                    if isinstance(ctext, str):
                        chunks.append(ctext)
        if chunks:
            return "".join(chunks)

    return ""


def _extract_gemini_response_text(response: Dict[str, Any]) -> str:
    candidates = response.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError(f"Unexpected Gemini response: no candidates field. response={response}")

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        content = cand.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        chunks: List[str] = []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
        merged = "".join(chunks).strip()
        if merged:
            return merged

    prompt_feedback = response.get("promptFeedback") or response.get("prompt_feedback")
    raise RuntimeError(f"No text in Gemini candidates. promptFeedback={prompt_feedback}")


def _gemini_usage_tokens(response: Dict[str, Any]) -> Dict[str, int]:
    usage = response.get("usageMetadata") or response.get("usage_metadata") or {}
    input_tokens = int(usage.get("promptTokenCount") or 0)
    output_tokens = int(usage.get("candidatesTokenCount") or 0)
    total_tokens = int(usage.get("totalTokenCount") or (input_tokens + output_tokens))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _call_gemini_generate_content(
    *,
    model: str,
    prompt_text: str,
    image_base64: Optional[str],
    image_mime: str,
    max_output_tokens: int,
    temperature: float,
    thinking_budget: int,
    gemini_api_key: Optional[str],
) -> Tuple[str, Dict[str, int], str]:
    key = gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Gemini LLM call requested but GOOGLE_API_KEY/GEMINI_API_KEY is not set (or --gemini_api_key missing).")

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    parts: List[Dict[str, Any]] = []
    if image_base64:
        parts.append(
            {
                "inline_data": {
                    "mime_type": image_mime,
                    "data": image_base64,
                }
            }
        )
    parts.append({"text": prompt_text})

    generation_config: Dict[str, Any] = {
        "temperature": float(temperature),
        "maxOutputTokens": int(max_output_tokens),
    }
    if "gemini-2.0-flash" not in str(model):
        generation_config["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}

    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": generation_config,
    }

    req = urllib.request.Request(
        url=endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        err = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API HTTP {exc.code}: {err}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini response is not JSON: {raw[:500]}") from exc

    output_text = _extract_gemini_response_text(parsed)
    usage = _gemini_usage_tokens(parsed)
    model_used = str(parsed.get("modelVersion") or model)
    return output_text, usage, model_used


def _encode_image_base64(image_path: pathlib.Path) -> Tuple[str, str]:
    raw = image_path.read_bytes()
    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime:
        mime = "image/png"
    return base64.b64encode(raw).decode("ascii"), mime


def _build_responses_input(
    prompt_text: str,
    image_base64: Optional[str],
    image_mime: str,
    image_detail: str,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt_text}]
    if image_base64:
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{image_mime};base64,{image_base64}",
                "detail": image_detail,
            }
        )
    return [{"role": "user", "content": content}]


def _create_response_with_fallback(client: OpenAI, kwargs: Dict[str, Any]) -> Any:
    attempts = [
        (),
        ("text",),
        ("reasoning",),
        ("text", "reasoning"),
    ]
    last_exc: Optional[Exception] = None
    for drop_keys in attempts:
        attempt_kwargs = {k: v for k, v in kwargs.items() if k not in set(drop_keys)}
        try:
            return client.responses.create(**attempt_kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    assert last_exc is not None
    raise last_exc


def _call_responses(
    client: OpenAI,
    model: str,
    prompt_text: str,
    image_base64: Optional[str],
    image_mime: str,
    image_detail: str,
    reasoning_effort: str,
    text_verbosity: str,
    max_output_tokens: int,
) -> Tuple[str, Dict[str, int], str]:
    effort = str(reasoning_effort or "").strip().lower()
    if effort in {"middle", "mid"}:
        effort = "medium"

    kwargs: Dict[str, Any] = {
        "model": model,
        "input": _build_responses_input(prompt_text, image_base64, image_mime, image_detail),
        "max_output_tokens": int(max_output_tokens),
    }
    if effort:
        kwargs["reasoning"] = {"effort": effort}
    if text_verbosity:
        kwargs["text"] = {"verbosity": text_verbosity}

    response = _create_response_with_fallback(client, kwargs)
    output_text = _extract_response_text(response)
    if not output_text.strip():
        raise RuntimeError("OpenAI response text was empty")

    usage = _usage_tokens(getattr(response, "usage", None))
    model_used = str(getattr(response, "model", model))
    return output_text, usage, model_used


def _load_json_file(path: pathlib.Path) -> Dict[str, Any]:
    raw = read_text(path)
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        return {"data": parsed}
    return parsed


def _collect_step1_category_counts(step1_json: Dict[str, Any]) -> Dict[str, int]:
    objects = step1_json.get("objects")
    if not isinstance(objects, list):
        return {}

    counts: Counter[str] = Counter()
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        category = str(obj.get("category") or "").strip().lower()
        if not category:
            continue
        if category in {"floor", "wall", "door", "window"}:
            continue
        counts[category] += 1

    return {k: int(v) for k, v in sorted(counts.items(), key=lambda kv: kv[0])}


def _build_default_gemini_target_prompt(step1_category_counts: Dict[str, int]) -> str:
    if not step1_category_counts:
        return "items"
    pairs = [f"{k} x{v}" for k, v in step1_category_counts.items()]
    return "objects matching this inventory: " + ", ".join(pairs)


def _is_room_inner_frame_category(category: str, label: str) -> bool:
    c = str(category or "").strip().lower().replace("-", "_")
    l = str(label or "").strip().lower()
    label_raw = str(label or "")

    explicit = {
        "room_inner_frame",
        "main_room_inner_frame",
        "subroom_inner_frame",
        "sub_room_inner_frame",
        "inner_room_frame",
    }
    if c in explicit:
        return True

    if "inner_frame" in c and ("room" in c or "subroom" in c):
        return True
    if "room" in c and "inner" in c and ("frame" in c or "boundary" in c or "outline" in c):
        return True
    if "room" in l and "inner" in l and ("frame" in l or "boundary" in l or "outline" in l):
        return True

    jp_keywords = ("内枠", "室内枠", "部屋内枠", "主室内枠", "メイン内枠")
    if any(k in label_raw for k in jp_keywords):
        return True

    return False


def _extract_room_inner_frame_objects(spatial_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    objects = spatial_json.get("furniture_objects")
    if not isinstance(objects, list):
        return []

    out: List[Dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        category = str(obj.get("category") or "")
        label = str(obj.get("label") or "")
        if not _is_room_inner_frame_category(category, label):
            continue
        out.append(obj)
    return out


def _build_main_room_inner_boundary_hint(
    step1_json: Dict[str, Any],
    room_inner_frame_objects: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    try:
        area_x = float(step1_json.get("area_size_X"))
        area_y = float(step1_json.get("area_size_Y"))
    except (TypeError, ValueError):
        return None

    if area_x <= 0.0 or area_y <= 0.0:
        return None

    def _non_arch_anchor_points() -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        objs = step1_json.get("objects")
        if not isinstance(objs, list):
            return pts
        for o in objs:
            if not isinstance(o, dict):
                continue
            cat = str(o.get("category") or "").strip().lower()
            if cat in {"", "floor", "wall", "door", "window"}:
                continue
            try:
                cx = float(o.get("cx"))
                cy = float(o.get("cy"))
            except (TypeError, ValueError):
                continue
            if 0.0 <= cx <= area_x and 0.0 <= cy <= area_y:
                pts.append((cx, cy))
        return pts

    candidates: List[Dict[str, Any]] = []

    for obj in room_inner_frame_objects:
        if not isinstance(obj, dict):
            continue
        category = str(obj.get("category") or "").strip().lower().replace("-", "_")
        label = str(obj.get("label") or "").strip().lower()

        box_norm = obj.get("box_2d_norm")
        if not isinstance(box_norm, list) or len(box_norm) < 4:
            continue

        try:
            nymin = float(box_norm[0])
            nxmin = float(box_norm[1])
            nymax = float(box_norm[2])
            nxmax = float(box_norm[3])
        except (TypeError, ValueError):
            continue

        xmin_w = max(0.0, min(1.0, nxmin)) * area_x
        xmax_w = max(0.0, min(1.0, nxmax)) * area_x
        ymax_w = (1.0 - max(0.0, min(1.0, nymin))) * area_y
        ymin_w = (1.0 - max(0.0, min(1.0, nymax))) * area_y

        if xmax_w < xmin_w:
            xmin_w, xmax_w = xmax_w, xmin_w
        if ymax_w < ymin_w:
            ymin_w, ymax_w = ymax_w, ymin_w

        w = xmax_w - xmin_w
        h = ymax_w - ymin_w
        if w <= 0.0 or h <= 0.0:
            continue

        candidates.append(
            {
                "source_object_id": str(obj.get("id") or ""),
                "is_subroom": ("subroom" in category or "sub room" in label),
                "xmin": xmin_w,
                "ymin": ymin_w,
                "xmax": xmax_w,
                "ymax": ymax_w,
                "w": w,
                "h": h,
                "area": w * h,
            }
        )

    if not candidates:
        return None

    main_candidates = [c for c in candidates if not bool(c.get("is_subroom"))]
    primary = max(main_candidates or candidates, key=lambda c: float(c.get("area") or 0.0))

    def _contains(cand: Dict[str, Any], p: Tuple[float, float]) -> bool:
        return (
            float(cand["xmin"]) <= p[0] <= float(cand["xmax"])
            and float(cand["ymin"]) <= p[1] <= float(cand["ymax"])
        )

    anchors = _non_arch_anchor_points()
    inside_count = sum(1 for p in anchors if _contains(primary, p))
    inside_ratio = (inside_count / len(anchors)) if anchors else 1.0
    coverage_x = float(primary["w"]) / area_x
    coverage_y = float(primary["h"]) / area_y

    is_bad_main = (
        coverage_x < 0.70
        or coverage_y < 0.70
        or inside_ratio < 0.55
    )

    selected = primary
    selection_mode = "primary_main"
    if is_bad_main and len(candidates) >= 2:
        env_xmin = min(float(c["xmin"]) for c in candidates)
        env_ymin = min(float(c["ymin"]) for c in candidates)
        env_xmax = max(float(c["xmax"]) for c in candidates)
        env_ymax = max(float(c["ymax"]) for c in candidates)
        env_w = max(0.0, env_xmax - env_xmin)
        env_h = max(0.0, env_ymax - env_ymin)
        if env_w > 0.05 and env_h > 0.05:
            selected = {
                "source_object_id": "room_inner_frame_envelope",
                "xmin": env_xmin,
                "ymin": env_ymin,
                "xmax": env_xmax,
                "ymax": env_ymax,
                "w": env_w,
                "h": env_h,
                "area": env_w * env_h,
            }
            selection_mode = "envelope_fallback"

    polygon = [
        {"X": round(float(selected["xmin"]), 3), "Y": round(float(selected["ymin"]), 3)},
        {"X": round(float(selected["xmax"]), 3), "Y": round(float(selected["ymin"]), 3)},
        {"X": round(float(selected["xmax"]), 3), "Y": round(float(selected["ymax"]), 3)},
        {"X": round(float(selected["xmin"]), 3), "Y": round(float(selected["ymax"]), 3)},
    ]

    return {
        "source_object_id": str(selected.get("source_object_id") or ""),
        "box_world": [
            round(float(selected["xmin"]), 3),
            round(float(selected["ymin"]), 3),
            round(float(selected["xmax"]), 3),
            round(float(selected["ymax"]), 3),
        ],
        "width_world": round(float(selected["w"]), 3),
        "height_world": round(float(selected["h"]), 3),
        "polygon": polygon,
        "selection_mode": selection_mode,
        "quality": {
            "primary_coverage_x": round(float(coverage_x), 4),
            "primary_coverage_y": round(float(coverage_y), 4),
            "primary_anchor_inside_ratio": round(float(inside_ratio), 4),
            "anchor_count": int(len(anchors)),
        },
    }


def _inner_frame_quality(
    main_room_inner_boundary_hint: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    quality = (
        main_room_inner_boundary_hint.get("quality")
        if isinstance(main_room_inner_boundary_hint, dict)
        else {}
    )
    if not isinstance(quality, dict):
        quality = {}
    return {
        "coverage_x": float(quality.get("primary_coverage_x", 0.0) or 0.0),
        "coverage_y": float(quality.get("primary_coverage_y", 0.0) or 0.0),
        "anchor_inside_ratio": float(quality.get("primary_anchor_inside_ratio", 0.0) or 0.0),
    }


def _needs_room_inner_frame_retry(
    main_room_inner_boundary_hint: Optional[Dict[str, Any]],
    *,
    min_coverage_x: float,
    min_coverage_y: float,
    min_anchor_inside_ratio: float,
) -> Tuple[bool, Dict[str, float]]:
    q = _inner_frame_quality(main_room_inner_boundary_hint)
    need = (
        q["coverage_x"] < float(min_coverage_x)
        or q["coverage_y"] < float(min_coverage_y)
        or q["anchor_inside_ratio"] < float(min_anchor_inside_ratio)
    )
    return need, q


def _is_opening_category(category: str, label: str) -> bool:
    c = str(category or "").strip().lower().replace("-", "_")
    l = str(label or "").strip().lower()
    label_raw = str(label or "")

    deny_en = {
        "storage",
        "cabinet",
        "closet",
        "cupboard",
        "wardrobe",
        "furniture",
    }
    deny_jp = ("収納", "キャビネット", "クローゼット", "戸棚", "家具")
    if any(k in c for k in deny_en) or any(k in l for k in deny_en):
        return False
    if any(k in label_raw for k in deny_jp):
        return False

    explicit_category = {
        "door",
        "sliding_door",
        "window",
        "entrance_door",
        "front_door",
        "main_door",
        "double_door",
    }
    if c in explicit_category:
        return True

    explicit_label = {
        "door",
        "sliding door",
        "sliding_door",
        "window",
        "entrance door",
        "front door",
        "main door",
        "double door",
    }
    if l in explicit_label:
        return True

    if label_raw in {"ドア", "引き戸", "窓"}:
        return True

    tokens = [t for t in re.split(r"[^a-z0-9]+", c) if t]
    if "window" in tokens:
        return True
    if "door" in tokens:
        if any(k in tokens for k in ("storage", "cabinet", "closet", "cupboard", "wardrobe")):
            return False
        if ("sliding" in tokens) or ("entrance" in tokens) or ("front" in tokens) or ("main" in tokens) or (len(tokens) == 1):
            return True

    return False


def _extract_opening_objects(spatial_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    objects = spatial_json.get("furniture_objects")
    if not isinstance(objects, list):
        return []

    out: List[Dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        category = str(obj.get("category") or "")
        label = str(obj.get("label") or "")
        if not _is_opening_category(category, label):
            continue
        out.append(obj)
    return out


def _extract_area_size_xy(step1_json: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    try:
        area_x = float(step1_json.get("area_size_X"))
        area_y = float(step1_json.get("area_size_Y"))
    except (TypeError, ValueError):
        return None
    if area_x <= 0.0 or area_y <= 0.0:
        return None
    return area_x, area_y


def _box_norm_to_world(box_norm: List[Any], area_x: float, area_y: float) -> Optional[Dict[str, float]]:
    if not isinstance(box_norm, list) or len(box_norm) < 4:
        return None
    try:
        nymin = float(box_norm[0])
        nxmin = float(box_norm[1])
        nymax = float(box_norm[2])
        nxmax = float(box_norm[3])
    except (TypeError, ValueError):
        return None

    xmin = max(0.0, min(1.0, nxmin)) * area_x
    xmax = max(0.0, min(1.0, nxmax)) * area_x
    ymax = (1.0 - max(0.0, min(1.0, nymin))) * area_y
    ymin = (1.0 - max(0.0, min(1.0, nymax))) * area_y
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    dx = xmax - xmin
    dy = ymax - ymin
    if dx <= 0.0 or dy <= 0.0:
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


def _opening_wall_role_step1(opening: Dict[str, Any]) -> str:
    room_ids_raw = opening.get("room_ids")
    if isinstance(room_ids_raw, list):
        room_ids = [str(r).strip().lower() for r in room_ids_raw if isinstance(r, (str, int, float))]
        if any(r in {"outside", "outdoor", "exterior"} for r in room_ids):
            return "outer"
        uniq = {r for r in room_ids if r and r not in {"outside", "outdoor", "exterior"}}
        if len(uniq) >= 2:
            return "interior"

    wall = str(opening.get("wall") or "").strip().lower()
    if any(tok in wall for tok in ("outer", "outside", "exterior", "north", "south", "east", "west")):
        return "outer"
    if any(tok in wall for tok in ("inner", "interior", "partition", "shared", "_left_wall", "_right_wall", "_top_wall", "_bottom_wall")):
        return "interior"
    return "unknown"


def _opening_orientation_step1(opening: Dict[str, Any], area_x: float, area_y: float) -> str:
    wall = str(opening.get("wall") or "").strip().lower()
    if "x" in wall and "wall" in wall:
        return "vertical"
    if "y" in wall and "wall" in wall:
        return "horizontal"
    if any(tok in wall for tok in ("east", "west", "_left_wall", "_right_wall")):
        return "vertical"
    if any(tok in wall for tok in ("north", "south", "_top_wall", "_bottom_wall")):
        return "horizontal"

    cx = float(opening.get("cx") or 0.0)
    cy = float(opening.get("cy") or 0.0)
    d_left = abs(cx - 0.0)
    d_right = abs(cx - area_x)
    d_bottom = abs(cy - 0.0)
    d_top = abs(cy - area_y)
    return "vertical" if min(d_left, d_right) <= min(d_bottom, d_top) else "horizontal"


def _evaluate_openings_quality_for_retry(
    step1_json: Dict[str, Any],
    opening_objects: List[Dict[str, Any]],
    *,
    min_outer_door_width_ratio: float,
    max_outer_center_dist_m: float,
) -> Dict[str, Any]:
    area = _extract_area_size_xy(step1_json)
    if area is None:
        return {"issue_count": 0, "score": 0.0, "issues": [], "evaluated_outer_doors": 0}
    area_x, area_y = area

    outer_band = max(0.25, 0.08 * min(max(area_x, 0.1), max(area_y, 0.1)))
    candidates: List[Dict[str, Any]] = []
    for idx, obj in enumerate(opening_objects):
        if not isinstance(obj, dict):
            continue
        cat = str(obj.get("category") or obj.get("label") or "").strip().lower().replace("-", "_")
        kind = "window" if "window" in cat else ("door" if "door" in cat else "")
        if not kind:
            continue
        bw = _box_norm_to_world(obj.get("box_2d_norm"), area_x, area_y)
        if bw is None:
            continue
        orient = "vertical" if bw["dy"] >= bw["dx"] else "horizontal"
        width = bw["dy"] if orient == "vertical" else bw["dx"]
        dist_outer = min(abs(bw["cx"] - 0.0), abs(area_x - bw["cx"]), abs(bw["cy"] - 0.0), abs(area_y - bw["cy"]))
        candidates.append(
            {
                "uid": f"cand_{idx}",
                "kind": kind,
                "orientation": orient,
                "width": width,
                "cx": bw["cx"],
                "cy": bw["cy"],
                "is_outer": dist_outer <= outer_band,
            }
        )

    openings = step1_json.get("openings")
    if not isinstance(openings, list):
        return {"issue_count": 0, "score": 0.0, "issues": [], "evaluated_outer_doors": 0}

    used: set[str] = set()
    issues: List[Dict[str, Any]] = []
    total_score = 0.0
    evaluated = 0

    for op in openings:
        if not isinstance(op, dict):
            continue
        if str(op.get("type") or "").strip().lower() != "door":
            continue
        if _opening_wall_role_step1(op) != "outer":
            continue
        evaluated += 1

        orient = _opening_orientation_step1(op, area_x, area_y)
        op_cx = float(op.get("cx") or 0.0)
        op_cy = float(op.get("cy") or 0.0)
        op_w = max(0.05, abs(float(op.get("w") or 0.0)))

        def _score(c: Dict[str, Any]) -> float:
            if orient == "vertical":
                d_along = abs(c["cy"] - op_cy)
                d_cross = abs(c["cx"] - op_cx)
            else:
                d_along = abs(c["cx"] - op_cx)
                d_cross = abs(c["cy"] - op_cy)
            orient_pen = 0.0 if c["orientation"] == orient else 1.2
            outer_pen = 0.0 if c["is_outer"] else 1.6
            return d_along + (2.5 * d_cross) + (0.35 * abs(c["width"] - op_w)) + orient_pen + outer_pen

        pool = [c for c in candidates if c["kind"] == "door" and c["uid"] not in used]
        if not pool:
            issues.append({"opening_id": op.get("opening_id"), "reason": "missing_candidate"})
            total_score += 10.0
            continue

        best = min(pool, key=_score)
        used.add(best["uid"])

        center_dist = ((best["cx"] - op_cx) ** 2 + (best["cy"] - op_cy) ** 2) ** 0.5
        ratio = best["width"] / op_w if op_w > 1e-9 else 1.0
        total_score += _score(best)

        if ratio < float(min_outer_door_width_ratio) or center_dist > float(max_outer_center_dist_m):
            issues.append(
                {
                    "opening_id": op.get("opening_id"),
                    "reason": "outer_door_mismatch",
                    "width_ratio": round(float(ratio), 4),
                    "center_dist_m": round(float(center_dist), 4),
                }
            )

    return {
        "issue_count": int(len(issues)),
        "score": round(float(total_score), 6),
        "issues": issues,
        "evaluated_outer_doors": int(evaluated),
    }


def _is_openings_eval_better(new_eval: Dict[str, Any], old_eval: Dict[str, Any]) -> bool:
    new_issues = int(new_eval.get("issue_count", 0))
    old_issues = int(old_eval.get("issue_count", 0))
    if new_issues != old_issues:
        return new_issues < old_issues
    return float(new_eval.get("score", 0.0)) < float(old_eval.get("score", 0.0))


def _run_gemini_spatial(
    image_path: pathlib.Path,
    out_dir: pathlib.Path,
    output_stem: str,
    gemini_api_key: Optional[str],
    gemini_model: str,
    gemini_task: str,
    gemini_target_prompt: str,
    gemini_label_language: str,
    gemini_temperature: float,
    gemini_thinking_budget: int,
    gemini_max_items: int,
    gemini_resize_max: int,
    gemini_prompt_text: Optional[str],
    gemini_include_non_furniture: bool,
) -> Dict[str, Any]:
    script_path = pathlib.Path(__file__).resolve().parent / "spatial_understanding_google.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Gemini spatial script not found: {script_path}")

    spatial_out_json = out_dir / f"{output_stem}.json"
    cmd = [
        sys.executable,
        str(script_path),
        "--image_path",
        str(image_path),
        "--out_json",
        str(spatial_out_json),
        "--task",
        gemini_task,
        "--model",
        gemini_model,
        "--target_prompt",
        gemini_target_prompt,
        "--label_language",
        gemini_label_language,
        "--temperature",
        str(gemini_temperature),
        "--thinking_budget",
        str(gemini_thinking_budget),
        "--max_items",
        str(gemini_max_items),
        "--resize_max",
        str(gemini_resize_max),
    ]

    if gemini_prompt_text:
        cmd.extend(["--prompt_text", gemini_prompt_text])
    if gemini_include_non_furniture:
        cmd.append("--include_non_furniture")

    key = gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Gemini enabled but GOOGLE_API_KEY/GEMINI_API_KEY is not set (or --gemini_api_key missing).")

    env = os.environ.copy()
    env["GOOGLE_API_KEY"] = key
    subprocess.run(cmd, check=True, env=env)

    manifest_path = spatial_out_json.with_name(f"{spatial_out_json.stem}_manifest.json")
    raw_response_path = spatial_out_json.with_name(f"{spatial_out_json.stem}_raw_response.json")
    plot_path = spatial_out_json.with_name(f"{spatial_out_json.stem}_plot.png")

    spatial_json = _load_json_file(spatial_out_json)
    manifest_json: Dict[str, Any] = _load_json_file(manifest_path) if manifest_path.exists() else {}

    return {
        "spatial_json": spatial_json,
        "manifest_json": manifest_json,
        "paths": {
            "spatial_json": str(spatial_out_json),
            "manifest_json": str(manifest_path),
            "raw_response_json": str(raw_response_path),
            "plot_png": str(plot_path),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate layout JSON (Step1 OpenAI + Step2 Spatial Understanding + Step3 rule-based)"
    )
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--dimensions_path", required=True)
    parser.add_argument("--prompt1_path", default=None)
    parser.add_argument("--analysis_input", default=None, help="Optional existing Step1 output text/JSON path")
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_dir", default=None)

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning_effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--text_verbosity", default=DEFAULT_TEXT_VERBOSITY)
    parser.add_argument("--max_output_tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--image_detail", default=DEFAULT_IMAGE_DETAIL)
    parser.add_argument("--api_key", default=None)

    parser.add_argument("--enable_gemini_spatial", action="store_true")
    parser.add_argument("--gemini_api_key", default=None)
    parser.add_argument("--gemini_model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--gemini_task", default=DEFAULT_GEMINI_TASK, choices=["boxes", "masks"])
    parser.add_argument("--gemini_target_prompt", default=None)
    parser.add_argument("--gemini_prompt_text", default=None)
    parser.add_argument("--gemini_prompt_text_path", default=None)
    parser.add_argument("--gemini_label_language", default=DEFAULT_GEMINI_LABEL_LANGUAGE)
    parser.add_argument("--gemini_temperature", type=float, default=DEFAULT_GEMINI_TEMPERATURE)
    parser.add_argument("--gemini_thinking_budget", type=int, default=DEFAULT_GEMINI_THINKING_BUDGET)
    parser.add_argument("--gemini_max_items", type=int, default=DEFAULT_GEMINI_MAX_ITEMS)
    parser.add_argument("--gemini_resize_max", type=int, default=DEFAULT_GEMINI_RESIZE_MAX)
    parser.add_argument("--gemini_include_non_furniture", action="store_true")
    parser.add_argument("--gemini_openings_prompt_text", default=None)
    parser.add_argument("--gemini_openings_prompt_text_path", default=None)
    parser.add_argument("--gemini_room_inner_frame_prompt_text_path", default=None)
    parser.add_argument("--gemini_openings_retry_max_retries", type=int, default=1)
    parser.add_argument("--gemini_openings_retry_temperature", type=float, default=-1.0)
    parser.add_argument("--gemini_openings_retry_min_outer_door_width_ratio", type=float, default=0.72)
    parser.add_argument("--gemini_openings_retry_max_outer_center_dist_m", type=float, default=0.85)
    parser.add_argument("--gemini_inner_frame_retry_max_retries", type=int, default=1)
    parser.add_argument("--gemini_inner_frame_retry_min_coverage_x", type=float, default=0.88)
    parser.add_argument("--gemini_inner_frame_retry_min_coverage_y", type=float, default=0.88)
    parser.add_argument("--gemini_inner_frame_retry_min_anchor_inside_ratio", type=float, default=0.55)
    parser.add_argument("--gemini_inner_frame_retry_temperature", type=float, default=-1.0)
    return parser


def main() -> None:
    load_dotenv()
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    image_path = pathlib.Path(args.image_path)
    dimensions_path = pathlib.Path(args.dimensions_path)
    prompt1_path = (
        pathlib.Path(args.prompt1_path)
        if args.prompt1_path
        else (repo_root / "prompts" / "fixed_mode_20260222" / "step1_openai_prompt.txt")
    )
    out_json = pathlib.Path(args.out_json)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else out_json.parent

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not dimensions_path.exists():
        raise FileNotFoundError(f"dimensions file not found: {dimensions_path}")
    if not prompt1_path.exists():
        raise FileNotFoundError(f"prompt1 not found: {prompt1_path}")

    dimensions_text = read_text(dimensions_path)
    prompt1_text = read_text(prompt1_path)
    gemini_prompt_text = args.gemini_prompt_text
    if args.gemini_prompt_text_path:
        gemini_prompt_text_path = pathlib.Path(args.gemini_prompt_text_path)
        if not gemini_prompt_text_path.exists():
            raise FileNotFoundError(f"gemini prompt text not found: {gemini_prompt_text_path}")
        gemini_prompt_text = read_text(gemini_prompt_text_path)

    gemini_room_inner_frame_prompt_text: Optional[str] = None
    if args.gemini_room_inner_frame_prompt_text_path:
        gemini_room_inner_frame_prompt_text_path = pathlib.Path(args.gemini_room_inner_frame_prompt_text_path)
        if not gemini_room_inner_frame_prompt_text_path.exists():
            raise FileNotFoundError(
                f"gemini room inner frame prompt text not found: {gemini_room_inner_frame_prompt_text_path}"
            )
        gemini_room_inner_frame_prompt_text = read_text(gemini_room_inner_frame_prompt_text_path)

    gemini_openings_prompt_text = args.gemini_openings_prompt_text
    if args.gemini_openings_prompt_text_path:
        gemini_openings_prompt_text_path = pathlib.Path(args.gemini_openings_prompt_text_path)
        if not gemini_openings_prompt_text_path.exists():
            raise FileNotFoundError(f"gemini openings prompt text not found: {gemini_openings_prompt_text_path}")
        gemini_openings_prompt_text = read_text(gemini_openings_prompt_text_path)

    image_base64, image_mime = _encode_image_base64(image_path)
    step1_provider = "openai"

    openai_needed = (not args.analysis_input)
    client: Optional[OpenAI] = None
    if openai_needed:
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Use env var or --api_key.")
        client = OpenAI(api_key=api_key)

    t0 = time.perf_counter()
    step1_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    step1_model = "cached"

    if args.analysis_input:
        step1_text = read_text(pathlib.Path(args.analysis_input))
    else:
        step1_prompt = f"{prompt1_text}\n\nLAYOUT_HINTS:\n{dimensions_text}"
        if client is None:
            raise RuntimeError("OpenAI client is not initialized for step1.")
        step1_text, step1_usage, step1_model = _call_responses(
            client=client,
            model=args.model,
            prompt_text=step1_prompt,
            image_base64=image_base64,
            image_mime=image_mime,
            image_detail=args.image_detail,
            reasoning_effort=args.reasoning_effort,
            text_verbosity=args.text_verbosity,
            max_output_tokens=args.max_output_tokens,
        )

    step1_json = extract_json_payload(step1_text)
    stage1_raw_path = out_dir / "stage1_output_raw.json"
    stage1_parsed_path = out_dir / "stage1_output_parsed.json"
    step1_raw_payload = {"text": step1_text, "provider": step1_provider}
    write_json(stage1_raw_path, step1_raw_payload)
    write_json(stage1_parsed_path, step1_json)

    gemini_details: Dict[str, Any] = {"enabled": bool(args.enable_gemini_spatial)}
    gemini_spatial_json: Optional[Dict[str, Any]] = None

    if args.enable_gemini_spatial:
        step1_category_counts = _collect_step1_category_counts(step1_json)
        gemini_target_prompt = args.gemini_target_prompt or _build_default_gemini_target_prompt(step1_category_counts)
        furniture_prompt_text = _build_gemini_furniture_prompt(
            gemini_prompt_text,
            dimensions_text,
            step1_category_counts=step1_category_counts,
        )
        inner_frame_target_prompt, inner_frame_constraints = _build_room_inner_frame_constraints(dimensions_text)
        inner_frame_prompt_text = gemini_room_inner_frame_prompt_text or DEFAULT_GEMINI_ROOM_INNER_FRAME_PROMPT
        if isinstance(inner_frame_constraints, str) and inner_frame_constraints.strip():
            inner_frame_prompt_text = f"{inner_frame_prompt_text}\n\nROOM_CONSTRAINTS:\n{inner_frame_constraints}"

        gemini_result = _run_gemini_spatial(
            image_path=image_path,
            out_dir=out_dir,
            output_stem="gemini_spatial_output",
            gemini_api_key=args.gemini_api_key,
            gemini_model=args.gemini_model,
            gemini_task=args.gemini_task,
            gemini_target_prompt=gemini_target_prompt,
            gemini_label_language=args.gemini_label_language,
            gemini_temperature=args.gemini_temperature,
            gemini_thinking_budget=args.gemini_thinking_budget,
            gemini_max_items=args.gemini_max_items,
            gemini_resize_max=args.gemini_resize_max,
            gemini_prompt_text=furniture_prompt_text,
            gemini_include_non_furniture=bool(args.gemini_include_non_furniture),
        )
        gemini_spatial_json = gemini_result["spatial_json"]
        room_inner_frame_result = _run_gemini_spatial(
            image_path=image_path,
            out_dir=out_dir,
            output_stem="gemini_room_inner_frame_output",
            gemini_api_key=args.gemini_api_key,
            gemini_model=args.gemini_model,
            gemini_task="masks",
            gemini_target_prompt=inner_frame_target_prompt,
            gemini_label_language="English",
            gemini_temperature=args.gemini_temperature,
            gemini_thinking_budget=args.gemini_thinking_budget,
            gemini_max_items=args.gemini_max_items,
            gemini_resize_max=args.gemini_resize_max,
            gemini_prompt_text=inner_frame_prompt_text,
            gemini_include_non_furniture=True,
        )
        room_inner_frame_json = room_inner_frame_result["spatial_json"]
        room_inner_frame_objects = _extract_room_inner_frame_objects(room_inner_frame_json)
        main_room_inner_boundary_hint = _build_main_room_inner_boundary_hint(
            step1_json=step1_json,
            room_inner_frame_objects=room_inner_frame_objects,
        )
        inner_frame_retry_info: Dict[str, Any] = {
            "enabled": int(args.gemini_inner_frame_retry_max_retries) > 0,
            "max_retries": int(max(0, int(args.gemini_inner_frame_retry_max_retries))),
            "attempts": 1,
            "retried": False,
            "final_quality": _inner_frame_quality(main_room_inner_boundary_hint),
        }
        retries = int(max(0, int(args.gemini_inner_frame_retry_max_retries)))
        while True:
            needs_retry, quality_now = _needs_room_inner_frame_retry(
                main_room_inner_boundary_hint,
                min_coverage_x=float(args.gemini_inner_frame_retry_min_coverage_x),
                min_coverage_y=float(args.gemini_inner_frame_retry_min_coverage_y),
                min_anchor_inside_ratio=float(args.gemini_inner_frame_retry_min_anchor_inside_ratio),
            )
            inner_frame_retry_info["final_quality"] = quality_now
            if (not needs_retry) or (inner_frame_retry_info["attempts"] > retries):
                break

            retry_temp = float(args.gemini_inner_frame_retry_temperature)
            if retry_temp < 0.0:
                retry_temp = min(float(args.gemini_temperature), 0.45)

            retry_prompt = (
                f"{inner_frame_prompt_text}\n\n"
                "RETRY_INSTRUCTION:\n"
                "- Previous room_inner_frame coverage was too small relative to dimensions.\n"
                f"- Require room_inner_frame coverage_x >= {float(args.gemini_inner_frame_retry_min_coverage_x):.3f} and "
                f"coverage_y >= {float(args.gemini_inner_frame_retry_min_coverage_y):.3f} whenever geometry allows.\n"
                "- Keep rectangle tightly on inner wall lines and maximize extent without crossing walls.\n"
                "- Do not shrink due to text labels."
            )

            room_inner_frame_result = _run_gemini_spatial(
                image_path=image_path,
                out_dir=out_dir,
                output_stem="gemini_room_inner_frame_output",
                gemini_api_key=args.gemini_api_key,
                gemini_model=args.gemini_model,
                gemini_task="masks",
                gemini_target_prompt=inner_frame_target_prompt,
                gemini_label_language="English",
                gemini_temperature=retry_temp,
                gemini_thinking_budget=args.gemini_thinking_budget,
                gemini_max_items=args.gemini_max_items,
                gemini_resize_max=args.gemini_resize_max,
                gemini_prompt_text=retry_prompt,
                gemini_include_non_furniture=True,
            )
            room_inner_frame_json = room_inner_frame_result["spatial_json"]
            room_inner_frame_objects = _extract_room_inner_frame_objects(room_inner_frame_json)
            main_room_inner_boundary_hint = _build_main_room_inner_boundary_hint(
                step1_json=step1_json,
                room_inner_frame_objects=room_inner_frame_objects,
            )
            inner_frame_retry_info["retried"] = True
            inner_frame_retry_info["attempts"] = int(inner_frame_retry_info["attempts"]) + 1
            inner_frame_retry_info["retry_temperature"] = retry_temp
            inner_frame_retry_info["last_trigger_quality"] = quality_now

        opening_objects: List[Dict[str, Any]] = []
        # Spatial ON means full 3-pass: furniture + room inner frame + openings.
        openings_enabled = True
        openings_details: Dict[str, Any] = {
            "enabled": openings_enabled,
            "control": "follow_enable_gemini_spatial",
            "note": "openings are always included when step2_spatial_understanding is enabled",
        }

        if openings_enabled:
            openings_prompt = gemini_openings_prompt_text or DEFAULT_GEMINI_OPENINGS_PROMPT
            openings_result = _run_gemini_spatial(
                image_path=image_path,
                out_dir=out_dir,
                output_stem="gemini_openings_output",
                gemini_api_key=args.gemini_api_key,
                gemini_model=args.gemini_model,
                gemini_task="boxes",
                gemini_target_prompt="doors, sliding doors, windows",
                gemini_label_language="English",
                gemini_temperature=args.gemini_temperature,
                gemini_thinking_budget=args.gemini_thinking_budget,
                gemini_max_items=args.gemini_max_items,
                gemini_resize_max=args.gemini_resize_max,
                gemini_prompt_text=openings_prompt,
                gemini_include_non_furniture=True,
            )
            opening_json = openings_result["spatial_json"]
            opening_objects = _extract_opening_objects(opening_json)
            openings_retry_info: Dict[str, Any] = {
                "enabled": int(args.gemini_openings_retry_max_retries) > 0,
                "max_retries": int(max(0, int(args.gemini_openings_retry_max_retries))),
                "attempts": 1,
                "retried": False,
            }
            best_openings_result = openings_result
            best_opening_objects = opening_objects
            best_eval = _evaluate_openings_quality_for_retry(
                step1_json=step1_json,
                opening_objects=best_opening_objects,
                min_outer_door_width_ratio=float(args.gemini_openings_retry_min_outer_door_width_ratio),
                max_outer_center_dist_m=float(args.gemini_openings_retry_max_outer_center_dist_m),
            )
            openings_retry_info["first_eval"] = best_eval

            max_retries = int(max(0, int(args.gemini_openings_retry_max_retries)))
            retry_idx = 0
            while retry_idx < max_retries and int(best_eval.get("issue_count", 0)) > 0:
                retry_idx += 1
                retry_temp = float(args.gemini_openings_retry_temperature)
                if retry_temp < 0.0:
                    retry_temp = min(float(args.gemini_temperature), 0.45)

                retry_prompt = (
                    f"{openings_prompt}\n\n"
                    "RETRY_INSTRUCTION:\n"
                    "- Previous result likely truncated one or more outer sliding-door openings.\n"
                    "- For each labeled door/sliding door, box the FULL wall-break opening span between jamb endpoints.\n"
                    "- Do not keep only a short center segment of the opening.\n"
                    "- Keep strict exclusion of pocket/storage/rail regions outside the opening gap.\n"
                )
                retry_result = _run_gemini_spatial(
                    image_path=image_path,
                    out_dir=out_dir,
                    output_stem="gemini_openings_output",
                    gemini_api_key=args.gemini_api_key,
                    gemini_model=args.gemini_model,
                    gemini_task="boxes",
                    gemini_target_prompt="doors, sliding doors, windows",
                    gemini_label_language="English",
                    gemini_temperature=retry_temp,
                    gemini_thinking_budget=args.gemini_thinking_budget,
                    gemini_max_items=args.gemini_max_items,
                    gemini_resize_max=args.gemini_resize_max,
                    gemini_prompt_text=retry_prompt,
                    gemini_include_non_furniture=True,
                )
                retry_objects = _extract_opening_objects(retry_result["spatial_json"])
                retry_eval = _evaluate_openings_quality_for_retry(
                    step1_json=step1_json,
                    opening_objects=retry_objects,
                    min_outer_door_width_ratio=float(args.gemini_openings_retry_min_outer_door_width_ratio),
                    max_outer_center_dist_m=float(args.gemini_openings_retry_max_outer_center_dist_m),
                )

                openings_retry_info["retried"] = True
                openings_retry_info["attempts"] = 1 + retry_idx
                openings_retry_info["last_retry_eval"] = retry_eval
                openings_retry_info["retry_temperature"] = retry_temp

                if _is_openings_eval_better(retry_eval, best_eval):
                    best_eval = retry_eval
                    best_openings_result = retry_result
                    best_opening_objects = retry_objects

            openings_result = best_openings_result
            opening_objects = best_opening_objects
            openings_details = {
                "enabled": True,
                "task": "boxes",
                "target_prompt": "doors, sliding doors, windows",
                "label_language": "English",
                "temperature": float(args.gemini_temperature),
                "thinking_budget": int(args.gemini_thinking_budget),
                "max_items": int(args.gemini_max_items),
                "resize_max": int(args.gemini_resize_max),
                "prompt_text": openings_prompt,
                "outputs": openings_result["paths"],
                "manifest": openings_result["manifest_json"],
                "object_count": len(opening_objects),
                "retry": openings_retry_info,
                "quality_eval": best_eval,
            }

        gemini_details = {
            "enabled": True,
            "model": args.gemini_model,
            "task": args.gemini_task,
            "target_prompt": gemini_target_prompt,
            "label_language": args.gemini_label_language,
            "temperature": float(args.gemini_temperature),
            "thinking_budget": int(args.gemini_thinking_budget),
            "max_items": int(args.gemini_max_items),
            "resize_max": int(args.gemini_resize_max),
            "include_non_furniture": bool(args.gemini_include_non_furniture),
            "step1_category_counts": step1_category_counts,
            "outputs": gemini_result["paths"],
            "manifest": gemini_result["manifest_json"],
            "room_inner_frame": {
                "task": "masks",
                "target_prompt": inner_frame_target_prompt,
                "label_language": "English",
                "temperature": float(args.gemini_temperature),
                "thinking_budget": int(args.gemini_thinking_budget),
                "max_items": int(args.gemini_max_items),
                "resize_max": int(args.gemini_resize_max),
                "prompt_text": inner_frame_prompt_text,
                "outputs": room_inner_frame_result["paths"],
                "manifest": room_inner_frame_result["manifest_json"],
                "object_count": len(room_inner_frame_objects),
                "main_room_inner_boundary_hint": main_room_inner_boundary_hint,
                "retry": inner_frame_retry_info,
            },
            "openings": openings_details,
        }
    else:
        room_inner_frame_objects = []
        main_room_inner_boundary_hint = None
        opening_objects = []

    step3_raw_path = out_dir / "stage3_output_raw.json"
    final_json = build_layout_rule_based(
        step1_json=step1_json,
        gemini_spatial_json=gemini_spatial_json,
        room_inner_frame_objects=room_inner_frame_objects,
        opening_objects=opening_objects,
        main_room_inner_boundary_hint=main_room_inner_boundary_hint,
    )
    step3_text = json.dumps(final_json, ensure_ascii=False, indent=2)
    step3_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    step3_model = "rule_based_step3_v1"
    step3_payload = {"mode": "rule", "text": step3_text}
    write_json(step3_raw_path, step3_payload)

    write_json(out_json, final_json)

    runtime_sec = time.perf_counter() - t0
    run_manifest = {
        "runtime_sec": runtime_sec,
        "model_requested": {"step1_openai": args.model, "step2_spatial_understanding": args.gemini_model},
        "model_used": {
            "step1_openai": step1_model,
            "step2_spatial_understanding": args.gemini_model if bool(args.enable_gemini_spatial) else "disabled",
            "step3_rule_based": step3_model,
        },
        "pipeline_stages": {
            "step1": "openai",
            "step2": "spatial_understanding",
            "step3": "rule_based",
        },
        "stage_provider": {
            "step1_openai": "cached" if args.analysis_input else step1_provider,
            "step2_spatial_understanding": ("gemini" if bool(args.enable_gemini_spatial) else "disabled"),
            "step3_rule_based": "rule",
        },
        "step3_mode": "rule",
        "gemini_spatial": gemini_details,
        "inputs": {
            "image_path": str(image_path),
            "dimensions_path": str(dimensions_path),
            "prompt1_path": str(prompt1_path),
            "analysis_input": str(args.analysis_input) if args.analysis_input else None,
        },
        "usage": {
            "step1_openai": step1_usage,
            "step3_rule_based": step3_usage,
            "total": {
                "input_tokens": int(step1_usage["input_tokens"] + step3_usage["input_tokens"]),
                "output_tokens": int(step1_usage["output_tokens"] + step3_usage["output_tokens"]),
                "total_tokens": int(step1_usage["total_tokens"] + step3_usage["total_tokens"]),
            },
        },
        "outputs": {
            "layout_json": str(out_json),
            "stage1_raw": str(stage1_raw_path),
            "stage1_parsed": str(stage1_parsed_path),
            "stage2_spatial_furniture_json": gemini_details.get("outputs", {}).get("spatial_json") if gemini_details.get("enabled") else None,
            "stage2_spatial_inner_frame_json": gemini_details.get("room_inner_frame", {}).get("outputs", {}).get("spatial_json") if gemini_details.get("enabled") else None,
            "stage2_spatial_openings_json": gemini_details.get("openings", {}).get("outputs", {}).get("spatial_json") if gemini_details.get("enabled") else None,
            "stage3_raw": str(step3_raw_path),
        },
    }
    write_json(out_dir / "generation_manifest.json", run_manifest)

    print(
        json.dumps(
            {
                "layout_json": str(out_json),
                "generation_manifest": str(out_dir / "generation_manifest.json"),
                "gemini_enabled": bool(args.enable_gemini_spatial),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
