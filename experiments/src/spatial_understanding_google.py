from __future__ import annotations

import argparse
import base64
import io
import json
import os
import pathlib
import re
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw

from layout_tools import write_json

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_TIMEOUT_SEC = 120
DEFAULT_MASK_THRESHOLD = 127

SEGMENTATION_COLORS_HEX = [
    "#E6194B",
    "#3C89D0",
    "#3CB44B",
    "#FFE119",
    "#911EB4",
    "#42D4F4",
    "#F58231",
    "#F032E6",
    "#BFEF45",
    "#469990",
]

JP_CATEGORY_ALIASES = {
    "ベッド": "bed",
    "椅子": "chair",
    "シンク": "sink",
    "トイレ": "toilet",
    "収納": "storage",
    "ストレージ": "storage",
    "テレビキャビネット": "tv_cabinet",
    "テレビ": "tv",
    "ソファ": "sofa",
    "テーブル": "table",
    "窓": "window",
    "ドア": "door",
    "引き戸": "sliding_door",
}


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    s = hex_color.strip().lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _to_category_id(label: str) -> str:
    raw = label.strip()
    if raw in JP_CATEGORY_ALIASES:
        return JP_CATEGORY_ALIASES[raw]
    t = raw.lower()
    t = re.sub(r"[^\w]+", "_", t, flags=re.UNICODE)
    t = re.sub(r"_+", "_", t).strip("_")
    return t or "unknown"


def _get_2d_prompt(target_prompt: str, label_prompt: str, max_items: int) -> str:
    label_text = label_prompt.strip() if label_prompt.strip() else "a text label"
    return (
        "This image is a floor plan. Detect the shapes (outline lines) of the labeled "
        "furniture/equipment and return tight bounding boxes around the outer contour of each item.\n\n"
        f"Target furniture/equipment: {target_prompt}\n"
        f"Return at most {int(max_items)} items as a JSON list. Each entry must contain "
        f"\"box_2d\": [ymin, xmin, ymax, xmax] and \"label\": {label_text}. "
        "box_2d values must be integers normalized to 0-1000.\n\n"
        "Important:\n"
        "Do NOT draw bounding boxes around printed text in the image "
        "(e.g., the words \"Sink\", \"Storage\" themselves).\n"
        "Use the text only to identify which furniture/equipment item it refers to, "
        "but the bounding box must enclose the drawn shape (the actual outline) of the item.\n"
        "Even if the label text overlaps the furniture, ignore the text and box only the outer contour of the shape.\n"
        "Do NOT include walls, windows, doors, or other architectural lines."
    )


def _get_segmentation_prompt(target_prompt: str, label_language: str) -> str:
    suffix = (
        '. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key "box_2d", '
        'the segmentation mask in key "mask", and the text label in the key "label". Use descriptive labels.'
    )

    lang = label_language.strip()
    if lang and lang.lower() != "english":
        suffix = (
            '. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key "box_2d", '
            f'the segmentation mask in key "mask", and the text label in language {lang} in the key "label". '
            f'Use descriptive labels in {lang}. Ensure labels are in {lang}. DO NOT USE ENGLISH FOR LABELS.'
        )

    quality = (
        " Use the physical object silhouette (frame/footprint), not text glyph regions. "
        "Each box_2d should tightly cover the full object extent represented by the mask."
    )
    return f"Give the segmentation masks for {target_prompt}{suffix}{quality}"


def _strip_json_fence(text: str) -> str:
    s = text.strip()
    if "```json" in s:
        try:
            return s.split("```json", 1)[1].split("```", 1)[0].strip()
        except Exception:
            return s
    if s.startswith("```"):
        lines = s.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return s


def _extract_response_text(response: Dict[str, Any]) -> str:
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
    raise RuntimeError(f"No text in candidates. promptFeedback={prompt_feedback}")


def _parse_response_json(text: str) -> Any:
    stripped = _strip_json_fence(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        m = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", stripped)
        if not m:
            raise RuntimeError("Gemini text does not contain parseable JSON.")
        return json.loads(m.group(1))


def _post_json(url: str, api_key: str, payload: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        err = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API HTTP {exc.code}: {err}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

    try:
        out = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini response is not JSON: {raw[:500]}") from exc

    return out


def _resize_for_inference(image_path: pathlib.Path, max_side: int) -> Tuple[Image.Image, str]:
    im = Image.open(image_path).convert("RGB")
    im.thumbnail((int(max_side), int(max_side)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return im, base64.b64encode(buf.getvalue()).decode("ascii")


def _coerce_box_2d(value: Any) -> Optional[List[float]]:
    if not isinstance(value, list):
        return None
    if len(value) < 4:
        return None
    vals: List[float] = []
    for x in value[:4]:
        try:
            vals.append(float(x))
        except (TypeError, ValueError):
            return None
    return vals


def _box_to_norm(box_2d: List[float]) -> List[float]:
    ymin, xmin, ymax, xmax = box_2d
    max_abs = max(abs(ymin), abs(xmin), abs(ymax), abs(xmax))

    if max_abs <= 1.5:
        nymin, nxmin, nymax, nxmax = ymin, xmin, ymax, xmax
    else:
        nymin, nxmin, nymax, nxmax = ymin / 1000.0, xmin / 1000.0, ymax / 1000.0, xmax / 1000.0

    nymin = min(1.0, max(0.0, nymin))
    nxmin = min(1.0, max(0.0, nxmin))
    nymax = min(1.0, max(0.0, nymax))
    nxmax = min(1.0, max(0.0, nxmax))
    return [nymin, nxmin, nymax, nxmax]


def _norm_to_px(box_norm: List[float], width: int, height: int) -> List[int]:
    nymin, nxmin, nymax, nxmax = box_norm
    x0 = int(round(nxmin * width))
    y0 = int(round(nymin * height))
    x1 = int(round(nxmax * width))
    y1 = int(round(nymax * height))

    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    x0 = min(max(0, x0), width)
    x1 = min(max(0, x1), width)
    y0 = min(max(0, y0), height)
    y1 = min(max(0, y1), height)
    return [x0, y0, x1, y1]


def _decode_mask(mask_data_url: str) -> Optional[Image.Image]:
    prefix = "data:image/png;base64,"
    if not isinstance(mask_data_url, str) or not mask_data_url.startswith(prefix):
        return None
    try:
        raw = base64.b64decode(mask_data_url[len(prefix) :])
        return Image.open(io.BytesIO(raw)).convert("L")
    except Exception:
        return None


def _largest_rect_in_binary(binary: np.ndarray) -> Optional[Tuple[int, int, int, int, int]]:
    h, w = binary.shape
    if h <= 0 or w <= 0:
        return None

    heights = [0] * w
    best_area = 0
    best_left = 0
    best_top = 0
    best_right = 0
    best_bottom = 0

    for row in range(h):
        for col in range(w):
            heights[col] = heights[col] + 1 if bool(binary[row, col]) else 0

        stack: List[Tuple[int, int]] = []
        for col in range(w + 1):
            cur_h = heights[col] if col < w else 0
            start = col
            while stack and stack[-1][1] > cur_h:
                idx, hgt = stack.pop()
                area = hgt * (col - idx)
                if area > best_area:
                    best_area = area
                    best_left = idx
                    best_right = col
                    best_top = row - hgt + 1
                    best_bottom = row + 1
                start = idx
            if not stack or stack[-1][1] < cur_h:
                stack.append((start, cur_h))

    if best_area <= 0:
        return None
    return best_left, best_top, best_right, best_bottom, best_area


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


def _refine_from_mask(
    mask_data_url: str,
    base_box_px: List[int],
    mask_threshold: int,
    prefer_largest_rect: bool,
) -> Optional[Tuple[List[int], Tuple[float, float], int]]:
    mask = _decode_mask(mask_data_url)
    if mask is None:
        return None

    x0, y0, x1, y1 = base_box_px
    bw = x1 - x0
    bh = y1 - y0
    if bw <= 1 or bh <= 1:
        return None

    resized = mask.resize((bw, bh), Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.uint8)
    binary = arr >= int(mask_threshold)
    ys, xs = np.where(binary)
    if xs.size < 4 or ys.size < 4:
        return None

    if prefer_largest_rect:
        largest = _largest_rect_in_binary(binary)
        if largest is not None:
            rx0, ry0, rx1, ry1, _ = largest
            cx = x0 + ((rx0 + rx1) / 2.0)
            cy = y0 + ((ry0 + ry1) / 2.0)
        else:
            rx0 = int(xs.min())
            rx1 = int(xs.max()) + 1
            ry0 = int(ys.min())
            ry1 = int(ys.max()) + 1
            cx = x0 + float(xs.mean())
            cy = y0 + float(ys.mean())
    else:
        rx0 = int(xs.min())
        rx1 = int(xs.max()) + 1
        ry0 = int(ys.min())
        ry1 = int(ys.max()) + 1
        cx = x0 + float(xs.mean())
        cy = y0 + float(ys.mean())

    fx0 = x0 + rx0
    fx1 = x0 + rx1
    fy0 = y0 + ry0
    fy1 = y0 + ry1

    return [fx0, fy0, fx1, fy1], (cx, cy), int(binary.sum())


def _normalize_detections(parsed: Any) -> List[Dict[str, Any]]:
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    if isinstance(parsed, dict):
        if isinstance(parsed.get("objects"), list):
            return [x for x in parsed.get("objects", []) if isinstance(x, dict)]
        return [parsed]
    return []


def _build_furniture_objects(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    include_non_furniture: bool,
    mask_threshold: int,
) -> Tuple[List[Dict[str, Any]], int]:
    excluded = {
        "door",
        "sliding_door",
        "window",
        "wall",
        "floor",
        "ceiling",
        "label",
        "text",
        "dimension",
        "room",
        "opening",
    }

    counts: Counter[str] = Counter()
    out: List[Dict[str, Any]] = []
    invalid = 0

    for det in detections:
        box_raw = _coerce_box_2d(det.get("box_2d"))
        if box_raw is None:
            invalid += 1
            continue

        label = str(det.get("label") or det.get("category") or "unknown")
        category = _to_category_id(label)
        is_room_inner_frame = _is_room_inner_frame_category(category, label)
        if not include_non_furniture and category in excluded:
            continue

        box_norm = _box_to_norm(box_raw)
        x0, y0, x1, y1 = _norm_to_px(box_norm, image_width, image_height)
        box_source = "box_2d"
        mask_fg_pixels = 0

        if isinstance(det.get("mask"), str):
            refined = _refine_from_mask(
                mask_data_url=str(det.get("mask")),
                base_box_px=[x0, y0, x1, y1],
                mask_threshold=int(mask_threshold),
                prefer_largest_rect=is_room_inner_frame,
            )
            if refined is not None:
                [x0, y0, x1, y1], (cx_refined, cy_refined), mask_fg_pixels = refined
                if is_room_inner_frame:
                    box_source = "mask_largest_inner_rect"
                else:
                    box_source = "mask_refined"
            else:
                cx_refined = None
                cy_refined = None
        else:
            cx_refined = None
            cy_refined = None

        w = x1 - x0
        h = y1 - y0
        if w <= 0 or h <= 0:
            invalid += 1
            continue

        idx = counts[category]
        counts[category] += 1
        obj_id = f"{category}_{idx:02d}"

        if isinstance(cx_refined, float) and isinstance(cy_refined, float):
            cx = cx_refined
            cy = cy_refined
        else:
            cx = x0 + w / 2.0
            cy = y0 + h / 2.0

        confidence_raw = det.get("confidence")
        try:
            confidence = float(confidence_raw) if confidence_raw is not None else None
        except (TypeError, ValueError):
            confidence = None

        out.append(
            {
                "id": obj_id,
                "label": label,
                "category": category,
                "confidence": confidence,
                "box_2d_norm": [round(v, 6) for v in box_norm],
                "box_2d_px": [x0, y0, x1, y1],
                "center_px": [round(cx, 2), round(cy, 2)],
                "size_px": [w, h],
                "center_norm": [round(cx / float(image_width), 6), round(cy / float(image_height), 6)],
                "size_norm": [round(w / float(image_width), 6), round(h / float(image_height), 6)],
                "has_mask": isinstance(det.get("mask"), str),
                "box_source": box_source,
                "mask_fg_pixels": int(mask_fg_pixels),
            }
        )

    return out, invalid


def _draw_overlay(
    image_path: pathlib.Path,
    detections: List[Dict[str, Any]],
    furniture_objects: List[Dict[str, Any]],
    out_png: pathlib.Path,
) -> None:
    frame_categories = {"room_inner_frame", "subroom_inner_frame"}
    cats = {
        str(obj.get("category") or "").strip().lower().replace("-", "_")
        for obj in furniture_objects
        if isinstance(obj, dict)
    }
    if cats and cats.issubset(frame_categories):
        _draw_overlay_room_frames_only(
            image_path=image_path,
            furniture_objects=furniture_objects,
            out_png=out_png,
        )
        return

    base = Image.open(image_path).convert("RGBA")
    width, height = base.size

    mask_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    box_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(box_layer, "RGBA")

    det_draw_list: List[Tuple[float, Dict[str, Any], List[int], Tuple[int, int, int]]] = []
    for i, det in enumerate(detections):
        box_raw = _coerce_box_2d(det.get("box_2d"))
        if box_raw is None:
            continue
        box_norm = _box_to_norm(box_raw)
        x0, y0, x1, y1 = _norm_to_px(box_norm, width, height)
        area = float(max(0, x1 - x0) * max(0, y1 - y0))
        color = _hex_to_rgb(SEGMENTATION_COLORS_HEX[i % len(SEGMENTATION_COLORS_HEX)])
        det_draw_list.append((area, det, [x0, y0, x1, y1], color))

    # Spatial-understanding app draws larger masks first.
    det_draw_list.sort(key=lambda t: t[0], reverse=True)

    for _, det, box, color in det_draw_list:
        x0, y0, x1, y1 = box
        mask = _decode_mask(str(det.get("mask") or ""))
        if mask is not None and x1 > x0 and y1 > y0:
            resized = mask.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)
            arr = np.array(resized, dtype=np.uint8)
            rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
            rgba[..., 0] = color[0]
            rgba[..., 1] = color[1]
            rgba[..., 2] = color[2]
            rgba[..., 3] = arr
            patch = Image.fromarray(rgba, mode="RGBA")
            mask_layer.alpha_composite(patch, dest=(x0, y0))

        draw.rectangle([x0, y0, x1, y1], outline=(color[0], color[1], color[2], 255), width=3)

    for i, obj in enumerate(furniture_objects):
        color = _hex_to_rgb(SEGMENTATION_COLORS_HEX[i % len(SEGMENTATION_COLORS_HEX)])
        x0, y0, x1, y1 = [int(v) for v in obj["box_2d_px"]]
        draw.rectangle([x0, y0, x1, y1], fill=(color[0], color[1], color[2], 50))

        cx, cy = obj["center_px"]
        r = 4
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(20, 40, 130, 255))

        label = f"{obj['id']} {obj['label']} {obj['size_px'][0]}x{obj['size_px'][1]}"
        tx = max(0, x0 + 2)
        ty = max(0, y0 - 16)
        draw.rectangle([tx - 2, ty - 2, tx + len(label) * 7, ty + 12], fill=(0, 0, 0, 170))
        draw.text((tx, ty), label, fill=(255, 255, 255, 255))

    composed = Image.alpha_composite(base, mask_layer)
    composed = Image.alpha_composite(composed, box_layer).convert("RGB")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    composed.save(out_png)


def _draw_overlay_room_frames_only(
    image_path: pathlib.Path,
    furniture_objects: List[Dict[str, Any]],
    out_png: pathlib.Path,
) -> None:
    base = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(base, "RGBA")

    frames: List[Dict[str, Any]] = []
    for obj in furniture_objects:
        if not isinstance(obj, dict):
            continue
        cat = str(obj.get("category") or "").strip().lower().replace("-", "_")
        if cat not in {"room_inner_frame", "subroom_inner_frame"}:
            continue
        box = obj.get("box_2d_px")
        center = obj.get("center_px")
        if not (isinstance(box, list) and len(box) >= 4):
            continue
        if not (isinstance(center, list) and len(center) >= 2):
            continue

        x0, y0, x1, y1 = [int(v) for v in box[:4]]
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        cx, cy = float(center[0]), float(center[1])
        frames.append(
            {
                "id": str(obj.get("id") or cat),
                "cat": cat,
                "box": (x0, y0, x1, y1),
                "center": (cx, cy),
            }
        )

    # main first, then subrooms left-to-right.
    frames.sort(key=lambda f: (0 if f["cat"] == "room_inner_frame" else 1, f["center"][0]))

    colors = [
        ((46, 125, 50, 255), (46, 125, 50, 230)),      # main: green
        ((198, 40, 40, 255), (198, 40, 40, 230)),      # sub1: red
        ((25, 118, 210, 255), (25, 118, 210, 230)),    # sub2: blue
        ((123, 31, 162, 255), (123, 31, 162, 230)),    # fallback
    ]

    for i, fr in enumerate(frames):
        line_col, text_col = colors[i % len(colors)]
        x0, y0, x1, y1 = fr["box"]
        cx, cy = fr["center"]

        # box
        draw.rectangle([x0, y0, x1, y1], outline=line_col, width=6)

        # center point + cross
        r = 7
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=line_col)
        draw.line([(cx - 14, cy), (cx + 14, cy)], fill=line_col, width=3)
        draw.line([(cx, cy - 14), (cx, cy + 14)], fill=line_col, width=3)

        # label
        label = f"{i + 1}:{fr['cat']}"
        tx = x0 + 8
        ty = max(4, y0 + 8)
        tw = tx + 10 + 8 * len(label)
        th = ty + 20
        draw.rectangle([tx - 4, ty - 3, tw, th], fill=(255, 255, 255, 200), outline=line_col, width=2)
        draw.text((tx, ty), label, fill=text_col)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(out_png)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Spatial understanding JSON generator (aligned to spatial-understanding app)")
    p.add_argument("--image_path", required=True)
    p.add_argument("--out_json", required=True)

    p.add_argument("--task", choices=["boxes", "masks"], default="boxes")
    p.add_argument("--target_prompt", default="items")
    p.add_argument("--label_prompt", default="")
    p.add_argument("--label_language", default="English")
    p.add_argument("--prompt_text", default=None)

    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--thinking_budget", type=int, default=0)
    p.add_argument("--max_items", type=int, default=20)
    p.add_argument("--resize_max", type=int, default=640)
    p.add_argument("--timeout_sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    p.add_argument("--mask_threshold", type=int, default=DEFAULT_MASK_THRESHOLD)

    p.add_argument("--include_non_furniture", action="store_true")
    p.add_argument("--api_key", default=None)
    p.add_argument("--plot_out", default=None)
    p.add_argument("--raw_response_out", default=None)
    p.add_argument("--dry_run", action="store_true")
    return p


def main() -> None:
    load_dotenv()
    args = build_arg_parser().parse_args()

    image_path = pathlib.Path(args.image_path)
    out_json = pathlib.Path(args.out_json)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    plot_out = pathlib.Path(args.plot_out) if args.plot_out else out_json.with_name(f"{out_json.stem}_plot.png")
    raw_out = pathlib.Path(args.raw_response_out) if args.raw_response_out else out_json.with_name(
        f"{out_json.stem}_raw_response.json"
    )
    manifest_out = out_json.with_name(f"{out_json.stem}_manifest.json")

    original = Image.open(image_path).convert("RGB")
    image_width, image_height = original.size

    resized_image, image_b64 = _resize_for_inference(image_path, max_side=int(args.resize_max))

    if args.prompt_text and str(args.prompt_text).strip():
        prompt = str(args.prompt_text).strip()
    elif args.task == "boxes":
        prompt = _get_2d_prompt(
            target_prompt=str(args.target_prompt),
            label_prompt=str(args.label_prompt),
            max_items=int(args.max_items),
        )
    else:
        prompt = _get_segmentation_prompt(
            target_prompt=str(args.target_prompt),
            label_language=str(args.label_language),
        )

    generation_config: Dict[str, Any] = {
        "temperature": float(args.temperature),
    }
    # Match spatial-understanding app behavior: 2.5 uses thinkingBudget=0 for spatial tasks.
    if "gemini-2.0-flash" not in str(args.model):
        generation_config["thinkingConfig"] = {"thinkingBudget": int(args.thinking_budget)}

    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_b64,
                        }
                    },
                    {"text": prompt},
                ],
            }
        ],
        "generationConfig": generation_config,
    }

    if args.dry_run:
        req_path = out_json.with_name(f"{out_json.stem}_request_payload.json")
        write_json(req_path, payload)
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "request_payload_json": str(req_path),
                    "out_json": str(out_json),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    api_key = args.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set. Use env var or --api_key.")

    endpoint = DEFAULT_ENDPOINT.format(model=args.model)
    raw_response = _post_json(endpoint, api_key=api_key, payload=payload, timeout_sec=int(args.timeout_sec))
    text = _extract_response_text(raw_response)
    parsed = _parse_response_json(text)
    detections = _normalize_detections(parsed)

    furniture_objects, invalid_count = _build_furniture_objects(
        detections=detections,
        image_width=image_width,
        image_height=image_height,
        include_non_furniture=bool(args.include_non_furniture),
        mask_threshold=int(args.mask_threshold),
    )

    _draw_overlay(
        image_path=image_path,
        detections=detections,
        furniture_objects=furniture_objects,
        out_png=plot_out,
    )

    output = {
        "schema_version": "spatial_understanding_v3",
        "source": {
            "provider": "google_ai_studio_gemini_api",
            "model": args.model,
            "endpoint": endpoint,
            "task": args.task,
            "image_path": str(image_path),
            "image_size_px": [image_width, image_height],
            "inference_image_size_px": [resized_image.width, resized_image.height],
            "temperature": float(args.temperature),
            "thinking_budget": int(args.thinking_budget),
            "max_items": int(args.max_items),
            "mask_threshold": int(args.mask_threshold),
            "target_prompt": str(args.target_prompt),
            "label_language": str(args.label_language),
        },
        "prompt_used": prompt,
        "detections_raw": detections,
        "furniture_objects": furniture_objects,
        "detection_count_raw": len(detections),
        "furniture_count": len(furniture_objects),
        "invalid_detection_count": int(invalid_count),
    }

    usage = raw_response.get("usageMetadata") or raw_response.get("usage_metadata") or {}

    write_json(out_json, output)
    write_json(raw_out, raw_response)
    write_json(
        manifest_out,
        {
            "out_json": str(out_json),
            "plot_png": str(plot_out),
            "raw_response_json": str(raw_out),
            "model": args.model,
            "task": args.task,
            "usage": usage,
            "detection_count_raw": len(detections),
            "furniture_count": len(furniture_objects),
            "invalid_detection_count": int(invalid_count),
        },
    )

    print(
        json.dumps(
            {
                "out_json": str(out_json),
                "plot_png": str(plot_out),
                "manifest": str(manifest_out),
                "raw_response": str(raw_out),
                "detection_count_raw": len(detections),
                "furniture_count": len(furniture_objects),
                "invalid_detection_count": int(invalid_count),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
