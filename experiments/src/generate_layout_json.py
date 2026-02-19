from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import pathlib
import subprocess
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from layout_tools import extract_json_payload, read_text, write_json

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
    "This image is a floor plan. Detect only the inner room boundary frames (inside face of walls). "
    "Return a JSON list where each entry has keys \"label\", \"box_2d\", and \"mask\". "
    "Use label \"room_inner_frame\" for the main room and \"subroom_inner_frame\" for enclosed sub-rooms. "
    "The mask must represent the interior area bounded by each inner frame. "
    "Do not include furniture, text, doors, windows, or wall thickness outside the interior."
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

    best: Optional[Dict[str, Any]] = None
    best_area = -1.0

    for obj in room_inner_frame_objects:
        if not isinstance(obj, dict):
            continue
        category = str(obj.get("category") or "").strip().lower().replace("-", "_")
        label = str(obj.get("label") or "").strip().lower()
        if "subroom" in category or "sub room" in label:
            continue

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

        area = w * h
        if area <= best_area:
            continue

        polygon = [
            {"X": round(xmin_w, 3), "Y": round(ymin_w, 3)},
            {"X": round(xmax_w, 3), "Y": round(ymin_w, 3)},
            {"X": round(xmax_w, 3), "Y": round(ymax_w, 3)},
            {"X": round(xmin_w, 3), "Y": round(ymax_w, 3)},
        ]

        best = {
            "source_object_id": str(obj.get("id") or ""),
            "box_world": [round(xmin_w, 3), round(ymin_w, 3), round(xmax_w, 3), round(ymax_w, 3)],
            "width_world": round(w, 3),
            "height_world": round(h, 3),
            "polygon": polygon,
        }
        best_area = area

    return best


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
    parser = argparse.ArgumentParser(description="Generate layout JSON (Step1 + optional Gemini + Step2) without Isaac/Omni")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--dimensions_path", required=True)
    parser.add_argument("--prompt1_path", default=None)
    parser.add_argument("--prompt2_path", default=None)
    parser.add_argument("--analysis_input", default=None, help="Optional existing Step1 output text/JSON path")
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_dir", default=None)

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning_effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--text_verbosity", default=DEFAULT_TEXT_VERBOSITY)
    parser.add_argument("--max_output_tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--image_detail", default=DEFAULT_IMAGE_DETAIL)
    parser.add_argument("--step2_text_only", action="store_true")
    parser.add_argument("--api_key", default=None)

    parser.add_argument("--enable_gemini_spatial", action="store_true")
    parser.add_argument("--gemini_api_key", default=None)
    parser.add_argument("--gemini_model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--gemini_task", default=DEFAULT_GEMINI_TASK, choices=["boxes", "masks"])
    parser.add_argument("--gemini_target_prompt", default=None)
    parser.add_argument("--gemini_prompt_text", default=None)
    parser.add_argument("--gemini_label_language", default=DEFAULT_GEMINI_LABEL_LANGUAGE)
    parser.add_argument("--gemini_temperature", type=float, default=DEFAULT_GEMINI_TEMPERATURE)
    parser.add_argument("--gemini_thinking_budget", type=int, default=DEFAULT_GEMINI_THINKING_BUDGET)
    parser.add_argument("--gemini_max_items", type=int, default=DEFAULT_GEMINI_MAX_ITEMS)
    parser.add_argument("--gemini_resize_max", type=int, default=DEFAULT_GEMINI_RESIZE_MAX)
    parser.add_argument("--gemini_include_non_furniture", action="store_true")
    parser.add_argument("--enable_gemini_openings", action="store_true")
    parser.add_argument("--gemini_openings_prompt_text", default=None)
    return parser


def main() -> None:
    load_dotenv()
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    image_path = pathlib.Path(args.image_path)
    dimensions_path = pathlib.Path(args.dimensions_path)
    prompt1_path = pathlib.Path(args.prompt1_path) if args.prompt1_path else (repo_root / "prompts" / "prompt_1_universal_v4_posfix2_gemini_bridge_v1.txt")
    prompt2_path = pathlib.Path(args.prompt2_path) if args.prompt2_path else (repo_root / "prompts" / "prompt_2_universal_v4_posfix2_gemini_bridge_v1.txt")
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
    if not prompt2_path.exists():
        raise FileNotFoundError(f"prompt2 not found: {prompt2_path}")

    dimensions_text = read_text(dimensions_path)
    prompt1_text = read_text(prompt1_path)
    prompt2_text = read_text(prompt2_path)

    image_base64, image_mime = _encode_image_base64(image_path)
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
    write_json(out_dir / "step1_output_raw.json", {"text": step1_text})
    write_json(out_dir / "step1_output_parsed.json", step1_json)

    gemini_details: Dict[str, Any] = {"enabled": bool(args.enable_gemini_spatial)}
    gemini_spatial_json: Optional[Dict[str, Any]] = None

    if args.enable_gemini_spatial:
        step1_category_counts = _collect_step1_category_counts(step1_json)
        gemini_target_prompt = args.gemini_target_prompt or _build_default_gemini_target_prompt(step1_category_counts)

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
            gemini_prompt_text=args.gemini_prompt_text,
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
            gemini_target_prompt="items",
            gemini_label_language="English",
            gemini_temperature=args.gemini_temperature,
            gemini_thinking_budget=args.gemini_thinking_budget,
            gemini_max_items=args.gemini_max_items,
            gemini_resize_max=args.gemini_resize_max,
            gemini_prompt_text=DEFAULT_GEMINI_ROOM_INNER_FRAME_PROMPT,
            gemini_include_non_furniture=True,
        )
        room_inner_frame_json = room_inner_frame_result["spatial_json"]
        room_inner_frame_objects = _extract_room_inner_frame_objects(room_inner_frame_json)
        main_room_inner_boundary_hint = _build_main_room_inner_boundary_hint(
            step1_json=step1_json,
            room_inner_frame_objects=room_inner_frame_objects,
        )
        opening_objects: List[Dict[str, Any]] = []
        openings_details: Dict[str, Any] = {"enabled": bool(args.enable_gemini_openings)}

        if args.enable_gemini_openings:
            openings_prompt = args.gemini_openings_prompt_text or DEFAULT_GEMINI_OPENINGS_PROMPT
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
                "target_prompt": "items",
                "label_language": "English",
                "temperature": float(args.gemini_temperature),
                "thinking_budget": int(args.gemini_thinking_budget),
                "max_items": int(args.gemini_max_items),
                "resize_max": int(args.gemini_resize_max),
                "prompt_text": DEFAULT_GEMINI_ROOM_INNER_FRAME_PROMPT,
                "outputs": room_inner_frame_result["paths"],
                "manifest": room_inner_frame_result["manifest_json"],
                "object_count": len(room_inner_frame_objects),
                "main_room_inner_boundary_hint": main_room_inner_boundary_hint,
            },
            "openings": openings_details,
        }
    else:
        room_inner_frame_objects = []
        main_room_inner_boundary_hint = None
        opening_objects = []

    step2_prompt_parts = [
        prompt2_text,
        "\n\nSTEP1_JSON:\n",
        json.dumps(step1_json, ensure_ascii=False, indent=2),
    ]

    if gemini_spatial_json is not None:
        step2_prompt_parts.extend(
            [
                "\n\nGEMINI_SPATIAL_UNDERSTANDING_JSON:\n",
                json.dumps(gemini_spatial_json, ensure_ascii=False, indent=2),
                "\n\nGEMINI_INTEGRATION_POLICY:\n"
                "- Keep object inventory (category/count) and semantic intent from STEP1_JSON.\n"
                "- Use GEMINI furniture_objects center/size hints as primary evidence for position/size.\n"
                "- If Gemini misses an object, infer it from STEP1_JSON and LAYOUT_HINTS.\n"
                "- Preserve valid room geometry/openings and output schema constraints.\n",
            ]
        )
    if room_inner_frame_objects:
        step2_prompt_parts.extend(
            [
                "\n\nGEMINI_ROOM_INNER_FRAME_JSON:\n",
                json.dumps({"furniture_objects": room_inner_frame_objects}, ensure_ascii=False, indent=2),
                "\n\nGEMINI_ROOM_POLICY:\n"
                "- Use room_inner_frame/subroom_inner_frame as primary evidence for room inside boundaries.\n"
                "- Treat these boxes as interior envelopes (inside face of walls), not outer wall thickness.\n",
            ]
        )
    if main_room_inner_boundary_hint is not None:
        step2_prompt_parts.extend(
            [
                "\n\nGEMINI_MAIN_ROOM_INNER_BOUNDARY_WORLD:\n",
                json.dumps(main_room_inner_boundary_hint, ensure_ascii=False, indent=2),
                "\n\nGEMINI_OUTER_BOUNDARY_POLICY:\n"
                "- Set final outer_polygon to GEMINI_MAIN_ROOM_INNER_BOUNDARY_WORLD.polygon exactly.\n"
                "- Treat this as the room inner wall line (walkable boundary), not wall outer face.\n"
                "- If STEP1 outer_polygon conflicts with this inner boundary, prioritize GEMINI_MAIN_ROOM_INNER_BOUNDARY_WORLD.\n",
            ]
        )
    if opening_objects:
        step2_prompt_parts.extend(
            [
                "\n\nGEMINI_OPENINGS_JSON:\n",
                json.dumps({"furniture_objects": opening_objects}, ensure_ascii=False, indent=2),
                "\n\nGEMINI_OPENINGS_POLICY:\n"
                "- Use Gemini door/window detections as primary evidence for opening positions.\n"
                "- Prefer opening centers and extents from GEMINI_OPENINGS_JSON when defining wall openings.\n",
            ]
        )

    if not args.step2_text_only:
        step2_prompt_parts.extend(["\n\nLAYOUT_HINTS:\n", dimensions_text])
    step2_prompt = "".join(step2_prompt_parts)

    step2_text, step2_usage, step2_model = _call_responses(
        client=client,
        model=args.model,
        prompt_text=step2_prompt,
        image_base64=None if args.step2_text_only else image_base64,
        image_mime=image_mime,
        image_detail=args.image_detail,
        reasoning_effort=args.reasoning_effort,
        text_verbosity=args.text_verbosity,
        max_output_tokens=args.max_output_tokens,
    )

    final_json = extract_json_payload(step2_text)
    write_json(out_dir / "step2_output_raw.json", {"text": step2_text})
    write_json(out_json, final_json)

    runtime_sec = time.perf_counter() - t0
    run_manifest = {
        "runtime_sec": runtime_sec,
        "model_requested": args.model,
        "model_used": {"step1": step1_model, "step2": step2_model},
        "step2_text_only": bool(args.step2_text_only),
        "gemini_spatial": gemini_details,
        "inputs": {
            "image_path": str(image_path),
            "dimensions_path": str(dimensions_path),
            "prompt1_path": str(prompt1_path),
            "prompt2_path": str(prompt2_path),
            "analysis_input": str(args.analysis_input) if args.analysis_input else None,
        },
        "usage": {
            "step1": step1_usage,
            "step2": step2_usage,
            "total": {
                "input_tokens": int(step1_usage["input_tokens"] + step2_usage["input_tokens"]),
                "output_tokens": int(step1_usage["output_tokens"] + step2_usage["output_tokens"]),
                "total_tokens": int(step1_usage["total_tokens"] + step2_usage["total_tokens"]),
            },
        },
        "outputs": {
            "layout_json": str(out_json),
            "step1_raw": str(out_dir / "step1_output_raw.json"),
            "step1_parsed": str(out_dir / "step1_output_parsed.json"),
            "step2_raw": str(out_dir / "step2_output_raw.json"),
            "gemini_spatial_json": gemini_details.get("outputs", {}).get("spatial_json") if gemini_details.get("enabled") else None,
            "gemini_plot_png": gemini_details.get("outputs", {}).get("plot_png") if gemini_details.get("enabled") else None,
            "gemini_room_inner_frame_json": gemini_details.get("room_inner_frame", {}).get("outputs", {}).get("spatial_json") if gemini_details.get("enabled") else None,
            "gemini_room_inner_frame_plot_png": gemini_details.get("room_inner_frame", {}).get("outputs", {}).get("plot_png") if gemini_details.get("enabled") else None,
            "gemini_openings_json": gemini_details.get("openings", {}).get("outputs", {}).get("spatial_json") if gemini_details.get("enabled") else None,
            "gemini_openings_plot_png": gemini_details.get("openings", {}).get("outputs", {}).get("plot_png") if gemini_details.get("enabled") else None,
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
