from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from layout_tools import extract_json_payload, read_text, write_json

DEFAULT_MODEL = "gpt-5.2"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_TEXT_VERBOSITY = "high"
DEFAULT_MAX_OUTPUT_TOKENS = 32000
DEFAULT_IMAGE_DETAIL = "high"


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
    kwargs: Dict[str, Any] = {
        "model": model,
        "input": _build_responses_input(prompt_text, image_base64, image_mime, image_detail),
        "max_output_tokens": int(max_output_tokens),
    }
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    if text_verbosity:
        kwargs["text"] = {"verbosity": text_verbosity}

    response = _create_response_with_fallback(client, kwargs)
    output_text = _extract_response_text(response)
    if not output_text.strip():
        raise RuntimeError("OpenAI response text was empty")

    usage = _usage_tokens(getattr(response, "usage", None))
    model_used = str(getattr(response, "model", model))
    return output_text, usage, model_used


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate layout JSON (Step1/Step2) without Isaac/Omni")
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
    return parser


def main() -> None:
    load_dotenv()
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    image_path = pathlib.Path(args.image_path)
    dimensions_path = pathlib.Path(args.dimensions_path)
    prompt1_path = pathlib.Path(args.prompt1_path) if args.prompt1_path else (repo_root / "prompts" / "prompt_step1_universal_v7.txt")
    prompt2_path = pathlib.Path(args.prompt2_path) if args.prompt2_path else (repo_root / "prompts" / "prompt_step2_universal_v7.txt")
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

    step2_prompt_parts = [
        prompt2_text,
        "\n\nSTEP1_JSON:\n",
        json.dumps(step1_json, ensure_ascii=False, indent=2),
    ]
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
        },
    }
    write_json(out_dir / "generation_manifest.json", run_manifest)

    print(json.dumps({"layout_json": str(out_json), "generation_manifest": str(out_dir / "generation_manifest.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
