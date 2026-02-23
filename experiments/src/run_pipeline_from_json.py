from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

STAGE_STEP123 = "step1_openai+step2_spatial_understanding+step3_rule_based"
STAGE_EVAL = "eval_metrics"
STAGE_PLOT = "plot_with_bg"


def _load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        t = value.strip().lower()
        if t in {"1", "true", "yes", "on"}:
            return True
        if t in {"0", "false", "no", "off"}:
            return False
    return default


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_abs(repo_root: pathlib.Path, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    p = pathlib.Path(text)
    if p.is_absolute():
        return str(p)
    return str((repo_root / p).resolve())


def _cli_args_from_options(options: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key, value in options.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                out.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                out.extend([flag, str(item)])
            continue
        out.extend([flag, str(value)])
    return out


def _print_cmd(cmd: List[str]) -> None:
    print("$ " + " ".join(shlex.quote(c) for c in cmd))


def _run_checked(
    cmd: List[str],
    *,
    stage: str,
    env_overrides: Optional[Dict[str, str]] = None,
) -> None:
    print(f"[stage:{stage}] start")
    _print_cmd(cmd)
    env = os.environ.copy()
    if isinstance(env_overrides, dict):
        for key, value in env_overrides.items():
            k = str(key).strip()
            if not k:
                continue
            env[k] = str(value)
    t0 = time.time()
    subprocess.run(cmd, check=True, env=env)
    print(f"[stage:{stage}] done ({round(time.time() - t0, 3)}s)")


def _assert_paths_exist(paths: Sequence[pathlib.Path], *, stage: str) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"stage={stage}: expected output file(s) missing: {missing}")


def _normalize_env(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for key, value in raw.items():
        k = str(key).strip()
        if not k or value is None:
            continue
        out[k] = str(value)
    return out


def _resolve_case_out_dir(
    *,
    repo_root: pathlib.Path,
    run_root: pathlib.Path,
    case_name: str,
    case_out_dir: Optional[str],
) -> pathlib.Path:
    if case_out_dir:
        p = pathlib.Path(_to_abs(repo_root, case_out_dir) or case_out_dir)
        return p
    return run_root / case_name


def _run_single_case(
    *,
    repo_root: pathlib.Path,
    scripts_dir: pathlib.Path,
    python_exec: str,
    run_root: pathlib.Path,
    default_generation_args: Dict[str, Any],
    default_eval_args: Dict[str, Any],
    default_plot_args: Dict[str, Any],
    default_env: Dict[str, str],
    case: Dict[str, Any],
) -> Dict[str, Any]:
    case_name = str(case.get("name") or pathlib.Path(str(case["image_path"])).stem)
    image_path = _to_abs(repo_root, str(case["image_path"]))
    dimensions_path = _to_abs(repo_root, str(case["dimensions_path"]))
    if not image_path or not dimensions_path:
        raise ValueError(f"case={case_name}: image_path/dimensions_path is required")

    out_dir = _resolve_case_out_dir(
        repo_root=repo_root,
        run_root=run_root,
        case_name=case_name,
        case_out_dir=case.get("out_dir"),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    layout_json = out_dir / "layout_generated.json"
    metrics_json = out_dir / "metrics.json"
    debug_dir = out_dir / "debug"
    plot_png = out_dir / "plot_with_bg.png"

    generation_args = _deep_merge(default_generation_args, case.get("generation_args") or {})
    eval_args = _deep_merge(default_eval_args, case.get("eval_args") or {})
    plot_args = _deep_merge(default_plot_args, case.get("plot_args") or {})
    env_overrides = _deep_merge(default_env, _normalize_env(case.get("env")))

    gen_cmd: List[str] = [
        python_exec,
        str(scripts_dir / "generate_layout_json.py"),
        "--image_path",
        image_path,
        "--dimensions_path",
        dimensions_path,
        "--out_json",
        str(layout_json),
        "--out_dir",
        str(out_dir),
    ]
    gen_cmd.extend(_cli_args_from_options(generation_args))
    _run_checked(gen_cmd, stage=STAGE_STEP123, env_overrides=env_overrides)
    _assert_paths_exist([layout_json, out_dir / "generation_manifest.json"], stage=STAGE_STEP123)

    eval_enabled = _as_bool(eval_args.pop("enabled", True), True)
    if eval_args.get("config"):
        eval_args["config"] = _to_abs(repo_root, eval_args["config"])
    if eval_enabled:
        eval_cmd: List[str] = [
            python_exec,
            str(scripts_dir / "eval_metrics.py"),
            "--layout",
            str(layout_json),
            "--out",
            str(metrics_json),
            "--debug_dir",
            str(debug_dir),
        ]
        eval_cmd.extend(_cli_args_from_options(eval_args))
        _run_checked(eval_cmd, stage=STAGE_EVAL, env_overrides=env_overrides)
        _assert_paths_exist([metrics_json], stage=STAGE_EVAL)

    if _as_bool(plot_args.get("enabled"), True):
        bg_image = _to_abs(repo_root, plot_args.get("bg_image") or image_path)
        if not bg_image:
            raise ValueError(f"case={case_name}: plot enabled but bg_image is empty")
        bg_inner_frame_json = _to_abs(
            repo_root,
            plot_args.get("bg_inner_frame_json") or str(out_dir / "gemini_room_inner_frame_output.json"),
        )
        plot_cmd: List[str] = [
            python_exec,
            str(scripts_dir / "plot_layout_json.py"),
            "--layout",
            str(layout_json),
            "--out",
            str(plot_png),
            "--bg_image",
            bg_image,
            "--bg_crop_mode",
            str(plot_args.get("bg_crop_mode") or "none"),
        ]
        if metrics_json.exists():
            plot_cmd.extend(["--metrics_json", str(metrics_json)])
        task_points_path = debug_dir / "task_points.json"
        if task_points_path.exists():
            plot_cmd.extend(["--task_points_json", str(task_points_path)])
        path_cells_path = debug_dir / "path_cells.json"
        if path_cells_path.exists():
            plot_cmd.extend(["--path_json", str(path_cells_path)])
        if bg_inner_frame_json and pathlib.Path(bg_inner_frame_json).exists():
            plot_cmd.extend(["--bg_inner_frame_json", bg_inner_frame_json])
        passthrough = {
            k: v
            for k, v in plot_args.items()
            if k not in {"enabled", "bg_image", "bg_crop_mode", "bg_inner_frame_json"}
        }
        plot_cmd.extend(_cli_args_from_options(passthrough))
        _run_checked(plot_cmd, stage=STAGE_PLOT, env_overrides=env_overrides)
        _assert_paths_exist([plot_png], stage=STAGE_PLOT)

    return {
        "name": case_name,
        "image_path": image_path,
        "dimensions_path": dimensions_path,
        "out_dir": str(out_dir),
        "outputs": {
            "layout_json": str(layout_json),
            "metrics_json": str(metrics_json),
            "plot_with_bg_png": str(plot_png),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run layout pipeline "
            "(Step1 OpenAI + Step2 Spatial Understanding + Step3 rule-based + eval + plot) "
            "from JSON config"
        )
    )
    parser.add_argument("--config", required=True, help="JSON config path")
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional case names to run (if omitted, run all cases in config order)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    scripts_dir = pathlib.Path(__file__).resolve().parent

    config_path = pathlib.Path(_to_abs(repo_root, args.config) or args.config)
    cfg = _load_json(config_path)

    defaults = cfg.get("defaults") or {}
    raw_python_exec = defaults.get("python_exec")
    if isinstance(raw_python_exec, str) and raw_python_exec.strip():
        p = pathlib.Path(raw_python_exec.strip())
        if not p.is_absolute():
            p = repo_root / p
        python_exec = str(p)
    else:
        venv_python = repo_root / ".venv" / "bin" / "python"
        python_exec = str(venv_python if venv_python.exists() else pathlib.Path(sys.executable).resolve())
    run_root = pathlib.Path(
        _to_abs(repo_root, defaults.get("run_root"))
        or str((repo_root / "experiments" / "runs" / "json_pipeline_runs").resolve())
    )
    run_root.mkdir(parents=True, exist_ok=True)

    default_generation_args = defaults.get("generation_args") or {}
    default_eval_args = defaults.get("eval_args") or {"config": "experiments/configs/eval/default_eval.json"}
    default_eval_args = dict(default_eval_args)
    if default_eval_args.get("config"):
        default_eval_args["config"] = _to_abs(repo_root, default_eval_args["config"])
    default_plot_args = defaults.get("plot_args") or {"enabled": True, "bg_crop_mode": "none"}
    default_env = _normalize_env(defaults.get("env"))

    cases = cfg.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("config must include non-empty 'cases' list")

    requested = set(args.cases or [])
    selected_cases: List[Dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        name = str(case.get("name") or pathlib.Path(str(case.get("image_path") or "")).stem)
        if requested and name not in requested:
            continue
        selected_cases.append(case)
    if not selected_cases:
        raise ValueError("No cases selected to run")

    continue_on_error = _as_bool(cfg.get("continue_on_error"), False)
    batch_started = time.time()
    results: List[Dict[str, Any]] = []

    for idx, case in enumerate(selected_cases, start=1):
        case_name = str(case.get("name") or pathlib.Path(str(case.get("image_path") or "")).stem)
        print(f"[{idx}/{len(selected_cases)}] CASE START: {case_name}")
        t0 = time.time()
        try:
            item = _run_single_case(
                repo_root=repo_root,
                scripts_dir=scripts_dir,
                python_exec=python_exec,
                run_root=run_root,
                default_generation_args=default_generation_args,
                default_eval_args=default_eval_args,
                default_plot_args=default_plot_args,
                default_env=default_env,
                case=case,
            )
            item["status"] = "ok"
            item["runtime_sec"] = round(time.time() - t0, 3)
            print(f"[{idx}/{len(selected_cases)}] CASE DONE: {case_name}")
            results.append(item)
        except Exception as exc:  # noqa: BLE001
            item = {
                "name": case_name,
                "status": "error",
                "runtime_sec": round(time.time() - t0, 3),
                "error": str(exc),
            }
            results.append(item)
            print(f"[{idx}/{len(selected_cases)}] CASE ERROR: {case_name}: {exc}")
            if not continue_on_error:
                break

    summary = {
        "config_path": str(config_path),
        "run_root": str(run_root),
        "python_exec": python_exec,
        "started_at_epoch": int(batch_started),
        "runtime_sec": round(time.time() - batch_started, 3),
        "case_count_requested": len(selected_cases),
        "case_count_completed": sum(1 for r in results if r.get("status") == "ok"),
        "case_count_failed": sum(1 for r in results if r.get("status") != "ok"),
        "results": results,
    }
    summary_path = run_root / "batch_manifest.json"
    _write_json(summary_path, summary)
    print(json.dumps({"batch_manifest": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
