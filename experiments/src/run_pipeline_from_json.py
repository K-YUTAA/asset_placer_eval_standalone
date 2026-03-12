from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


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


def _build_batch_summary_png(
    *,
    case_dirs: List[pathlib.Path],
    run_root: pathlib.Path,
    image_relpath: str,
    out_name: str,
    title: str,
    include_metrics: bool = False,
) -> Optional[str]:
    if not case_dirs:
        return None
    try:
        from PIL import Image  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"[summary:{out_name}] skipped: failed to import plotting deps: {exc}")
        return None

    cols = 3
    rows = max(1, int(math.ceil(len(case_dirs) / float(cols))))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 5.4 * rows), dpi=160)
    try:
        flat_axes = list(axes.reshape(-1))
    except Exception:
        flat_axes = [axes]

    for idx, ax in enumerate(flat_axes):
        if idx >= len(case_dirs):
            ax.axis("off")
            continue
        case_dir = case_dirs[idx]
        img_path = case_dir / image_relpath
        if img_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
            except Exception:  # noqa: BLE001
                ax.text(0.5, 0.5, "invalid image", ha="center", va="center", fontsize=12)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=12)
        ax.axis("off")

        subtitle = ""
        if include_metrics:
            metrics_path = case_dir / "metrics.json"
            if metrics_path.exists():
                try:
                    metrics = _load_json(metrics_path)
                    subtitle = (
                        f"C_vis={_as_float(metrics.get('C_vis'), 0.0):.4f}  "
                        f"R_reach={_as_float(metrics.get('R_reach'), 0.0):.4f}  "
                        f"clr_min={_as_float(metrics.get('clr_min'), 0.0):.4f}"
                    )
                except Exception:  # noqa: BLE001
                    subtitle = ""
        if subtitle:
            ax.set_title(f"{case_dir.name}\n{subtitle}", fontsize=10)
        else:
            ax.set_title(case_dir.name, fontsize=10)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_path = run_root / out_name
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[summary:{out_name}] saved: {out_path}")
    return str(out_path)


def _generate_default_batch_summaries(
    *,
    results: List[Dict[str, Any]],
    run_root: pathlib.Path,
) -> Dict[str, str]:
    case_dirs: List[pathlib.Path] = []
    for r in results:
        if str(r.get("status")) != "ok":
            continue
        out_dir = r.get("out_dir")
        if not out_dir:
            continue
        p = pathlib.Path(str(out_dir))
        if p.exists():
            case_dirs.append(p)
    if not case_dirs:
        return {}

    specs = [
        {
            "image_relpath": "plot_with_bg.png",
            "out_name": "plot_with_bg_batch_summary.png",
            "title": f"Batch plot_with_bg summary ({len(case_dirs)} cases)",
            "include_metrics": True,
            "key": "plot_with_bg_batch_summary",
        },
        {
            "image_relpath": "debug/c_vis_area.png",
            "out_name": "c_vis_area_batch_summary.png",
            "title": f"Batch c_vis_area summary ({len(case_dirs)} cases)",
            "include_metrics": False,
            "key": "c_vis_area_batch_summary",
        },
        {
            "image_relpath": "debug/c_vis_start_area.png",
            "out_name": "c_vis_start_area_batch_summary.png",
            "title": f"Batch c_vis_start_area summary ({len(case_dirs)} cases)",
            "include_metrics": False,
            "key": "c_vis_start_area_batch_summary",
        },
        {
            "image_relpath": "debug/c_vis_objects_area.png",
            "out_name": "c_vis_objects_area_batch_summary.png",
            "title": f"Batch c_vis_objects_area summary ({len(case_dirs)} cases)",
            "include_metrics": False,
            "key": "c_vis_objects_area_batch_summary",
        },
        {
            "image_relpath": "debug/c_vis_start_objects_area.png",
            "out_name": "c_vis_start_objects_area_batch_summary.png",
            "title": f"Batch c_vis_start_objects_area summary ({len(case_dirs)} cases)",
            "include_metrics": False,
            "key": "c_vis_start_objects_area_batch_summary",
        },
    ]

    out: Dict[str, str] = {}
    for spec in specs:
        saved = _build_batch_summary_png(
            case_dirs=case_dirs,
            run_root=run_root,
            image_relpath=str(spec["image_relpath"]),
            out_name=str(spec["out_name"]),
            title=str(spec["title"]),
            include_metrics=bool(spec["include_metrics"]),
        )
        if saved:
            out[str(spec["key"])] = saved
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
    eval_config_path = pathlib.Path(str(eval_args.get("config"))) if eval_args.get("config") else None
    eval_hash = _sha256_file(eval_config_path) if eval_config_path and eval_config_path.exists() else None
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
        eval_debug_dir = _to_abs(repo_root, plot_args.get("eval_debug_dir") or str(debug_dir))
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
        if eval_debug_dir and pathlib.Path(eval_debug_dir).exists():
            plot_cmd.extend(["--eval_debug_dir", eval_debug_dir])
        if bg_inner_frame_json and pathlib.Path(bg_inner_frame_json).exists():
            plot_cmd.extend(["--bg_inner_frame_json", bg_inner_frame_json])
        passthrough = {
            k: v
            for k, v in plot_args.items()
            if k not in {"enabled", "bg_image", "bg_crop_mode", "bg_inner_frame_json", "eval_debug_dir"}
        }
        plot_cmd.extend(_cli_args_from_options(passthrough))
        _run_checked(plot_cmd, stage=STAGE_PLOT, env_overrides=env_overrides)
        _assert_paths_exist([plot_png], stage=STAGE_PLOT)

    return {
        "name": case_name,
        "image_path": image_path,
        "dimensions_path": dimensions_path,
        "out_dir": str(out_dir),
        "eval_config_path": str(eval_config_path) if eval_config_path else None,
        "eval_config_name": eval_config_path.name if eval_config_path else None,
        "eval_hash": eval_hash,
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
    default_eval_args = defaults.get("eval_args") or {"config": "experiments/configs/eval/eval_v1.json"}
    default_eval_args = dict(default_eval_args)
    if default_eval_args.get("config"):
        default_eval_args["config"] = _to_abs(repo_root, default_eval_args["config"])
    default_plot_args = defaults.get("plot_args") or {"enabled": True, "bg_crop_mode": "none"}
    default_summary_args = defaults.get("summary_args") or {"enabled": True}
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

    summary_images: Dict[str, str] = {}
    if _as_bool(default_summary_args.get("enabled"), True):
        summary_images = _generate_default_batch_summaries(results=results, run_root=run_root)

    summary = {
        "config_path": str(config_path),
        "run_root": str(run_root),
        "python_exec": python_exec,
        "eval_hashes": sorted({str(r.get("eval_hash")) for r in results if r.get("eval_hash")}),
        "started_at_epoch": int(batch_started),
        "runtime_sec": round(time.time() - batch_started, 3),
        "case_count_requested": len(selected_cases),
        "case_count_completed": sum(1 for r in results if r.get("status") == "ok"),
        "case_count_failed": sum(1 for r in results if r.get("status") != "ok"),
        "summary_images": summary_images,
        "results": results,
    }
    summary_path = run_root / "batch_manifest.json"
    _write_json(summary_path, summary)
    print(json.dumps({"batch_manifest": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
