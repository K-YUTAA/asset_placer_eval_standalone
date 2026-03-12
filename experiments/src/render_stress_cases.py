from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
from typing import Any, Dict, List, Optional

from stress_dataset_qa import build_dataset_qa_report, write_dataset_qa_report


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _run(cmd: List[str], cwd: pathlib.Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _build_batch_summary_png(*, case_dirs: List[pathlib.Path], run_root: pathlib.Path, image_relpath: str, out_name: str, title: str) -> Optional[str]:
    if not case_dirs:
        return None
    try:
        from PIL import Image
        import matplotlib.pyplot as plt
    except Exception:
        return None

    cols = min(5, max(1, len(case_dirs)))
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
            except Exception:
                ax.text(0.5, 0.5, "invalid image", ha="center", va="center", fontsize=12)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=12)
        ax.axis("off")
        subtitle = ""
        metrics_path = case_dir / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = _load_json(metrics_path)
                clr_value = float(metrics.get("clr_feasible", metrics.get("clr_min_astar", metrics.get("clr_min", 0.0))))
                subtitle = (
                    f"C_vis={float(metrics.get('C_vis', 0.0)):.4f}  "
                    f"R_reach={float(metrics.get('R_reach', 0.0)):.4f}  "
                    f"clr={clr_value:.4f}"
                )
            except Exception:
                subtitle = ""
        ax.set_title(f"{case_dir.parent.name}/{case_dir.name}\n{subtitle}" if subtitle else f"{case_dir.parent.name}/{case_dir.name}", fontsize=10)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_path = run_root / out_name
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def _build_stress_matrix_summary_png(
    *,
    entries: List[Dict[str, Any]],
    run_root: pathlib.Path,
    image_relpath: str,
    out_name: str,
    title: str,
    variant_order: List[str],
) -> Optional[str]:
    if not entries:
        return None
    try:
        from PIL import Image
        import matplotlib.pyplot as plt
    except Exception:
        return None

    case_order: List[str] = []
    matrix: Dict[str, Dict[str, pathlib.Path]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        layout_path = pathlib.Path(str(entry.get("layout_path") or ""))
        if not layout_path.exists():
            continue
        case_dir = layout_path.parent
        base_case_id = str(entry.get("base_case") or entry.get("base_case_id") or case_dir.parent.name)
        variant = str(entry.get("variant") or entry.get("scenario") or entry.get("stress_family") or case_dir.name)
        if base_case_id not in matrix:
            matrix[base_case_id] = {}
            case_order.append(base_case_id)
        matrix[base_case_id][variant] = case_dir

    cols = len(case_order)
    rows = len(variant_order)
    if cols <= 0 or rows <= 0:
        return None

    fig, axes = plt.subplots(rows, cols, figsize=(5.6 * cols, 4.8 * rows), dpi=160)
    if rows == 1 and cols == 1:
        axes_grid = [[axes]]
    elif rows == 1:
        axes_grid = [list(axes)]
    elif cols == 1:
        axes_grid = [[ax] for ax in axes]
    else:
        axes_grid = axes

    for row_idx, variant in enumerate(variant_order):
        for col_idx, case_id in enumerate(case_order):
            ax = axes_grid[row_idx][col_idx]
            case_dir = matrix.get(case_id, {}).get(variant)
            if case_dir is not None:
                img_path = case_dir / image_relpath
                if img_path.exists():
                    try:
                        img = Image.open(img_path).convert("RGB")
                        ax.imshow(img)
                    except Exception:
                        ax.text(0.5, 0.5, "invalid image", ha="center", va="center", fontsize=12)
                else:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=12)
                subtitle = ""
                metrics_path = case_dir / "metrics.json"
                if metrics_path.exists():
                    try:
                        metrics = _load_json(metrics_path)
                        clr_value = float(metrics.get("clr_feasible", metrics.get("clr_min_astar", metrics.get("clr_min", 0.0))))
                        subtitle = (
                            f"C_vis={float(metrics.get('C_vis', 0.0)):.4f}  "
                            f"R_reach={float(metrics.get('R_reach', 0.0)):.4f}  "
                            f"clr={clr_value:.4f}"
                        )
                    except Exception:
                        subtitle = ""
                if row_idx == 0:
                    ax.set_title(f"{case_id}\n{subtitle}" if subtitle else case_id, fontsize=9)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(variant, fontsize=10)
            ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_path = run_root / out_name
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def _source_bg_image(source_layout_path: pathlib.Path) -> Optional[pathlib.Path]:
    source_case_dir = source_layout_path.parent
    manifest_path = source_case_dir / "generation_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = _load_json(manifest_path)
    except Exception:
        return None
    inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
    image_path = inputs.get("image_path")
    if not image_path:
        return None
    p = pathlib.Path(str(image_path))
    return p if p.exists() else None


def _task_and_path(debug_dir: pathlib.Path) -> Dict[str, Optional[pathlib.Path]]:
    task_points = debug_dir / "task_points.json"
    path_cells = debug_dir / "path_cells.json"
    return {
        "task_points": task_points if task_points.exists() else None,
        "path_cells": path_cells if path_cells.exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and render generated stress cases.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--eval_config", default="experiments/configs/eval/eval_v1.json")
    parser.add_argument("--python_exec", default="/Users/yuuta/Research/asset_placer_eval_standalone/.venv/bin/python")
    args = parser.parse_args()

    repo_root = _repo_root()
    run_root = pathlib.Path(args.run_root)
    if not run_root.is_absolute():
        run_root = (repo_root / run_root).resolve()
    eval_config = pathlib.Path(args.eval_config)
    if not eval_config.is_absolute():
        eval_config = (repo_root / eval_config).resolve()
    python_exec = pathlib.Path(args.python_exec)

    manifest_path = run_root / "stress_cases_manifest.json"
    manifest = _load_json(manifest_path)
    entries = manifest.get("entries") if isinstance(manifest.get("entries"), list) else []

    case_dirs: List[pathlib.Path] = []
    variant_dirs: Dict[str, List[pathlib.Path]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        layout_path = pathlib.Path(str(entry.get("layout_path") or ""))
        source_layout_path = pathlib.Path(str(entry.get("source_layout_path") or ""))
        if not layout_path.exists() or not source_layout_path.exists():
            continue
        case_dir = layout_path.parent
        variant = str(entry.get("variant") or case_dir.name)
        debug_dir = case_dir / "debug"
        metrics_json = case_dir / "metrics.json"
        plot_png = case_dir / "plot_with_bg.png"
        bg_image = _source_bg_image(source_layout_path)
        bg_inner_frame_json = source_layout_path.parent / "gemini_room_inner_frame_output.json"

        eval_cmd = [
            str(python_exec),
            str(repo_root / "experiments/src/eval_metrics.py"),
            "--layout", str(layout_path),
            "--config", str(eval_config),
            "--out", str(metrics_json),
            "--debug_dir", str(debug_dir),
        ]
        _run(eval_cmd, repo_root)

        plot_cmd = [
            str(python_exec),
            str(repo_root / "experiments/src/plot_layout_json.py"),
            "--layout", str(layout_path),
            "--out", str(plot_png),
            "--bg_crop_mode", "none",
        ]
        if bg_image is not None:
            plot_cmd.extend(["--bg_image", str(bg_image)])
        if metrics_json.exists():
            plot_cmd.extend(["--metrics_json", str(metrics_json)])
        task_paths = _task_and_path(debug_dir)
        if task_paths["task_points"] is not None:
            plot_cmd.extend(["--task_points_json", str(task_paths["task_points"])])
        if task_paths["path_cells"] is not None:
            plot_cmd.extend(["--path_json", str(task_paths["path_cells"])])
        if debug_dir.exists():
            plot_cmd.extend(["--eval_debug_dir", str(debug_dir)])
        if bg_inner_frame_json.exists():
            plot_cmd.extend(["--bg_inner_frame_json", str(bg_inner_frame_json)])
        _run(plot_cmd, repo_root)

        case_dirs.append(case_dir)
        variant_dirs.setdefault(variant, []).append(case_dir)

    sample_schema = _load_json((repo_root / "experiments/configs/stress/stress_sample_manifest_schema.json").resolve())
    stress_cfg_path = pathlib.Path(str(manifest.get("degrade_config") or ""))
    if not stress_cfg_path.is_absolute():
        stress_cfg_path = (repo_root / stress_cfg_path).resolve()
    stress_cfg = _load_json(stress_cfg_path)
    qa_report = build_dataset_qa_report(
        out_root=run_root,
        stress_cfg=stress_cfg,
        stress_config_path=stress_cfg_path,
        sample_schema=sample_schema,
        entries=entries,
        repo_root=repo_root,
    )
    write_dataset_qa_report(run_root / "stress_dataset_qa_report.json", qa_report)

    summary_images: Dict[str, str] = {}
    variant_order = [str(x) for x in (stress_cfg.get("variants") or []) if str(x)]
    saved = _build_stress_matrix_summary_png(
        entries=[entry for entry in entries if isinstance(entry, dict)],
        run_root=run_root,
        image_relpath="plot_with_bg.png",
        out_name="plot_with_bg_summary_all_variants.png",
        title=f"Stress plot_with_bg summary ({len(case_dirs)} cases)",
        variant_order=variant_order or sorted(variant_dirs.keys()),
    )
    if saved:
        summary_images["plot_with_bg_summary_all_variants"] = saved

    for variant, dirs in sorted(variant_dirs.items()):
        saved = _build_batch_summary_png(
            case_dirs=dirs,
            run_root=run_root,
            image_relpath="plot_with_bg.png",
            out_name=f"plot_with_bg_summary_{variant}.png",
            title=f"Stress plot_with_bg summary: {variant} ({len(dirs)} cases)",
        )
        if saved:
            summary_images[f"plot_with_bg_summary_{variant}"] = saved

    render_manifest = {
        "run_root": str(run_root),
        "eval_config": str(eval_config),
        "case_count": len(case_dirs),
        "summary_images": summary_images,
        "qa_report": str(run_root / "stress_dataset_qa_report.json"),
    }
    (run_root / "stress_render_manifest.json").write_text(json.dumps(render_manifest, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    print(json.dumps(render_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
