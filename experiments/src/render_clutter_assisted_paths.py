from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
from typing import Any, Dict, List, Optional, Tuple


VARIANT_ORDER_ALL = ["base", "usage_shift", "clutter", "compound"]
VARIANT_ORDER_ACTIVE = ["clutter", "compound"]


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _run(cmd: List[str], cwd: pathlib.Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _stress_case_dir_from_trial_manifest(trial_manifest: Dict[str, Any]) -> Optional[pathlib.Path]:
    trial_cfg = trial_manifest.get("trial_config") if isinstance(trial_manifest.get("trial_config"), dict) else {}
    inputs = trial_cfg.get("inputs") if isinstance(trial_cfg.get("inputs"), dict) else {}
    layout_input = pathlib.Path(str(inputs.get("layout_input") or ""))
    if not layout_input.exists():
        return None
    return layout_input.parent


def _source_layout_path(stress_case_dir: pathlib.Path) -> Optional[pathlib.Path]:
    layout_path = stress_case_dir / "layout_generated.json"
    if not layout_path.exists():
        return None
    try:
        layout = _load_json(layout_path)
    except Exception:
        return None
    meta = layout.get("meta") if isinstance(layout.get("meta"), dict) else {}
    raw = meta.get("source_layout_path")
    if not raw:
        return None
    path = pathlib.Path(str(raw))
    return path if path.exists() else None


def _source_bg_image(trial_manifest: Dict[str, Any], source_layout_path: Optional[pathlib.Path]) -> Optional[pathlib.Path]:
    trial_cfg = trial_manifest.get("trial_config") if isinstance(trial_manifest.get("trial_config"), dict) else {}
    inputs = trial_cfg.get("inputs") if isinstance(trial_cfg.get("inputs"), dict) else {}
    sketch_path = pathlib.Path(str(inputs.get("sketch_path") or ""))
    if sketch_path.exists():
        return sketch_path
    if source_layout_path is None:
        return None
    generation_manifest = source_layout_path.parent / "generation_manifest.json"
    if not generation_manifest.exists():
        return None
    try:
        manifest = _load_json(generation_manifest)
    except Exception:
        return None
    gm_inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
    image_path = pathlib.Path(str(gm_inputs.get("image_path") or ""))
    return image_path if image_path.exists() else None


def _build_plot_with_bg(
    *,
    repo_root: pathlib.Path,
    python_exec: pathlib.Path,
    eval_config: pathlib.Path,
    run_dir: pathlib.Path,
    stress_case_dir: pathlib.Path,
    trial_manifest: Dict[str, Any],
) -> Optional[pathlib.Path]:
    refined_layout = run_dir / "layout_refined.json"
    if not refined_layout.exists():
        return None

    source_layout = _source_layout_path(stress_case_dir)
    bg_image = _source_bg_image(trial_manifest, source_layout)
    bg_inner_frame = source_layout.parent / "gemini_room_inner_frame_output.json" if source_layout is not None else None

    debug_dir = run_dir / "debug_refined"
    metrics_json = run_dir / "metrics_refined_eval.json"
    plot_png = run_dir / "plot_with_bg_refined.png"

    eval_cmd = [
        str(python_exec),
        str(repo_root / "experiments/src/eval_metrics.py"),
        "--layout",
        str(refined_layout),
        "--config",
        str(eval_config),
        "--out",
        str(metrics_json),
        "--debug_dir",
        str(debug_dir),
    ]
    _run(eval_cmd, repo_root)

    plot_cmd = [
        str(python_exec),
        str(repo_root / "experiments/src/plot_layout_json.py"),
        "--layout",
        str(refined_layout),
        "--out",
        str(plot_png),
        "--bg_crop_mode",
        "none",
    ]
    if bg_image is not None:
        plot_cmd.extend(["--bg_image", str(bg_image)])
    if metrics_json.exists():
        plot_cmd.extend(["--metrics_json", str(metrics_json)])
    task_points = debug_dir / "task_points.json"
    path_cells = debug_dir / "path_cells.json"
    if task_points.exists():
        plot_cmd.extend(["--task_points_json", str(task_points)])
    if path_cells.exists():
        plot_cmd.extend(["--path_json", str(path_cells)])
    if debug_dir.exists():
        plot_cmd.extend(["--eval_debug_dir", str(debug_dir)])
    if bg_inner_frame is not None and bg_inner_frame.exists():
        plot_cmd.extend(["--bg_inner_frame_json", str(bg_inner_frame)])
    _run(plot_cmd, repo_root)
    return plot_png if plot_png.exists() else None


def _build_compare_image(run_dir: pathlib.Path) -> Optional[pathlib.Path]:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None

    trial_manifest = _load_json(run_dir / "trial_manifest.json")
    stress_case_dir = _stress_case_dir_from_trial_manifest(trial_manifest)
    if stress_case_dir is None:
        return None
    before_png = stress_case_dir / "plot_with_bg.png"
    after_png = run_dir / "plot_with_bg_refined.png"
    if not before_png.exists() or not after_png.exists():
        return None

    before = Image.open(before_png).convert("RGB")
    after = Image.open(after_png).convert("RGB")
    height = max(before.height, after.height)
    width = before.width + after.width
    out = Image.new("RGB", (width, height + 28), "white")
    out.paste(before, (0, 28))
    out.paste(after, (before.width, 28))

    draw = ImageDraw.Draw(out)
    draw.text((12, 6), "before (stress input)", fill="black")
    draw.text((before.width + 12, 6), "after (clutter-assisted refined)", fill="black")

    out_path = run_dir / "plot_with_bg_compare.png"
    out.save(out_path)
    return out_path


def _collect_matrix_items(run_root: pathlib.Path, image_name: str) -> Tuple[List[str], Dict[str, Dict[str, pathlib.Path]]]:
    case_order: List[str] = []
    matrix: Dict[str, Dict[str, pathlib.Path]] = {}
    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "trial_manifest.json"
        image_path = run_dir / image_name
        if not manifest_path.exists() or not image_path.exists():
            continue
        manifest = _load_json(manifest_path)
        summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
        layout_id = str(summary.get("layout_id") or "")
        if not layout_id:
            continue
        stress_case_dir = _stress_case_dir_from_trial_manifest(manifest)
        variant = stress_case_dir.name if stress_case_dir is not None else ""
        if not variant:
            continue
        if layout_id not in matrix:
            matrix[layout_id] = {}
            case_order.append(layout_id)
        matrix[layout_id][variant] = image_path
    return case_order, matrix


def _build_summary(
    *,
    run_root: pathlib.Path,
    image_name: str,
    out_name: str,
    title: str,
    variant_order: List[str],
) -> Optional[pathlib.Path]:
    try:
        from PIL import Image
        import matplotlib.pyplot as plt
    except Exception:
        return None

    case_order, matrix = _collect_matrix_items(run_root, image_name)
    if not case_order:
        return None

    rows = len(variant_order)
    cols = len(case_order)
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
            img_path = matrix.get(case_id, {}).get(variant)
            if img_path is None or not img_path.exists():
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=12)
            else:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(case_id, fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(variant, fontsize=10)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_path = run_root.parent / out_name
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render clutter-assisted plot_with_bg visualizations.")
    parser.add_argument("--compare_root", required=True)
    parser.add_argument("--eval_config", default="experiments/configs/eval/eval_v1.json")
    parser.add_argument("--python_exec", default="/Users/yuuta/Research/asset_placer_eval_standalone/.venv/bin/python")
    args = parser.parse_args()

    repo_root = _repo_root()
    compare_root = pathlib.Path(args.compare_root)
    if not compare_root.is_absolute():
        compare_root = (repo_root / compare_root).resolve()
    eval_config = pathlib.Path(args.eval_config)
    if not eval_config.is_absolute():
        eval_config = (repo_root / eval_config).resolve()
    python_exec = pathlib.Path(args.python_exec)

    render_manifest: Dict[str, Any] = {
        "compare_root": str(compare_root),
        "eval_config": str(eval_config),
        "methods": {},
    }

    for method_dir_name in ("heuristic_run", "proposed_run"):
        run_root = compare_root / method_dir_name
        if not run_root.exists():
            continue
        rendered = 0
        compared = 0
        for run_dir in sorted(run_root.iterdir()):
            if not run_dir.is_dir():
                continue
            manifest_path = run_dir / "trial_manifest.json"
            if not manifest_path.exists():
                continue
            trial_manifest = _load_json(manifest_path)
            stress_case_dir = _stress_case_dir_from_trial_manifest(trial_manifest)
            if stress_case_dir is None:
                continue
            if _build_plot_with_bg(
                repo_root=repo_root,
                python_exec=python_exec,
                eval_config=eval_config,
                run_dir=run_dir,
                stress_case_dir=stress_case_dir,
                trial_manifest=trial_manifest,
            ) is not None:
                rendered += 1
            if _build_compare_image(run_dir) is not None:
                compared += 1

        summary_refined = _build_summary(
            run_root=run_root,
            image_name="plot_with_bg_refined.png",
            out_name=f"plot_with_bg_refined_summary_{method_dir_name.replace('_run', '')}.png",
            title=f"Clutter-assisted refined plot_with_bg: {method_dir_name.replace('_run', '')}",
            variant_order=VARIANT_ORDER_ALL,
        )
        summary_refined_active = _build_summary(
            run_root=run_root,
            image_name="plot_with_bg_refined.png",
            out_name=f"plot_with_bg_refined_summary_{method_dir_name.replace('_run', '')}_active.png",
            title=f"Clutter-assisted refined plot_with_bg (active variants): {method_dir_name.replace('_run', '')}",
            variant_order=VARIANT_ORDER_ACTIVE,
        )
        summary_compare_active = _build_summary(
            run_root=run_root,
            image_name="plot_with_bg_compare.png",
            out_name=f"plot_with_bg_compare_summary_{method_dir_name.replace('_run', '')}_active.png",
            title=f"Clutter-assisted before/after compare (active variants): {method_dir_name.replace('_run', '')}",
            variant_order=VARIANT_ORDER_ACTIVE,
        )

        render_manifest["methods"][method_dir_name] = {
            "rendered_plot_with_bg_refined_count": rendered,
            "rendered_compare_count": compared,
            "summary_refined": str(summary_refined) if summary_refined is not None else None,
            "summary_refined_active": str(summary_refined_active) if summary_refined_active is not None else None,
            "summary_compare_active": str(summary_compare_active) if summary_compare_active is not None else None,
        }

    out_path = compare_root / "plot_with_bg_render_manifest.json"
    out_path.write_text(json.dumps(render_manifest, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    print(json.dumps(render_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
