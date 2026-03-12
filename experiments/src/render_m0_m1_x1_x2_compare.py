from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
from typing import Any, Dict, List, Optional


PROTOCOL_ORDER = ["M0", "M1", "X1", "X2"]
ACTIVE_VARIANTS = ["clutter", "compound"]
METHODS = ["heuristic", "proposed"]


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _run(cmd: List[str], cwd: pathlib.Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_index(path: pathlib.Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _stress_case_dir_from_row(row: Dict[str, Any]) -> Optional[pathlib.Path]:
    layout_path = pathlib.Path(str(row.get("layout_path") or ""))
    if row.get("protocol") == "M0":
        return layout_path.parent if layout_path.exists() else None
    stress_manifest_path = pathlib.Path(str(row.get("stress_manifest_path") or ""))
    return stress_manifest_path.parent if stress_manifest_path.exists() else None


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


def _source_bg_image(stress_case_dir: pathlib.Path, source_layout_path: Optional[pathlib.Path]) -> Optional[pathlib.Path]:
    stress_manifest = stress_case_dir / "stress_manifest.json"
    if stress_manifest.exists():
        try:
            sm = _load_json(stress_manifest)
            raw_bg = str(sm.get("source_image_path") or "").strip()
            bg_image = pathlib.Path(raw_bg) if raw_bg else None
            if bg_image is not None and bg_image.exists() and bg_image.is_file():
                return bg_image
        except Exception:
            pass
    if source_layout_path is None:
        return None
    generation_manifest = source_layout_path.parent / "generation_manifest.json"
    if not generation_manifest.exists():
        return None
    try:
        gm = _load_json(generation_manifest)
    except Exception:
        return None
    inputs = gm.get("inputs") if isinstance(gm.get("inputs"), dict) else {}
    raw_image = str(inputs.get("image_path") or "").strip()
    if not raw_image:
        return None
    image_path = pathlib.Path(raw_image)
    return image_path if image_path.exists() and image_path.is_file() else None


def _ensure_refined_plot(
    *,
    repo_root: pathlib.Path,
    python_exec: pathlib.Path,
    eval_config: pathlib.Path,
    row: Dict[str, Any],
) -> Optional[pathlib.Path]:
    if row.get("protocol") == "M0":
        plot_path = pathlib.Path(str(row.get("plot_path") or ""))
        return plot_path if plot_path.exists() else None

    run_dir = pathlib.Path(str(row.get("run_dir") or ""))
    refined_layout = run_dir / "layout_refined.json"
    if not refined_layout.exists():
        return None

    stress_case_dir = _stress_case_dir_from_row(row)
    if stress_case_dir is None:
        return None
    source_layout = _source_layout_path(stress_case_dir)
    bg_image = _source_bg_image(stress_case_dir, source_layout)
    bg_inner_frame = source_layout.parent / "gemini_room_inner_frame_output.json" if source_layout is not None else None

    debug_dir = run_dir / "debug_refined"
    metrics_json = run_dir / "metrics_refined_eval.json"
    plot_png = run_dir / "plot_with_bg_refined.png"
    if plot_png.exists():
        return plot_png

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


def _summary_matrix(
    *,
    rows: List[Dict[str, Any]],
    method: str,
    variant: str,
    out_path: pathlib.Path,
    title: str,
) -> Optional[pathlib.Path]:
    try:
        from PIL import Image
        import matplotlib.pyplot as plt
    except Exception:
        return None

    row_map = {}
    case_order: List[str] = []
    for row in rows:
        if row.get("variant") != variant:
            continue
        protocol = row.get("protocol")
        row_method = row.get("method")
        if protocol == "M0":
            key = (row.get("base_case"), protocol)
        elif row_method == method:
            key = (row.get("base_case"), protocol)
        else:
            continue
        row_map[key] = row
        case_id = str(row.get("base_case") or "")
        if case_id and case_id not in case_order:
            case_order.append(case_id)

    case_order = sorted(case_order)
    if not case_order:
        return None

    fig, axes = plt.subplots(len(PROTOCOL_ORDER), len(case_order), figsize=(5.8 * len(case_order), 4.8 * len(PROTOCOL_ORDER)), dpi=160)
    for r_idx, protocol in enumerate(PROTOCOL_ORDER):
        for c_idx, case_id in enumerate(case_order):
            ax = axes[r_idx][c_idx]
            row = row_map.get((case_id, protocol))
            if row is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=12)
                ax.axis("off")
                continue
            img_path = pathlib.Path(str(row.get("rendered_plot_path") or row.get("plot_path") or ""))
            if not img_path.exists():
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=12)
                ax.axis("off")
                continue
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            ax.axis("off")
            if r_idx == 0:
                ax.set_title(case_id, fontsize=9)
            if c_idx == 0:
                label = protocol
                if protocol == "M0":
                    label = "M0\nbaseline"
                elif protocol == "M1":
                    label = "M1\nfurniture"
                elif protocol == "X1":
                    label = "X1\nclutter"
                elif protocol == "X2":
                    label = "X2\nfurniture->clutter"
                ax.set_ylabel(label, fontsize=10)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _summary_md(compare_root: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# M0 / M1 / X1 / X2 visualization report")
    lines.append("")
    lines.append(f"- compare root: `{compare_root}`")
    lines.append("- active variants:")
    lines.append("  - `clutter`")
    lines.append("  - `compound`")
    lines.append("")
    for method in METHODS:
        lines.append(f"## {method}")
        lines.append("")
        for variant in ACTIVE_VARIANTS:
            img = compare_root / f"protocol_compare_summary_{method}_{variant}.png"
            lines.append(f"### {variant}")
            lines.append("")
            lines.append(f"![]({img.name})")
            lines.append("")
    (compare_root / "protocol_compare_visual_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render M0/M1/X1/X2 compare summaries.")
    parser.add_argument("--compare_root", default="experiments/runs/m0_m1_x1_x2_compare_from_latest_design_frozen_20260312")
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

    rows = _load_index(compare_root / "protocol_compare_index.csv")
    render_manifest: Dict[str, Any] = {
        "compare_root": str(compare_root),
        "eval_config": str(eval_config),
        "entries": [],
        "summaries": [],
    }

    for row in rows:
        plot_path = _ensure_refined_plot(repo_root=repo_root, python_exec=python_exec, eval_config=eval_config, row=row)
        row["rendered_plot_path"] = str(plot_path) if plot_path is not None else ""
        render_manifest["entries"].append(
            {
                "base_case": row.get("base_case"),
                "variant": row.get("variant"),
                "protocol": row.get("protocol"),
                "method": row.get("method"),
                "rendered_plot_path": row.get("rendered_plot_path"),
            }
        )

    for method in METHODS:
        for variant in ACTIVE_VARIANTS:
            out_path = compare_root / f"protocol_compare_summary_{method}_{variant}.png"
            built = _summary_matrix(
                rows=rows,
                method=method,
                variant=variant,
                out_path=out_path,
                title=f"{method} - {variant} (rows: M0/M1/X1/X2, cols: 5 cases)",
            )
            if built is not None:
                render_manifest["summaries"].append({"method": method, "variant": variant, "path": str(built)})

    _summary_md(compare_root, rows)
    (compare_root / "protocol_compare_render_manifest.json").write_text(
        json.dumps(render_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
