from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _find_object(layout: Dict[str, Any], object_id: str) -> Optional[Dict[str, Any]]:
    for obj in layout.get("objects", []):
        if str(obj.get("id") or "") == object_id:
            return obj
    return None


def _pose_xy(obj: Dict[str, Any]) -> Tuple[float, float]:
    pose = obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]
    x = float(pose[0]) if len(pose) >= 1 else 0.0
    y = float(pose[1]) if len(pose) >= 2 else 0.0
    return x, y


def _obb_corners_xy(obj: Dict[str, Any]) -> List[Tuple[float, float]]:
    pose = obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]
    dims = obj.get("dimensions_xyz") or [0.0, 0.0, 0.0]
    cx = float(pose[0]) if len(pose) >= 1 else 0.0
    cy = float(pose[1]) if len(pose) >= 2 else 0.0
    yaw = float(pose[3]) if len(pose) >= 4 else 0.0
    lx = float(dims[0]) if len(dims) >= 1 else 0.0
    ly = float(dims[1]) if len(dims) >= 2 else 0.0
    hx = 0.5 * lx
    hy = 0.5 * ly
    cs = math.cos(yaw)
    sn = math.sin(yaw)
    local = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    out: List[Tuple[float, float]] = []
    for px, py in local:
        wx = cx + px * cs - py * sn
        wy = cy + px * sn + py * cs
        out.append((wx, wy))
    return out


def _moved_objects(before: Dict[str, Any], after: Dict[str, Any]) -> List[Dict[str, Any]]:
    moved: List[Dict[str, Any]] = []
    for before_obj in before.get("objects", []):
        oid = str(before_obj.get("id") or "")
        if not oid:
            continue
        after_obj = _find_object(after, oid)
        if after_obj is None:
            continue
        bx, by = _pose_xy(before_obj)
        ax, ay = _pose_xy(after_obj)
        disp = math.hypot(ax - bx, ay - by)
        if disp <= 1e-9:
            continue
        moved.append(
            {
                "id": oid,
                "category": str(after_obj.get("category") or before_obj.get("category") or ""),
                "before": before_obj,
                "after": after_obj,
                "disp": disp,
            }
        )
    moved.sort(key=lambda item: item["id"])
    return moved


def _room_boundary(layout: Dict[str, Any]) -> List[Tuple[float, float]]:
    room = layout.get("room") or {}
    poly = room.get("boundary_poly_xy") or []
    out: List[Tuple[float, float]] = []
    for pt in poly:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            out.append((float(pt[0]), float(pt[1])))
    return out


def _bounds_from_room(poly: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = max(0.25, 0.03 * max(1e-6, max_x - min_x))
    pad_y = max(0.25, 0.03 * max(1e-6, max_y - min_y))
    return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y


def _render_overlay(run_dir: pathlib.Path) -> Optional[pathlib.Path]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except Exception:
        return None

    trial_manifest = _load_json(run_dir / "trial_manifest.json")
    trial_cfg = trial_manifest.get("trial_config") if isinstance(trial_manifest.get("trial_config"), dict) else {}
    layout_in_path = pathlib.Path(str(((trial_cfg.get("inputs") or {}).get("layout_input")) or ""))
    if not layout_in_path.exists():
        return None

    before = _load_json(layout_in_path)
    after_path = run_dir / "layout_refined.json"
    if not after_path.exists():
        return None
    after = _load_json(after_path)
    moved = _moved_objects(before, after)
    room_poly = _room_boundary(after if _room_boundary(after) else before)
    if not room_poly:
        return None

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    room_patch = Polygon(room_poly, closed=True, fill=False, edgecolor="black", linewidth=2.0, zorder=1)
    ax.add_patch(room_patch)

    for obj in before.get("objects", []):
        cat = str(obj.get("category") or "").strip().lower()
        if cat == "floor":
            continue
        poly = _obb_corners_xy(obj)
        patch = Polygon(poly, closed=True, fill=False, edgecolor="#BBBBBB", linewidth=0.8, alpha=0.8, zorder=2)
        ax.add_patch(patch)

    if moved:
        for item in moved:
            before_obj = item["before"]
            after_obj = item["after"]
            bpoly = _obb_corners_xy(before_obj)
            apoly = _obb_corners_xy(after_obj)
            bx, by = _pose_xy(before_obj)
            axp, ayp = _pose_xy(after_obj)
            ax.add_patch(Polygon(bpoly, closed=True, fill=False, edgecolor="#D55E00", linewidth=2.0, linestyle="--", zorder=4))
            ax.add_patch(Polygon(apoly, closed=True, fill=False, edgecolor="#009E73", linewidth=2.2, zorder=5))
            ax.scatter([bx], [by], color="#D55E00", s=20, zorder=6)
            ax.scatter([axp], [ayp], color="#009E73", s=20, zorder=6)
            ax.annotate(
                "",
                xy=(axp, ayp),
                xytext=(bx, by),
                arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#0072B2"},
                zorder=6,
            )
            ax.text(
                axp,
                ayp,
                f"{item['id']}\n{item['disp']:.2f}m",
                fontsize=8,
                ha="left",
                va="bottom",
                color="#111111",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.0},
                zorder=7,
            )
    else:
        ax.text(0.5, 0.5, "no moved objects", transform=ax.transAxes, ha="center", va="center", fontsize=14)

    min_x, max_x, min_y, max_y = _bounds_from_room(room_poly)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    summary = trial_manifest.get("summary") if isinstance(trial_manifest.get("summary"), dict) else {}
    title = (
        f"{summary.get('layout_id', run_dir.name)} / {summary.get('method', '')} / "
        f"{summary.get('recovery_protocol', '')}\n"
        f"moved={len(moved)} clutter_disp={float(summary.get('delta_layout_clutter', 0.0) or 0.0):.3f}m"
    )
    ax.set_title(title, fontsize=10)

    out_path = run_dir / "movement_overlay.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _build_method_summary(run_root: pathlib.Path, *, out_name: str) -> Optional[pathlib.Path]:
    try:
        from PIL import Image
        import matplotlib.pyplot as plt
    except Exception:
        return None

    variant_order = ["base", "usage_shift", "clutter", "compound"]
    items: Dict[str, Dict[str, pathlib.Path]] = {}
    case_order: List[str] = []
    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "trial_manifest.json"
        image_path = run_dir / "movement_overlay.png"
        if not manifest_path.exists() or not image_path.exists():
            continue
        manifest = _load_json(manifest_path)
        summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
        trial_id = str(summary.get("trial_id") or run_dir.name)
        parts = trial_id.split("__")
        if len(parts) < 2:
            continue
        case_id = str(summary.get("layout_id") or parts[0])
        variant = parts[1]
        if case_id not in items:
            items[case_id] = {}
            case_order.append(case_id)
        items[case_id][variant] = run_dir

    if not case_order:
        return None

    rows = len(variant_order)
    cols = len(case_order)
    fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 5.0 * rows), dpi=160)
    if rows == 1 and cols == 1:
        grid = [[axes]]
    elif rows == 1:
        grid = [list(axes)]
    elif cols == 1:
        grid = [[ax] for ax in axes]
    else:
        grid = axes

    for r, variant in enumerate(variant_order):
        for c, case_id in enumerate(case_order):
            ax = grid[r][c]
            run_dir = items.get(case_id, {}).get(variant)
            if run_dir is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=12)
                ax.axis("off")
                if c == 0:
                    ax.set_ylabel(variant, fontsize=10)
                continue
            img = Image.open(run_dir / "movement_overlay.png").convert("RGB")
            ax.imshow(img)
            ax.axis("off")
            if r == 0:
                ax.set_title(case_id, fontsize=9)
            if c == 0:
                ax.set_ylabel(variant, fontsize=10)

    method = run_root.name
    fig.suptitle(f"Movement overlay summary: {method}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = run_root.parent / out_name
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render moved-position overlays for clutter-assisted compare runs.")
    parser.add_argument("--compare_root", required=True)
    args = parser.parse_args()

    repo_root = _repo_root()
    compare_root = pathlib.Path(args.compare_root)
    if not compare_root.is_absolute():
        compare_root = (repo_root / compare_root).resolve()

    render_manifest: Dict[str, Any] = {"compare_root": str(compare_root), "methods": {}}
    for method_dir_name in ("heuristic_run", "proposed_run"):
        run_root = compare_root / method_dir_name
        if not run_root.exists():
            continue
        count = 0
        for run_dir in sorted(run_root.iterdir()):
            if not run_dir.is_dir():
                continue
            if _render_overlay(run_dir) is not None:
                count += 1
        summary_path = _build_method_summary(run_root, out_name=f"movement_overlay_summary_{method_dir_name.replace('_run', '')}.png")
        render_manifest["methods"][method_dir_name] = {
            "overlay_count": count,
            "summary_path": str(summary_path) if summary_path is not None else None,
        }

    out_path = compare_root / "movement_overlay_render_manifest.json"
    out_path.write_text(json.dumps(render_manifest, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    print(json.dumps(render_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
