from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
from typing import Any, Dict, List, Optional, Sequence


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def git_head_sha(repo_root: pathlib.Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _clearance_value(metrics: Dict[str, Any]) -> float:
    for key in ("clr_feasible", "clr_min_astar", "clr_min"):
        try:
            if key in metrics:
                return float(metrics.get(key, 0.0))
        except Exception:
            continue
    return 0.0


def _entry_gate_cfg(eval_cfg: Dict[str, Any]) -> Dict[str, Any]:
    adopt_cfg = eval_cfg.get("adopt") if isinstance(eval_cfg.get("adopt"), dict) else {}
    gate = adopt_cfg.get("entry_gate") if isinstance(adopt_cfg.get("entry_gate"), dict) else {}
    return gate


def classify_difficulty(base_metrics: Dict[str, Any], metrics_after: Dict[str, Any], eval_cfg: Dict[str, Any]) -> str:
    tau_r = float(eval_cfg.get("tau_R", 0.0))
    tau_v = float(eval_cfg.get("tau_V", 0.0))
    tau_clr = float(eval_cfg.get("tau_clr_feasible", eval_cfg.get("tau_clr", 0.0)))
    gate = _entry_gate_cfg(eval_cfg)
    gate_enabled = bool(gate.get("enabled", False))
    gate_metric = str(gate.get("metric") or "OOE_R_rec_entry_surf")
    gate_min = float(gate.get("min_value", 0.0))

    hard = False
    if int(metrics_after.get("validity", 0)) != 1:
        hard = True
    if float(metrics_after.get("R_reach", 0.0)) + 1e-9 < tau_r:
        hard = True
    if _clearance_value(metrics_after) + 1e-9 < tau_clr:
        hard = True
    if float(metrics_after.get("C_vis", 0.0)) + 1e-9 < tau_v:
        hard = True
    if gate_enabled and float(metrics_after.get(gate_metric, 0.0)) + 1e-9 < gate_min:
        hard = True
    if int(base_metrics.get("Adopt_core", 1)) == 1 and int(metrics_after.get("Adopt_core", 1)) == 0:
        hard = True
    if int(base_metrics.get("Adopt_entry", 1)) == 1 and int(metrics_after.get("Adopt_entry", 1)) == 0:
        hard = True
    if hard:
        return "hard_recoverable"

    borderline = False
    if float(metrics_after.get("R_reach", 0.0)) < tau_r + 0.05:
        borderline = True
    if _clearance_value(metrics_after) < tau_clr + 0.03:
        borderline = True
    if float(metrics_after.get("C_vis", 0.0)) < tau_v + 0.05:
        borderline = True
    if gate_enabled and float(metrics_after.get(gate_metric, 0.0)) < gate_min + 0.05:
        borderline = True
    if float(metrics_after.get("C_vis_start", 0.0)) < float(base_metrics.get("C_vis_start", 0.0)) - 0.08:
        borderline = True
    return "borderline" if borderline else "mild"


def metric_deltas(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, float]:
    keys = sorted(set(before.keys()) | set(after.keys()))
    out: Dict[str, float] = {}
    for key in keys:
        try:
            out[key] = float(after.get(key, 0.0)) - float(before.get(key, 0.0))
        except Exception:
            continue
    return out


def build_sample_manifest(
    *,
    stress_cfg: Dict[str, Any],
    stress_config_path: pathlib.Path,
    stress_config_hash: str,
    repo_root: pathlib.Path,
    base_case_id: str,
    scene_id: str,
    stress_family: str,
    seed: int,
    base_metrics: Dict[str, Any],
    variant_metrics: Dict[str, Any],
    disturb: Dict[str, Any],
    eval_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    actions = disturb.get("actions") if isinstance(disturb.get("actions"), list) else []
    moved_objects: List[Dict[str, Any]] = []
    added_clutter: List[Dict[str, Any]] = []
    delta_movable = 0.0
    added_clutter_area = 0.0

    for action in actions:
        if not isinstance(action, dict):
            continue
        atype = str(action.get("type") or "")
        if atype == "add_clutter":
            size_xy = action.get("size_xy_m") or [0.0, 0.0]
            pose = action.get("pose_xytheta") or [0.0, 0.0, 0.0]
            size_x = float(size_xy[0]) if len(size_xy) >= 1 else 0.0
            size_y = float(size_xy[1]) if len(size_xy) >= 2 else 0.0
            added_clutter.append(
                {
                    "clutter_id": str(action.get("object_id") or ""),
                    "size_xy": [size_x, size_y],
                    "height": float(action.get("height_m", 0.0)),
                    "pose": [float(pose[0]), float(pose[1]), float(pose[2])],
                    "band_id": str(action.get("band_id") or ""),
                }
            )
            added_clutter_area += size_x * size_y
            continue

        moved = {
            "object_id": str(action.get("object_id") or ""),
            "category": str(action.get("category") or ""),
            "dx_local": float(action.get("dx_local", 0.0)),
            "dy_local": float(action.get("dy_local", 0.0)),
            "dtheta_deg": float(action.get("dtheta_deg", action.get("delta_yaw_deg", 0.0))),
            "same_room_ok": bool(action.get("same_room_ok", True)),
        }
        moved_objects.append(moved)
        delta_movable += abs(moved["dx_local"]) + abs(moved["dy_local"])

    disturb_summary = {
        "Delta_movable": delta_movable,
        "num_added_clutter": len(added_clutter),
        "added_clutter_area": added_clutter_area,
        "compound": bool(stress_family == "compound"),
        "legacy_Delta_disturb": float(disturb.get("Delta_disturb", 0.0)),
    }

    validity_checks = disturb.get("validity_checks") if isinstance(disturb.get("validity_checks"), dict) else {}
    rejection_summary = disturb.get("rejection_summary") if isinstance(disturb.get("rejection_summary"), dict) else {}
    pool_size = int(disturb.get("pool_size", 1) or 1)
    selected_from_pool_index = int(disturb.get("selected_from_pool_index", 0) or 0)
    selection_notes = str(disturb.get("selection_notes") or "")
    if not selection_notes:
        if stress_family == "base":
            selection_notes = "No perturbation applied."
        else:
            selection_notes = f"Selected candidate {selected_from_pool_index} from pool of {pool_size}."

    return {
        "stress_version": str(stress_cfg.get("stress_version") or "unknown"),
        "stress_family": stress_family,
        "base_case_id": base_case_id,
        "scene_id": scene_id,
        "seed": int(seed),
        "config_path": str(stress_config_path),
        "config_hash": stress_config_hash,
        "generator_commit": git_head_sha(repo_root),
        "selected_from_pool_index": selected_from_pool_index,
        "pool_size": pool_size,
        "validity_checks": validity_checks,
        "rejection_summary": rejection_summary,
        "moved_objects": moved_objects,
        "added_clutter": added_clutter,
        "metrics_before": base_metrics,
        "metrics_after": variant_metrics,
        "metric_deltas": metric_deltas(base_metrics, variant_metrics),
        "difficulty_label": classify_difficulty(base_metrics, variant_metrics, eval_cfg),
        "selection_notes": selection_notes,
        "disturb_summary": disturb_summary,
        "stress_config_name": stress_config_path.name,
    }
