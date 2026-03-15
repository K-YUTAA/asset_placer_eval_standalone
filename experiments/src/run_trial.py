from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from argparse import Namespace
from datetime import datetime
from typing import Any, Dict

from eval_metrics import evaluate_layout, merge_eval_config
from layout_tools import load_layout_contract, write_json
from refine_clutter_recovery import run_refinement as run_refinement_clutter
from refine_heuristic import run_refinement as run_refinement_heuristic
from refine_proposed_beam import run_refinement as run_refinement_proposed
from run_v0_freeze import run_v0_freeze


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _default_eval_config_path() -> pathlib.Path:
    return _repo_root() / "experiments" / "configs" / "eval" / "eval_v1.json"


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        t = value.strip().lower()
        if t in {"1", "true", "yes", "on"}:
            return True
        if t in {"0", "false", "no", "off"}:
            return False
    return default


def _load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_payload(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _object_index(layout: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for obj in layout.get("objects", []):
        oid = str(obj.get("id") or "")
        if oid:
            out[oid] = obj
    return out


def _pose_xy(obj: Dict[str, Any]) -> tuple[float, float]:
    pose = obj.get("pose_xyz_yaw") or [0.0, 0.0, 0.0, 0.0]
    x = float(pose[0]) if len(pose) >= 1 else 0.0
    y = float(pose[1]) if len(pose) >= 2 else 0.0
    return x, y


def _movement_breakdown(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    import math

    before_idx = _object_index(before)
    after_idx = _object_index(after)
    delta_layout_furniture = 0.0
    delta_layout_clutter = 0.0
    moved_furniture_count = 0
    moved_clutter_count = 0
    sum_furniture_displacement_m = 0.0
    sum_clutter_displacement_m = 0.0
    max_clutter_displacement_m = 0.0

    for oid, before_obj in before_idx.items():
        after_obj = after_idx.get(oid)
        if after_obj is None:
            continue
        bx, by = _pose_xy(before_obj)
        ax, ay = _pose_xy(after_obj)
        disp = math.hypot(ax - bx, ay - by)
        if disp <= 1e-9:
            continue
        cat = str(after_obj.get("category") or before_obj.get("category") or "").strip().lower()
        if cat == "clutter" or bool(after_obj.get("movable_in_clutter_recovery", False)):
            moved_clutter_count += 1
            sum_clutter_displacement_m += disp
            max_clutter_displacement_m = max(max_clutter_displacement_m, disp)
            delta_layout_clutter += disp
        else:
            moved_furniture_count += 1
            sum_furniture_displacement_m += disp
            delta_layout_furniture += disp

    return {
        "delta_layout_furniture": delta_layout_furniture,
        "delta_layout_clutter": delta_layout_clutter,
        "moved_furniture_count": moved_furniture_count,
        "moved_clutter_count": moved_clutter_count,
        "sum_furniture_displacement_m": sum_furniture_displacement_m,
        "sum_clutter_displacement_m": sum_clutter_displacement_m,
        "max_clutter_displacement_m": max_clutter_displacement_m,
    }


def _resolve_method_spec(trial_cfg: Dict[str, Any], method: str) -> Dict[str, Any]:
    recovery_protocol = str(trial_cfg.get("recovery_protocol", "layout_only"))
    if recovery_protocol == "clutter_assisted":
        defaults_path = _repo_root() / "experiments" / "configs" / "refine" / "clutter_assisted_v1.json"
        defaults = _load_json(defaults_path) if defaults_path.exists() else {}
        heuristic_defaults = defaults.get("heuristic") if isinstance(defaults.get("heuristic"), dict) else {}
        proposed_defaults = defaults.get("proposed") if isinstance(defaults.get("proposed"), dict) else {}
        common = {
            "method": method,
            "recovery_protocol": recovery_protocol,
            "refine_clutter_grid_step_m": float(trial_cfg.get("refine_clutter_grid_step_m", defaults.get("grid_step_m", 0.2))),
            "refine_clutter_candidate_limit": int(trial_cfg.get("refine_clutter_candidate_limit", defaults.get("candidate_limit_per_object", 28))),
            "refine_door_keepout_radius_m": float(trial_cfg.get("refine_door_keepout_radius_m", defaults.get("door_keepout_radius_m", 0.5))),
            "refine_overlap_ratio_max": float(trial_cfg.get("refine_overlap_ratio_max", defaults.get("overlap_ratio_max", 0.05))),
            "refine_delta_weight": float(trial_cfg.get("refine_delta_weight", defaults.get("delta_weight", 0.02))),
            "refine_max_changed_objects": int(trial_cfg.get("refine_max_changed_objects", 3)),
            "refine_ooe_primary": str(trial_cfg.get("refine_ooe_primary", "OOE_R_rec_entry_surf")),
        }
        if method == "proposed":
            return {
                **common,
                "refine_beam_width": int(trial_cfg.get("refine_beam_width", proposed_defaults.get("beam_width", 6))),
                "refine_depth": int(trial_cfg.get("refine_depth", proposed_defaults.get("depth", 3))),
            }
        if method == "heuristic":
            return {
                **common,
                "refine_max_iterations": int(trial_cfg.get("refine_max_iterations", heuristic_defaults.get("max_iterations", 4))),
            }
        return common
    if method == "heuristic":
        return {
            "method": "heuristic",
            "recovery_protocol": recovery_protocol,
            "refine_max_iterations": int(trial_cfg.get("refine_max_iterations", 30)),
            "refine_step_m": float(trial_cfg.get("refine_step_m", 0.1)),
            "refine_rot_deg": float(trial_cfg.get("refine_rot_deg", 15.0)),
            "refine_max_changed_objects": int(trial_cfg.get("refine_max_changed_objects", 3)),
        }
    if method == "proposed":
        return {
            "method": "proposed",
            "recovery_protocol": recovery_protocol,
            "refine_step_m": float(trial_cfg.get("refine_step_m", 0.1)),
            "refine_rot_deg": float(trial_cfg.get("refine_rot_deg", 15.0)),
            "refine_max_changed_objects": int(trial_cfg.get("refine_max_changed_objects", 3)),
            "refine_beam_width": int(trial_cfg.get("refine_beam_width", 5)),
            "refine_depth": int(trial_cfg.get("refine_depth", 3)),
            "refine_candidate_objects_per_state": int(trial_cfg.get("refine_candidate_objects_per_state", 2)),
            "refine_eval_budget": int(trial_cfg.get("refine_eval_budget", 0)),
            "refine_ooe_primary": str(trial_cfg.get("refine_ooe_primary", "OOE_R_rec_entry_surf")),
            "refine_use_lexicographic": _as_bool(trial_cfg.get("refine_use_lexicographic"), True),
            "refine_allow_intermediate_regression": _as_bool(trial_cfg.get("refine_allow_intermediate_regression"), True),
            "refine_door_keepout_radius_m": float(trial_cfg.get("refine_door_keepout_radius_m", 0.0)),
            "refine_overlap_ratio_max": float(trial_cfg.get("refine_overlap_ratio_max", 0.05)),
            "refine_delta_weight": float(trial_cfg.get("refine_delta_weight", 0.3)),
        }
    return {"method": method}


def _append_metrics_csv(csv_path: pathlib.Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_id",
        "layout_id",
        "method",
        "recovery_protocol",
        "eval_config_path",
        "eval_config_name",
        "eval_hash",
        "method_hash",
        "seed",
        "status",
        "error_msg",
        "C_vis",
        "R_reach",
        "clr_min",
        "clr_min_astar",
        "clr_feasible",
        "Delta_layout",
        "Adopt",
        "Adopt_core",
        "Adopt_entry",
        "Adopt_clearance_metric",
        "Adopt_clearance_value",
        "Adopt_clearance_threshold",
        "Adopt_entry_gate_enabled",
        "Adopt_entry_metric",
        "Adopt_entry_metric_value",
        "Adopt_entry_min_value",
        "validity",
        "C_vis_start",
        "OOE_C_obj_entry_hit",
        "OOE_R_rec_entry_hit",
        "OOE_C_obj_entry_surf",
        "OOE_R_rec_entry_surf",
        "delta_layout_furniture",
        "delta_layout_clutter",
        "moved_furniture_count",
        "moved_clutter_count",
        "sum_furniture_displacement_m",
        "sum_clutter_displacement_m",
        "max_clutter_displacement_m",
        "runtime_sec",
        "run_dir",
    ]

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_trial(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = _repo_root()
    trial_cfg = _load_json(pathlib.Path(args.trial_config))

    trial_id = str(trial_cfg.get("trial_id") or f"trial_{_timestamp()}")
    layout_id = str(trial_cfg.get("layout_id") or "layout")
    method = str(trial_cfg.get("method") or "original")
    recovery_protocol = str(trial_cfg.get("recovery_protocol") or "layout_only")
    seed = int(trial_cfg.get("seed", 0))

    run_dir = pathlib.Path(args.out_root) / f"{trial_id}_{_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    eval_config_path = pathlib.Path(args.eval_config or str(_default_eval_config_path()))
    if not eval_config_path.is_absolute():
        eval_config_path = (repo_root / eval_config_path).resolve()
    eval_config_raw = _load_json(eval_config_path)
    eval_config_name = eval_config_path.name
    eval_hash = _sha256_file(eval_config_path)
    cfg = merge_eval_config(eval_config_raw, trial_cfg.get("eval", {}))
    cfg["recovery_protocol"] = recovery_protocol

    method_spec = _resolve_method_spec(trial_cfg, method)
    method_hash = _sha256_payload(method_spec)
    method_config_path = None
    method_config_sha256 = None
    alignment_prior_config_path = None
    alignment_prior_config_sha256 = None
    if recovery_protocol == "clutter_assisted":
        clutter_cfg_path = repo_root / "experiments" / "configs" / "refine" / "clutter_assisted_v1.json"
        if clutter_cfg_path.exists():
            method_config_path = str(clutter_cfg_path)
            method_config_sha256 = _sha256_file(clutter_cfg_path)
    elif method == "proposed":
        proposed_cfg_path = repo_root / "experiments" / "configs" / "refine" / "proposed_beam_v1.json"
        if proposed_cfg_path.exists():
            method_config_path = str(proposed_cfg_path)
            method_config_sha256 = _sha256_file(proposed_cfg_path)

    alignment_prior_path_raw = trial_cfg.get("layout_axis_alignment_prior_path")
    if alignment_prior_path_raw:
        alignment_prior_path = pathlib.Path(str(alignment_prior_path_raw))
        if not alignment_prior_path.is_absolute():
            alignment_prior_path = (repo_root / alignment_prior_path).resolve()
        if alignment_prior_path.exists():
            alignment_prior_payload = _load_json(alignment_prior_path)
            prior_section = alignment_prior_payload.get("layout_axis_alignment_prior")
            cfg["layout_axis_alignment_prior"] = prior_section if isinstance(prior_section, dict) else alignment_prior_payload
            alignment_prior_config_path = str(alignment_prior_path)
            alignment_prior_config_sha256 = _sha256_file(alignment_prior_path)

    inputs = trial_cfg.get("inputs", {})

    try:
        freeze_args = Namespace(
            sketch_path=str(inputs.get("sketch_path", "")),
            hints_path=str(inputs.get("hints_path", "")),
            seed=seed,
            out_dir=str(run_dir / "v0"),
            llm_cache_mode=str(trial_cfg.get("llm_cache_mode", "write")),
            layout_input=inputs.get("layout_input"),
            layout_id=layout_id,
            model=str(trial_cfg.get("model", "gpt-5.2")),
            prompt_1_name=str(trial_cfg.get("prompt_1_name", "prompt_1")),
            prompt_2_name=str(trial_cfg.get("prompt_2_name", trial_cfg.get("prompt_name", "prompt_2"))),
            prompt_name=str(trial_cfg.get("prompt_name", "prompt_2")),
            reasoning=str(trial_cfg.get("reasoning", "high")),
            temperature=float(trial_cfg.get("temperature", 0.0)),
            top_p=float(trial_cfg.get("top_p", 1.0)),
            grid_resolution=float(cfg.get("grid_resolution_m", 0.1)),
            robot_radius=float(cfg.get("robot_radius_m", 0.3)),
            start_x=float((cfg.get("start_xy") or [0.8, 0.8])[0]),
            start_y=float((cfg.get("start_xy") or [0.8, 0.8])[1]),
            goal_x=float((cfg.get("goal_xy") or [5.0, 5.0])[0]),
            goal_y=float((cfg.get("goal_xy") or [5.0, 5.0])[1]),
            max_iterations=int(trial_cfg.get("placement_max_iterations", 30)),
            push_step=float(trial_cfg.get("placement_push_step", 0.1)),
            placement_order=str(trial_cfg.get("placement_order", "category_then_area")),
        )

        v0_outputs = run_v0_freeze(freeze_args)
        layout_v0 = load_layout_contract(pathlib.Path(v0_outputs["layout_v0"]))

        baseline_layout = layout_v0
        selected_layout = layout_v0
        refine_log = None
        movement_breakdown = {
            "delta_layout_furniture": 0.0,
            "delta_layout_clutter": 0.0,
            "moved_furniture_count": 0,
            "moved_clutter_count": 0,
            "sum_furniture_displacement_m": 0.0,
            "sum_clutter_displacement_m": 0.0,
            "max_clutter_displacement_m": 0.0,
        }
        baseline_task_points = None
        if isinstance(cfg.get("task"), dict) and cfg.get("task"):
            _, baseline_debug = evaluate_layout(layout_v0, baseline_layout, cfg)
            baseline_task_points = baseline_debug.get("task_points")
            if isinstance(baseline_task_points, dict):
                cfg["start_xy"] = (baseline_task_points.get("start") or {}).get("xy") or cfg.get("start_xy")
                cfg["goal_xy"] = (baseline_task_points.get("goal") or {}).get("xy") or cfg.get("goal_xy")

        if recovery_protocol == "clutter_assisted" and method in {"heuristic", "proposed"}:
            refined_layout, refined_metrics, refine_steps = run_refinement_clutter(
                layout=layout_v0,
                baseline_layout=baseline_layout,
                config=cfg,
                method=method,
                max_iterations=int(method_spec.get("refine_max_iterations", 4)),
                max_changed_objects=int(method_spec.get("refine_max_changed_objects", 3)),
                grid_step_m=float(method_spec.get("refine_clutter_grid_step_m", 0.2)),
                candidate_limit=int(method_spec.get("refine_clutter_candidate_limit", 28)),
                door_keepout_radius_m=float(method_spec.get("refine_door_keepout_radius_m", 0.5)),
                overlap_ratio_max=float(method_spec.get("refine_overlap_ratio_max", 0.05)),
                delta_weight=float(method_spec.get("refine_delta_weight", 0.02)),
                beam_width=int(method_spec.get("refine_beam_width", 6)),
                depth=int(method_spec.get("refine_depth", 3)),
                ooe_primary=str(method_spec.get("refine_ooe_primary", "OOE_R_rec_entry_surf")),
            )
            selected_layout = refined_layout
            write_json(run_dir / "layout_refined.json", refined_layout)
            write_json(run_dir / "metrics_refined.json", refined_metrics)
            refine_log = {"steps": refine_steps}
            movement_breakdown = _movement_breakdown(layout_v0, refined_layout)
            write_json(run_dir / "refine_log.json", refine_log)
        elif method == "heuristic":
            refined_layout, refined_metrics, refine_steps = run_refinement_heuristic(
                layout=layout_v0,
                baseline_layout=baseline_layout,
                config=cfg,
                max_iterations=int(trial_cfg.get("refine_max_iterations", 30)),
                step_m=float(trial_cfg.get("refine_step_m", 0.1)),
                rot_deg=float(trial_cfg.get("refine_rot_deg", 15.0)),
                max_changed_objects=int(trial_cfg.get("refine_max_changed_objects", 3)),
            )
            selected_layout = refined_layout
            write_json(run_dir / "layout_refined.json", refined_layout)
            write_json(run_dir / "metrics_refined.json", refined_metrics)
            refine_log = {"steps": refine_steps}
            movement_breakdown = _movement_breakdown(layout_v0, refined_layout)
            write_json(run_dir / "refine_log.json", refine_log)
        elif method == "proposed":
            refined_layout, refined_metrics, refine_steps = run_refinement_proposed(
                layout=layout_v0,
                baseline_layout=baseline_layout,
                config=cfg,
                step_m=float(trial_cfg.get("refine_step_m", 0.1)),
                rot_deg=float(trial_cfg.get("refine_rot_deg", 15.0)),
                max_changed_objects=int(trial_cfg.get("refine_max_changed_objects", 3)),
                beam_width=int(trial_cfg.get("refine_beam_width", 5)),
                depth=int(trial_cfg.get("refine_depth", 3)),
                candidate_objects_per_state=int(trial_cfg.get("refine_candidate_objects_per_state", 2)),
                max_eval_calls=int(trial_cfg.get("refine_eval_budget", 0)),
                ooe_primary=str(trial_cfg.get("refine_ooe_primary", "OOE_R_rec_entry_surf")),
                use_lexicographic=_as_bool(trial_cfg.get("refine_use_lexicographic"), True),
                allow_intermediate_regression=_as_bool(trial_cfg.get("refine_allow_intermediate_regression"), True),
                door_keepout_radius_m=float(trial_cfg.get("refine_door_keepout_radius_m", 0.0)),
                overlap_ratio_max=float(trial_cfg.get("refine_overlap_ratio_max", 0.05)),
                delta_weight=float(trial_cfg.get("refine_delta_weight", 0.3)),
            )
            selected_layout = refined_layout
            write_json(run_dir / "layout_refined.json", refined_layout)
            write_json(run_dir / "metrics_refined.json", refined_metrics)
            refine_log = {"steps": refine_steps}
            movement_breakdown = _movement_breakdown(layout_v0, refined_layout)
            write_json(run_dir / "refine_log.json", refine_log)

        metrics, debug = evaluate_layout(selected_layout, baseline_layout, cfg)
        write_json(run_dir / "metrics.json", metrics)

        summary = {
            "trial_id": trial_id,
            "layout_id": layout_id,
            "method": method,
            "recovery_protocol": recovery_protocol,
            "eval_config_path": str(eval_config_path),
            "eval_config_name": eval_config_name,
            "eval_hash": eval_hash,
            "method_hash": method_hash,
            "seed": seed,
            "status": "ok",
            "error_msg": "",
            "run_dir": str(run_dir),
            **movement_breakdown,
            **metrics,
        }

        write_json(
            run_dir / "trial_manifest.json",
            {
                "trial_config": trial_cfg,
                "eval_config_path": str(eval_config_path),
                "eval_config_name": eval_config_name,
                "eval_config_sha256": eval_hash,
                "resolved_eval_config": cfg,
                "resolved_method_spec": method_spec,
                "resolved_method_hash": method_hash,
                "method_config_path": method_config_path,
                "method_config_sha256": method_config_sha256,
                "alignment_prior_config_path": alignment_prior_config_path,
                "alignment_prior_config_sha256": alignment_prior_config_sha256,
                "movement_breakdown": movement_breakdown,
                "v0_outputs": v0_outputs,
                "summary": summary,
                "refine_log": refine_log,
                "debug_meta": {
                    "path_length_cells": len(debug.get("path_cells") or []),
                    "bottleneck_cell": debug.get("bottleneck_cell"),
                    "task_points": baseline_task_points,
                    "task_points_final": debug.get("task_points"),
                },
            },
        )

    except Exception as exc:
        summary = {
            "trial_id": trial_id,
            "layout_id": layout_id,
            "method": method,
            "recovery_protocol": recovery_protocol,
            "eval_config_path": str(eval_config_path),
            "eval_config_name": eval_config_name,
            "eval_hash": eval_hash,
            "method_hash": method_hash,
            "seed": seed,
            "status": "error",
            "error_msg": str(exc),
            "run_dir": str(run_dir),
            "C_vis": "",
            "R_reach": "",
            "clr_min": "",
            "clr_min_astar": "",
            "clr_feasible": "",
            "Delta_layout": "",
            "Adopt": "",
            "Adopt_core": "",
            "Adopt_entry": "",
            "Adopt_clearance_metric": "",
            "Adopt_clearance_value": "",
            "Adopt_clearance_threshold": "",
            "Adopt_entry_gate_enabled": "",
            "Adopt_entry_metric": "",
            "Adopt_entry_metric_value": "",
            "Adopt_entry_min_value": "",
            "validity": "",
            "C_vis_start": "",
            "OOE_C_obj_entry_hit": "",
            "OOE_R_rec_entry_hit": "",
            "OOE_C_obj_entry_surf": "",
            "OOE_R_rec_entry_surf": "",
            "delta_layout_furniture": "",
            "delta_layout_clutter": "",
            "moved_furniture_count": "",
            "moved_clutter_count": "",
            "sum_furniture_displacement_m": "",
            "sum_clutter_displacement_m": "",
            "max_clutter_displacement_m": "",
            "runtime_sec": "",
        }
        write_json(run_dir / "trial_error.json", summary)

    _append_metrics_csv(pathlib.Path(args.out_root) / "metrics.csv", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one trial: v0 -> eval -> optional refine")
    parser.add_argument("--trial_config", required=True)
    parser.add_argument("--eval_config", default="experiments/configs/eval/eval_v1.json")
    parser.add_argument("--out_root", required=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = run_trial(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary.get("status") == "ok" else 1)


if __name__ == "__main__":
    main()
