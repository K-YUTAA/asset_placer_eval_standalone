from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


METHODS = ["heuristic", "proposed"]
PROTOCOLS = ["M0", "M1", "X1", "X2"]
VARIANT_ORDER = ["base", "usage_shift", "clutter", "compound"]
METRIC_KEYS = [
    "C_vis",
    "R_reach",
    "clr_min",
    "clr_min_astar",
    "clr_feasible",
    "Delta_layout",
    "Adopt",
    "Adopt_core",
    "Adopt_entry",
    "C_vis_start",
    "OOE_C_obj_entry_hit",
    "OOE_R_rec_entry_hit",
    "OOE_C_obj_entry_surf",
    "OOE_R_rec_entry_surf",
]


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def _run(cmd: List[str], cwd: pathlib.Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _source_generation_manifest(source_layout_path: pathlib.Path) -> Dict[str, Any]:
    manifest_path = source_layout_path.parent / "generation_manifest.json"
    if not manifest_path.exists():
        return {}
    return _load_json(manifest_path)


def _source_inputs(entry: Dict[str, Any]) -> Dict[str, str]:
    source_layout_path = pathlib.Path(str(entry.get("source_layout_path") or ""))
    manifest = _source_generation_manifest(source_layout_path) if source_layout_path.exists() else {}
    inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
    return {
        "sketch_path": str(inputs.get("image_path") or ""),
        "hints_path": str(inputs.get("hints_path") or ""),
    }


def _latest_run_dir(stage_root: pathlib.Path, trial_id: str) -> pathlib.Path:
    matches = sorted(stage_root.glob(f"{trial_id}_*"))
    if not matches:
        raise FileNotFoundError(f"run dir not found for trial_id={trial_id} under {stage_root}")
    return matches[-1]


def _build_trial_config(
    *,
    entry: Dict[str, Any],
    method: str,
    protocol: str,
    layout_input: pathlib.Path,
) -> Dict[str, Any]:
    base_case = str(entry.get("base_case") or "")
    variant = str(entry.get("variant") or "")
    source_inputs = _source_inputs(entry)
    recovery_protocol = "layout_only"
    if protocol in {"X1", "X2"}:
        recovery_protocol = "clutter_assisted"

    trial_cfg: Dict[str, Any] = {
        "layout_id": base_case,
        "seed": 0,
        "inputs": {
            "layout_input": str(layout_input),
            "sketch_path": source_inputs.get("sketch_path", ""),
            "hints_path": source_inputs.get("hints_path", ""),
        },
        "trial_id": f"{base_case}__{variant}__{method}__{protocol.lower()}",
        "method": method,
        "recovery_protocol": recovery_protocol,
    }
    if protocol == "X1":
        trial_cfg["refine_max_iterations"] = 30
        trial_cfg["refine_step_m"] = 0.1
        trial_cfg["refine_rot_deg"] = 15.0
        trial_cfg["refine_max_changed_objects"] = 3
    if protocol == "X2":
        trial_cfg["refine_max_iterations"] = 30
        trial_cfg["refine_step_m"] = 0.1
        trial_cfg["refine_rot_deg"] = 15.0
        trial_cfg["refine_max_changed_objects"] = 3
    return trial_cfg


def _baseline_row(entry: Dict[str, Any]) -> Dict[str, Any]:
    layout_path = pathlib.Path(str(entry.get("layout_path") or ""))
    stress_manifest_path = pathlib.Path(str(entry.get("stress_manifest_path") or ""))
    stress_case_dir = layout_path.parent
    metrics = entry.get("variant_metrics") if isinstance(entry.get("variant_metrics"), dict) else {}
    row: Dict[str, Any] = {
        "base_case": str(entry.get("base_case") or ""),
        "variant": str(entry.get("variant") or ""),
        "protocol": "M0",
        "method": "baseline",
        "trial_id": f"{entry.get('base_case')}__{entry.get('variant')}__m0",
        "run_dir": str(stress_case_dir),
        "layout_path": str(layout_path),
        "metrics_path": str(stress_case_dir / "metrics.json"),
        "plot_path": str(stress_case_dir / "plot_with_bg.png"),
        "stress_manifest_path": str(stress_manifest_path),
        "delta_layout_furniture": 0.0,
        "delta_layout_clutter": 0.0,
        "moved_furniture_count": 0,
        "moved_clutter_count": 0,
        "sum_furniture_displacement_m": 0.0,
        "sum_clutter_displacement_m": 0.0,
        "max_clutter_displacement_m": 0.0,
        "status": "ok",
        "source_protocol": "",
    }
    for key in METRIC_KEYS:
        row[key] = metrics.get(key, "")
    return row


def _trial_row(
    *,
    entry: Dict[str, Any],
    protocol: str,
    method: str,
    run_dir: pathlib.Path,
    source_protocol: str = "",
) -> Dict[str, Any]:
    manifest = _load_json(run_dir / "trial_manifest.json")
    summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    row: Dict[str, Any] = {
        "base_case": str(entry.get("base_case") or ""),
        "variant": str(entry.get("variant") or ""),
        "protocol": protocol,
        "method": method,
        "trial_id": str(summary.get("trial_id") or ""),
        "run_dir": str(run_dir),
        "layout_path": str(run_dir / "layout_refined.json"),
        "metrics_path": str(run_dir / "metrics_refined.json"),
        "plot_path": str(run_dir / "plot_with_bg_refined.png"),
        "stress_manifest_path": str(entry.get("stress_manifest_path") or ""),
        "delta_layout_furniture": summary.get("delta_layout_furniture", 0.0),
        "delta_layout_clutter": summary.get("delta_layout_clutter", 0.0),
        "moved_furniture_count": summary.get("moved_furniture_count", 0),
        "moved_clutter_count": summary.get("moved_clutter_count", 0),
        "sum_furniture_displacement_m": summary.get("sum_furniture_displacement_m", 0.0),
        "sum_clutter_displacement_m": summary.get("sum_clutter_displacement_m", 0.0),
        "max_clutter_displacement_m": summary.get("max_clutter_displacement_m", 0.0),
        "status": str(summary.get("status") or ""),
        "source_protocol": source_protocol,
    }
    for key in METRIC_KEYS:
        row[key] = summary.get(key, "")
    return row


def _write_csv(path: pathlib.Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _summaries(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    baseline_map = {
        (r["base_case"], r["variant"]): r
        for r in rows
        if r["protocol"] == "M0"
    }
    groups: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["protocol"] == "M0":
            continue
        groups[(row["method"], row["protocol"], row["variant"])].append(row)

    out: List[Dict[str, Any]] = []
    for (method, protocol, variant), items in sorted(groups.items()):
        n = len(items)
        if n == 0:
            continue
        summary: Dict[str, Any] = {
            "method": method,
            "protocol": protocol,
            "variant": variant,
            "n": n,
            "core_recovered": 0,
            "entry_recovered": 0,
        }
        for key in [
            "clr_feasible",
            "C_vis_start",
            "OOE_R_rec_entry_surf",
            "R_reach",
            "C_vis",
            "delta_layout_furniture",
            "delta_layout_clutter",
        ]:
            summary[f"mean_{key}"] = 0.0
            summary[f"mean_delta_{key}"] = 0.0

        for item in items:
            base = baseline_map[(item["base_case"], item["variant"])]
            if int(base.get("Adopt_core", 0)) == 0 and int(item.get("Adopt_core", 0)) == 1:
                summary["core_recovered"] += 1
            if int(base.get("Adopt_entry", 0)) == 0 and int(item.get("Adopt_entry", 0)) == 1:
                summary["entry_recovered"] += 1
            for key in [
                "clr_feasible",
                "C_vis_start",
                "OOE_R_rec_entry_surf",
                "R_reach",
                "C_vis",
                "delta_layout_furniture",
                "delta_layout_clutter",
            ]:
                val = float(item.get(key, 0.0) or 0.0)
                summary[f"mean_{key}"] += val
                if key.startswith("delta_layout_"):
                    summary[f"mean_delta_{key}"] += val
                else:
                    base_val = float(base.get(key, 0.0) or 0.0)
                    summary[f"mean_delta_{key}"] += (val - base_val)

        for key in list(summary.keys()):
            if key.startswith("mean_"):
                summary[key] = summary[key] / n
        out.append(summary)
    return out


def _write_md(out_path: pathlib.Path, rows: List[Dict[str, Any]], summary_rows: List[Dict[str, Any]], stress_root: pathlib.Path) -> None:
    lines: List[str] = []
    lines.append("# M0 / M1 / X1 / X2 compare")
    lines.append("")
    lines.append(f"- stress input: `{stress_root}`")
    lines.append("- protocols:")
    lines.append("  - `M0`: no refine")
    lines.append("  - `M1`: furniture refine")
    lines.append("  - `X1`: clutter refine")
    lines.append("  - `X2`: furniture -> clutter refine")
    lines.append("- methods:")
    lines.append("  - `heuristic`")
    lines.append("  - `proposed`")
    lines.append("")
    lines.append("## Summary by method / protocol / variant")
    lines.append("")
    lines.append("| method | protocol | variant | n | core_recovered | entry_recovered | mean_delta_clr_feasible | mean_delta_C_vis_start | mean_delta_OOE_R_rec_entry_surf | mean_delta_layout_furniture | mean_delta_layout_clutter |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['protocol']} | {row['variant']} | {row['n']} | "
            f"{row['core_recovered']} | {row['entry_recovered']} | "
            f"{row['mean_delta_clr_feasible']:.4f} | {row['mean_delta_C_vis_start']:.4f} | "
            f"{row['mean_delta_OOE_R_rec_entry_surf']:.4f} | "
            f"{row['mean_delta_delta_layout_furniture']:.4f} | {row['mean_delta_delta_layout_clutter']:.4f} |"
        )
    lines.append("")
    lines.append(f"- case rows: `{out_path.parent / 'protocol_compare_index.csv'}`")
    lines.append(f"- summary rows: `{out_path.parent / 'protocol_compare_summary.csv'}`")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M0/M1/X1/X2 compare on stress cases.")
    parser.add_argument("--stress_root", default="experiments/runs/stress_v2_natural_from_latest_design_frozen_20260312")
    parser.add_argument("--eval_config", default="experiments/configs/eval/eval_v1.json")
    parser.add_argument("--out_root", default="experiments/runs/m0_m1_x1_x2_compare_from_latest_design_frozen_20260312")
    parser.add_argument("--python_exec", default="/Users/yuuta/Research/asset_placer_eval_standalone/.venv/bin/python")
    args = parser.parse_args()

    repo_root = _repo_root()
    stress_root = pathlib.Path(args.stress_root)
    if not stress_root.is_absolute():
        stress_root = (repo_root / stress_root).resolve()
    out_root = pathlib.Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (repo_root / out_root).resolve()
    eval_config = pathlib.Path(args.eval_config)
    if not eval_config.is_absolute():
        eval_config = (repo_root / eval_config).resolve()
    python_exec = pathlib.Path(args.python_exec)

    manifest = _load_json(stress_root / "stress_cases_manifest.json")
    entries = manifest.get("entries") if isinstance(manifest.get("entries"), list) else []
    entries_sorted = sorted(
        entries,
        key=lambda e: (
            VARIANT_ORDER.index(str(e.get("variant") or "base")),
            str(e.get("base_case") or ""),
        ),
    )

    trial_cfg_root = out_root / "_trial_configs"
    rows: List[Dict[str, Any]] = []

    for entry in entries_sorted:
        base_case = str(entry.get("base_case") or "")
        variant = str(entry.get("variant") or "")
        stress_layout = pathlib.Path(str(entry.get("layout_path") or ""))
        rows.append(_baseline_row(entry))

        for method in METHODS:
            method_stage_root = out_root / f"{method}_m1_run"
            x1_stage_root = out_root / f"{method}_x1_run"
            x2_stage_root = out_root / f"{method}_x2_run"

            trial_cfg_m1 = _build_trial_config(entry=entry, method=method, protocol="M1", layout_input=stress_layout)
            cfg_m1_path = trial_cfg_root / f"{trial_cfg_m1['trial_id']}.json"
            _write_json(cfg_m1_path, trial_cfg_m1)
            _run(
                [
                    str(python_exec),
                    str(repo_root / "experiments/src/run_trial.py"),
                    "--trial_config",
                    str(cfg_m1_path),
                    "--eval_config",
                    str(eval_config),
                    "--out_root",
                    str(method_stage_root),
                ],
                repo_root,
            )
            m1_run_dir = _latest_run_dir(method_stage_root, str(trial_cfg_m1["trial_id"]))
            rows.append(_trial_row(entry=entry, protocol="M1", method=method, run_dir=m1_run_dir))

            trial_cfg_x1 = _build_trial_config(entry=entry, method=method, protocol="X1", layout_input=stress_layout)
            cfg_x1_path = trial_cfg_root / f"{trial_cfg_x1['trial_id']}.json"
            _write_json(cfg_x1_path, trial_cfg_x1)
            _run(
                [
                    str(python_exec),
                    str(repo_root / "experiments/src/run_trial.py"),
                    "--trial_config",
                    str(cfg_x1_path),
                    "--eval_config",
                    str(eval_config),
                    "--out_root",
                    str(x1_stage_root),
                ],
                repo_root,
            )
            x1_run_dir = _latest_run_dir(x1_stage_root, str(trial_cfg_x1["trial_id"]))
            rows.append(_trial_row(entry=entry, protocol="X1", method=method, run_dir=x1_run_dir))

            m1_layout_input = m1_run_dir / "layout_refined.json"
            trial_cfg_x2 = _build_trial_config(entry=entry, method=method, protocol="X2", layout_input=m1_layout_input)
            cfg_x2_path = trial_cfg_root / f"{trial_cfg_x2['trial_id']}.json"
            _write_json(cfg_x2_path, trial_cfg_x2)
            _run(
                [
                    str(python_exec),
                    str(repo_root / "experiments/src/run_trial.py"),
                    "--trial_config",
                    str(cfg_x2_path),
                    "--eval_config",
                    str(eval_config),
                    "--out_root",
                    str(x2_stage_root),
                ],
                repo_root,
            )
            x2_run_dir = _latest_run_dir(x2_stage_root, str(trial_cfg_x2["trial_id"]))
            rows.append(_trial_row(entry=entry, protocol="X2", method=method, run_dir=x2_run_dir, source_protocol="M1"))

    fieldnames = [
        "base_case",
        "variant",
        "protocol",
        "method",
        "trial_id",
        "run_dir",
        "layout_path",
        "metrics_path",
        "plot_path",
        "stress_manifest_path",
        "status",
        "source_protocol",
        *METRIC_KEYS,
        "delta_layout_furniture",
        "delta_layout_clutter",
        "moved_furniture_count",
        "moved_clutter_count",
        "sum_furniture_displacement_m",
        "sum_clutter_displacement_m",
        "max_clutter_displacement_m",
    ]
    _write_csv(out_root / "protocol_compare_index.csv", rows, fieldnames)

    summary_rows = _summaries(rows)
    summary_fields = [
        "method",
        "protocol",
        "variant",
        "n",
        "core_recovered",
        "entry_recovered",
        "mean_clr_feasible",
        "mean_delta_clr_feasible",
        "mean_C_vis_start",
        "mean_delta_C_vis_start",
        "mean_OOE_R_rec_entry_surf",
        "mean_delta_OOE_R_rec_entry_surf",
        "mean_R_reach",
        "mean_delta_R_reach",
        "mean_C_vis",
        "mean_delta_C_vis",
        "mean_delta_layout_furniture",
        "mean_delta_delta_layout_furniture",
        "mean_delta_layout_clutter",
        "mean_delta_delta_layout_clutter",
    ]
    _write_csv(out_root / "protocol_compare_summary.csv", summary_rows, summary_fields)
    _write_md(out_root / "protocol_compare_summary.md", rows, summary_rows, stress_root)


if __name__ == "__main__":
    main()
