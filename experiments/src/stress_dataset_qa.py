from __future__ import annotations

import pathlib
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from layout_tools import utc_now_iso, write_json
from stress_manifest import git_head_sha, sha256_file


def _required_fields(schema: Dict[str, Any]) -> List[str]:
    req = schema.get("required")
    if not isinstance(req, list):
        return []
    return [str(x) for x in req]


def _nested_item_required(schema: Dict[str, Any], key: str) -> List[str]:
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    node = props.get(key) if isinstance(props, dict) else None
    if not isinstance(node, dict):
        return []
    items = node.get("items") if isinstance(node.get("items"), dict) else {}
    req = items.get("required") if isinstance(items.get("required"), list) else []
    return [str(x) for x in req]


def validate_sample_manifest(manifest: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    for field in _required_fields(schema):
        if field not in manifest:
            missing.append(field)
    if missing:
        return missing

    for key in ("moved_objects", "added_clutter"):
        nested_req = _nested_item_required(schema, key)
        items = manifest.get(key)
        if not isinstance(items, list):
            missing.append(key)
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                missing.append(f"{key}[{idx}]")
                continue
            for sub in nested_req:
                if sub not in item:
                    missing.append(f"{key}[{idx}].{sub}")
    return missing


def build_dataset_qa_report(
    *,
    out_root: pathlib.Path,
    stress_cfg: Dict[str, Any],
    stress_config_path: pathlib.Path,
    sample_schema: Dict[str, Any],
    entries: Sequence[Dict[str, Any]],
    repo_root: pathlib.Path,
) -> Dict[str, Any]:
    case_reports: List[Dict[str, Any]] = []
    exists_pass = 0
    complete_pass = 0
    explainable_pass = 0

    for entry in entries:
        base_case_id = str(entry.get("base_case") or entry.get("base_case_id") or "")
        variant = str(entry.get("scenario") or entry.get("stress_family") or entry.get("variant") or "")
        manifest_path = pathlib.Path(str(entry.get("stress_manifest_path") or ""))
        exists = manifest_path.is_file()
        complete = False
        explainable = False
        missing_required_fields: List[str] = []
        notes = ""
        if exists:
            exists_pass += 1
            try:
                import json
                manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
                missing_required_fields = validate_sample_manifest(manifest, sample_schema)
                complete = len(missing_required_fields) == 0
                if complete:
                    complete_pass += 1
                explainable = complete and bool(str(manifest.get("selection_notes") or "").strip())
                if variant != "base":
                    explainable = explainable and (
                        bool(manifest.get("moved_objects")) or bool(manifest.get("added_clutter"))
                    )
                if explainable:
                    explainable_pass += 1
                if not explainable:
                    notes = "selection_notes or perturbation details are insufficient"
            except Exception as exc:
                notes = f"manifest parse failed: {exc}"
        else:
            notes = "manifest file missing"

        case_reports.append(
            {
                "base_case_id": base_case_id,
                "scene_id": f"{base_case_id}__{variant}",
                "stress_family": variant,
                "manifest_path": str(manifest_path),
                "exists": exists,
                "complete": complete,
                "explainable": explainable,
                "missing_required_fields": missing_required_fields,
                "notes": notes,
            }
        )

    report = {
        "schema_version": "stress_dataset_qa_v1",
        "dataset_id": out_root.name,
        "stress_version": str(stress_cfg.get("stress_version") or "unknown"),
        "config_path": str(stress_config_path),
        "config_hash": sha256_file(stress_config_path),
        "generator_commit": git_head_sha(repo_root),
        "generated_at": utc_now_iso(),
        "expected_case_count": len({str(entry.get("base_case") or entry.get("base_case_id") or "") for entry in entries}),
        "expected_variant_count_per_case": len(stress_cfg.get("variants") or []),
        "case_reports": case_reports,
        "summary": {
            "case_count": len(case_reports),
            "exists_pass_count": exists_pass,
            "complete_pass_count": complete_pass,
            "explainable_pass_count": explainable_pass,
        },
    }
    return report


def write_dataset_qa_report(path: pathlib.Path, report: Dict[str, Any]) -> None:
    write_json(path, report)
