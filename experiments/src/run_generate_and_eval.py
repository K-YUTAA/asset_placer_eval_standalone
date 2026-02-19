from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate layout JSON and run eval/plot in one command")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--dimensions_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--eval_config", default="experiments/configs/eval/default_eval.json")

    parser.add_argument("--prompt1_path", default=None)
    parser.add_argument("--prompt2_path", default=None)
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--reasoning_effort", default="medium")
    parser.add_argument("--text_verbosity", default="high")
    parser.add_argument("--max_output_tokens", type=int, default=32000)
    parser.add_argument("--image_detail", default="high")
    parser.add_argument("--step2_text_only", action="store_true")

    parser.add_argument("--enable_gemini_spatial", action="store_true")
    parser.add_argument("--gemini_api_key", default=None)
    parser.add_argument("--gemini_model", default="gemini-2.5-flash")
    parser.add_argument("--gemini_task", choices=["boxes", "masks"], default="boxes")
    parser.add_argument("--gemini_target_prompt", default=None)
    parser.add_argument("--gemini_prompt_text", default=None)
    parser.add_argument("--gemini_label_language", default="English")
    parser.add_argument("--gemini_temperature", type=float, default=0.6)
    parser.add_argument("--gemini_thinking_budget", type=int, default=0)
    parser.add_argument("--gemini_max_items", type=int, default=20)
    parser.add_argument("--gemini_resize_max", type=int, default=640)
    parser.add_argument("--gemini_include_non_furniture", action="store_true")
    parser.add_argument("--enable_gemini_openings", action="store_true")
    parser.add_argument("--gemini_openings_prompt_text", default=None)

    parser.add_argument("--bg_image", default=None, help="Optional. If set, also generate plot_with_bg.png")
    parser.add_argument("--bg_crop_mode", default="none", choices=["none", "beige", "nonwhite"])
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layout_json = out_dir / "layout_generated.json"
    metrics_json = out_dir / "metrics.json"
    debug_dir = out_dir / "debug"
    script_dir = pathlib.Path(__file__).resolve().parent

    gen_cmd = [
        sys.executable,
        str(script_dir / "generate_layout_json.py"),
        "--image_path",
        args.image_path,
        "--dimensions_path",
        args.dimensions_path,
        "--out_json",
        str(layout_json),
        "--out_dir",
        str(out_dir),
        "--model",
        args.model,
        "--reasoning_effort",
        args.reasoning_effort,
        "--text_verbosity",
        args.text_verbosity,
        "--max_output_tokens",
        str(args.max_output_tokens),
        "--image_detail",
        args.image_detail,
    ]
    if args.prompt1_path:
        gen_cmd.extend(["--prompt1_path", args.prompt1_path])
    if args.prompt2_path:
        gen_cmd.extend(["--prompt2_path", args.prompt2_path])
    if args.step2_text_only:
        gen_cmd.append("--step2_text_only")

    if args.enable_gemini_spatial:
        gen_cmd.append("--enable_gemini_spatial")
        gen_cmd.extend(["--gemini_model", args.gemini_model])
        gen_cmd.extend(["--gemini_task", args.gemini_task])
        gen_cmd.extend(["--gemini_label_language", args.gemini_label_language])
        gen_cmd.extend(["--gemini_temperature", str(args.gemini_temperature)])
        gen_cmd.extend(["--gemini_thinking_budget", str(args.gemini_thinking_budget)])
        gen_cmd.extend(["--gemini_max_items", str(args.gemini_max_items)])
        gen_cmd.extend(["--gemini_resize_max", str(args.gemini_resize_max)])
        if args.gemini_api_key:
            gen_cmd.extend(["--gemini_api_key", args.gemini_api_key])
        if args.gemini_target_prompt:
            gen_cmd.extend(["--gemini_target_prompt", args.gemini_target_prompt])
        if args.gemini_prompt_text:
            gen_cmd.extend(["--gemini_prompt_text", args.gemini_prompt_text])
        if args.gemini_include_non_furniture:
            gen_cmd.append("--gemini_include_non_furniture")
        if args.enable_gemini_openings:
            gen_cmd.append("--enable_gemini_openings")
        if args.gemini_openings_prompt_text:
            gen_cmd.extend(["--gemini_openings_prompt_text", args.gemini_openings_prompt_text])

    subprocess.run(gen_cmd, check=True)

    eval_cmd = [
        sys.executable,
        str(script_dir / "eval_metrics.py"),
        "--layout",
        str(layout_json),
        "--config",
        args.eval_config,
        "--out",
        str(metrics_json),
        "--debug_dir",
        str(debug_dir),
    ]
    subprocess.run(eval_cmd, check=True)

    if args.bg_image:
        plot_cmd = [
            sys.executable,
            str(script_dir / "plot_layout_json.py"),
            "--layout",
            str(layout_json),
            "--out",
            str(out_dir / "plot_with_bg.png"),
            "--bg_image",
            args.bg_image,
            "--bg_crop_mode",
            args.bg_crop_mode,
            "--metrics_json",
            str(metrics_json),
            "--task_points_json",
            str(debug_dir / "task_points.json"),
            "--path_json",
            str(debug_dir / "path_cells.json"),
        ]
        subprocess.run(plot_cmd, check=True)


if __name__ == "__main__":
    main()
