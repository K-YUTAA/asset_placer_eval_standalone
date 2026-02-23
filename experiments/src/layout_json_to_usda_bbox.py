from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import shutil
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_COLORS: Dict[str, Tuple[float, float, float]] = {
    "floor": (0.75, 0.75, 0.75),
    "bed": (0.80, 0.55, 0.55),
    "chair": (0.95, 0.75, 0.45),
    "sink": (0.60, 0.85, 0.95),
    "toilet": (0.85, 0.95, 0.85),
    "storage": (0.95, 0.90, 0.70),
    "cabinet": (0.82, 0.82, 0.82),
    "tv_cabinet": (0.82, 0.82, 0.82),
    "door": (0.95, 0.85, 0.60),
    "window": (0.70, 0.85, 1.00),
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _sanitize_prim_name(name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "_", str(name or "obj"))
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "obj"
    if text[0].isdigit():
        text = f"_{text}"
    return text


def _format_float(v: float) -> str:
    return f"{float(v):.6f}".rstrip("0").rstrip(".") if "." in f"{float(v):.6f}" else f"{float(v):.6f}"


def _build_cube_block(
    prim_name: str,
    x: float,
    y: float,
    z: float,
    length: float,
    width: float,
    height: float,
    rot_z_deg: float,
    functional_rot_z_deg: float,
    color: Tuple[float, float, float],
) -> str:
    tx = _format_float(x)
    ty = _format_float(y)
    tz = _format_float(z)
    sx = _format_float(max(0.01, length))
    sy = _format_float(max(0.01, width))
    sz = _format_float(max(0.01, height))
    rz = _format_float(rot_z_deg)
    frz = _format_float(functional_rot_z_deg)
    cr = _format_float(color[0])
    cg = _format_float(color[1])
    cb = _format_float(color[2])
    return f"""        def Cube "{prim_name}"
        {{
            double size = 1
            float3 xformOp:translate = ({tx}, {ty}, {tz})
            float3 xformOp:rotateXYZ = (0, 0, {rz})
            float3 xformOp:scale = ({sx}, {sy}, {sz})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
            custom float userProperties:functionalRotationZ = {frz}
            color3f[] primvars:displayColor = [({cr}, {cg}, {cb})]
        }}
"""


def _parse_polygon(raw_poly: Any) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    if not isinstance(raw_poly, list):
        return points
    for p in raw_poly:
        if isinstance(p, dict):
            points.append((_as_float(p.get("X"), 0.0), _as_float(p.get("Y"), 0.0)))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            points.append((_as_float(p[0], 0.0), _as_float(p[1], 0.0)))
    return points


def _iter_polygon_edges(poly: List[Tuple[float, float]]) -> Iterable[Tuple[Tuple[float, float], Tuple[float, float]]]:
    n = len(poly)
    if n < 2:
        return
    for i in range(n):
        p0 = poly[i]
        p1 = poly[(i + 1) % n]
        if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) <= 1e-6:
            continue
        yield p0, p1


def _segment_key(p0: Tuple[float, float], p1: Tuple[float, float], q: float = 1000.0) -> Tuple[int, int, int, int]:
    a = (int(round(p0[0] * q)), int(round(p0[1] * q)))
    b = (int(round(p1[0] * q)), int(round(p1[1] * q)))
    if a <= b:
        return (a[0], a[1], b[0], b[1])
    return (b[0], b[1], a[0], a[1])


def _collect_wall_segments(layout_json: Dict[str, Any], include_room_walls: bool) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    segments: Dict[Tuple[int, int, int, int], Tuple[Tuple[float, float], Tuple[float, float]]] = {}

    outer_poly = _parse_polygon(layout_json.get("outer_polygon"))
    for p0, p1 in _iter_polygon_edges(outer_poly):
        segments[_segment_key(p0, p1)] = (p0, p1)

    if include_room_walls:
        rooms = layout_json.get("rooms")
        if isinstance(rooms, list):
            for room in rooms:
                if not isinstance(room, dict):
                    continue
                poly = _parse_polygon(room.get("room_polygon"))
                for p0, p1 in _iter_polygon_edges(poly):
                    segments[_segment_key(p0, p1)] = (p0, p1)

    return list(segments.values())


def _build_wall_blocks(
    layout_json: Dict[str, Any],
    wall_height: float,
    wall_thickness: float,
    include_room_walls: bool,
    used_names: set[str],
) -> List[str]:
    wall_blocks: List[str] = []
    wall_color = DEFAULT_COLORS.get("wall", (0.35, 0.35, 0.35))
    segments = _collect_wall_segments(layout_json, include_room_walls=include_room_walls)
    idx = 0
    for p0, p1 in segments:
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        seg_len = math.hypot(dx, dy)
        if seg_len <= 1e-6:
            continue
        cx = (p0[0] + p1[0]) * 0.5
        cy = (p0[1] + p1[1]) * 0.5
        rot_z = math.degrees(math.atan2(dy, dx))
        prim_name = _sanitize_prim_name(f"wall_{idx:03d}")
        while prim_name in used_names:
            idx += 1
            prim_name = _sanitize_prim_name(f"wall_{idx:03d}")
        used_names.add(prim_name)
        idx += 1
        wall_blocks.append(
            _build_cube_block(
                prim_name=prim_name,
                x=cx,
                y=cy,
                z=max(0.0, wall_height * 0.5),
                length=seg_len,
                width=max(0.01, wall_thickness),
                height=max(0.01, wall_height),
                rot_z_deg=rot_z,
                functional_rot_z_deg=rot_z,
                color=wall_color,
            )
        )
    return wall_blocks


def _build_group_block(group_name: str, child_blocks: List[str]) -> str:
    if not child_blocks:
        return ""
    body = "".join(child_blocks)
    return f"""    def Xform "{group_name}"
    {{
{body}    }}
"""


def _layout_bounds(layout_json: Dict[str, Any]) -> Tuple[float, float, float, float]:
    poly = _parse_polygon(layout_json.get("outer_polygon"))
    if poly:
        min_x = min(p[0] for p in poly)
        max_x = max(p[0] for p in poly)
        min_y = min(p[1] for p in poly)
        max_y = max(p[1] for p in poly)
        if (max_x - min_x) > 1e-6 and (max_y - min_y) > 1e-6:
            return min_x, max_x, min_y, max_y

    size_x = _as_float(layout_json.get("area_size_X"), 1.0)
    size_y = _as_float(layout_json.get("area_size_Y"), 1.0)
    return 0.0, max(0.01, size_x), 0.0, max(0.01, size_y)


def _build_overlay_materials_block(overlay_assets: List[str]) -> str:
    if not overlay_assets:
        return ""

    mats: List[str] = []
    for idx, asset_rel in enumerate(overlay_assets):
        mat_name = _sanitize_prim_name(f"OverlayMat_{idx:03d}")
        pbr = f"/LayoutBBox/Materials/{mat_name}/PBRShader"
        tex = f"/LayoutBBox/Materials/{mat_name}/DiffuseTexture"
        st = f"/LayoutBBox/Materials/{mat_name}/PrimvarReader_st"
        mats.append(
            f"""        def Material "{mat_name}"
        {{
            token outputs:surface.connect = <{pbr}.outputs:surface>

            def Shader "PBRShader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = <{tex}.outputs:rgb>
                float inputs:roughness = 1
                float inputs:metallic = 0
                token outputs:surface
            }}

            def Shader "PrimvarReader_st"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                token inputs:varname = "st"
                float2 outputs:result
            }}

            def Shader "DiffuseTexture"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{asset_rel}@
                float2 inputs:st.connect = <{st}.outputs:result>
                token inputs:sourceColorSpace = "sRGB"
                token inputs:wrapS = "clamp"
                token inputs:wrapT = "clamp"
                float3 outputs:rgb
            }}
        }}
"""
        )
    body = "".join(mats)
    return f"""    def Scope "Materials"
    {{
{body}    }}
"""


def _build_overlay_meshes_block(
    layout_json: Dict[str, Any],
    overlay_assets: List[str],
    overlay_z_base: float,
    overlay_z_step: float,
) -> str:
    if not overlay_assets:
        return ""

    min_x, max_x, min_y, max_y = _layout_bounds(layout_json)
    z_base = max(0.0001, float(overlay_z_base))
    z_step = max(0.0001, float(overlay_z_step))

    mesh_blocks: List[str] = []
    for idx, _asset_rel in enumerate(overlay_assets):
        mesh_name = _sanitize_prim_name(f"Overlay_{idx:03d}")
        mat_name = _sanitize_prim_name(f"OverlayMat_{idx:03d}")
        z = _format_float(z_base + (idx * z_step))
        x0 = _format_float(min_x)
        x1 = _format_float(max_x)
        y0 = _format_float(min_y)
        y1 = _format_float(max_y)
        mesh_blocks.append(
            f"""        def Mesh "{mesh_name}"
        {{
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [({x0}, {y0}, {z}), ({x1}, {y0}, {z}), ({x1}, {y1}, {z}), ({x0}, {y1}, {z})]
            texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
                interpolation = "vertex"
            )
            rel material:binding = </LayoutBBox/Materials/{mat_name}>
        }}
"""
        )

    body = "".join(mesh_blocks)
    return f"""    def Xform "Overlays"
    {{
{body}    }}
"""


def build_usda_text(
    layout_json: Dict[str, Any],
    meters_per_unit: float = 1.0,
    add_walls: bool = False,
    wall_height: float = 2.4,
    wall_thickness: float = 0.12,
    include_room_walls: bool = True,
    overlay_assets: List[str] | None = None,
    overlay_z_base: float = 0.002,
    overlay_z_step: float = 0.002,
) -> str:
    objects = layout_json.get("area_objects_list")
    if not isinstance(objects, list):
        objects = []
    overlay_assets = overlay_assets or []
    size_mode = str(layout_json.get("size_mode") or "local").strip().lower()
    world_axis_mode = size_mode in {"world", "meters", "meter", "global"}

    used_names: set[str] = set()
    object_blocks: List[str] = []
    for idx, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        base_name = str(obj.get("object_name") or f"obj_{idx:03d}")
        prim_name = _sanitize_prim_name(base_name)
        if prim_name in used_names:
            prim_name = f"{prim_name}_{idx:03d}"
        used_names.add(prim_name)

        category = str(obj.get("category") or "object").strip().lower()
        color = DEFAULT_COLORS.get(category, (0.85, 0.85, 0.85))

        x = _as_float(obj.get("X"), 0.0)
        y = _as_float(obj.get("Y"), 0.0)
        length = _as_float(obj.get("Length"), 0.1)
        width = _as_float(obj.get("Width"), 0.1)
        height = _as_float(obj.get("Height"), 0.1)
        functional_rot_z = _as_float(obj.get("rotationZ"), 0.0)
        rot_z = 0.0 if world_axis_mode else functional_rot_z
        z = max(0.0, height * 0.5)

        object_blocks.append(
            _build_cube_block(
                prim_name=prim_name,
                x=x,
                y=y,
                z=z,
                length=length,
                width=width,
                height=height,
                rot_z_deg=rot_z,
                functional_rot_z_deg=functional_rot_z,
                color=color,
            )
        )

    wall_blocks: List[str] = []
    if add_walls:
        wall_blocks = _build_wall_blocks(
            layout_json=layout_json,
            wall_height=float(wall_height),
            wall_thickness=float(wall_thickness),
            include_room_walls=bool(include_room_walls),
            used_names=used_names,
        )

    meters_text = _format_float(meters_per_unit)
    header = f"""#usda 1.0
(
    upAxis = "Z"
    metersPerUnit = {meters_text}
    defaultPrim = "LayoutBBox"
)

def Xform "LayoutBBox"
{{
"""
    groups = (
        _build_group_block("Objects", object_blocks)
        + _build_group_block("Walls", wall_blocks)
        + _build_overlay_materials_block(overlay_assets)
        + _build_overlay_meshes_block(
            layout_json=layout_json,
            overlay_assets=overlay_assets,
            overlay_z_base=float(overlay_z_base),
            overlay_z_step=float(overlay_z_step),
        )
    )
    footer = """}
"""
    return header + groups + footer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert layout_generated.json to USD ASCII (bbox cubes)")
    parser.add_argument("--layout_json", required=True, help="Input layout JSON path")
    parser.add_argument("--out_usda", required=True, help="Output .usda path")
    parser.add_argument("--meters_per_unit", type=float, default=1.0, help="USD metersPerUnit")
    parser.add_argument("--add_walls", action="store_true", help="Generate wall bbox cubes from outer/room polygons")
    parser.add_argument("--wall_height", type=float, default=2.4, help="Wall height in meters")
    parser.add_argument("--wall_thickness", type=float, default=0.12, help="Wall thickness in meters")
    parser.add_argument(
        "--outer_walls_only",
        action="store_true",
        help="If set, generate walls only from outer_polygon (skip rooms polygons).",
    )
    parser.add_argument(
        "--overlay_images",
        nargs="*",
        default=None,
        help="Optional image paths to place as textured ground overlays in order.",
    )
    parser.add_argument(
        "--overlay_z_base",
        type=float,
        default=0.002,
        help="Base Z for first overlay mesh.",
    )
    parser.add_argument(
        "--overlay_z_step",
        type=float,
        default=0.002,
        help="Z increment per additional overlay mesh.",
    )
    return parser


def _prepare_overlay_assets(image_paths: List[str], out_dir: pathlib.Path) -> List[str]:
    assets: List[str] = []
    for idx, raw in enumerate(image_paths):
        src = pathlib.Path(raw)
        if not src.exists():
            raise FileNotFoundError(f"overlay image not found: {src}")
        suffix = src.suffix or ".png"
        stem = _sanitize_prim_name(src.stem)
        dst_name = f"overlay_{idx:03d}_{stem}{suffix.lower()}"
        dst = out_dir / dst_name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        assets.append(dst_name)
    return assets


def main() -> None:
    args = build_arg_parser().parse_args()
    layout_path = pathlib.Path(args.layout_json)
    out_path = pathlib.Path(args.out_usda)

    if not layout_path.exists():
        raise FileNotFoundError(f"layout_json not found: {layout_path}")

    layout_json = json.loads(layout_path.read_text(encoding="utf-8-sig"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_images = list(args.overlay_images or [])
    overlay_assets = _prepare_overlay_assets(overlay_images, out_path.parent) if overlay_images else []
    usda_text = build_usda_text(
        layout_json,
        meters_per_unit=float(args.meters_per_unit),
        add_walls=bool(args.add_walls),
        wall_height=float(args.wall_height),
        wall_thickness=float(args.wall_thickness),
        include_room_walls=not bool(args.outer_walls_only),
        overlay_assets=overlay_assets,
        overlay_z_base=float(args.overlay_z_base),
        overlay_z_step=float(args.overlay_z_step),
    )

    out_path.write_text(usda_text, encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
