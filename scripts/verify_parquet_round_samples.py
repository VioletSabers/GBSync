#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从某一轮的 parquet 中随机抽取若干条样本，在对应图像右侧拼接白底备注区，
将 content_dict 中的信息按行写出，便于人工核对生成数据是否正确。

用法:
  python scripts/verify_parquet_round_samples.py /path/to/output/.../parquet/round_0001.parquet
  python scripts/verify_parquet_round_samples.py /path/to/round_0001.parquet -n 10 --seed 42

输出目录: 与 parquet、images 同级的 check/（例如 .../config_01_cn_color/check/）
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


def _parse_content_dict(raw: Any) -> Any:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    s = raw.strip()
    if not s:
        return None
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return None


def _content_dict_to_lines(content: Any) -> List[str]:
    """将 content_dict 解析结果转为多行纯文本（按块、按字段分行）。"""
    if content is None:
        return ["(content_dict 无法解析)"]
    if isinstance(content, list):
        out: List[str] = []
        for idx, block in enumerate(content):
            out.append(f"--- [{idx}] ---")
            if isinstance(block, dict):
                for key in sorted(block.keys()):
                    val = block[key]
                    if isinstance(val, (dict, list)):
                        val_str = json.dumps(val, ensure_ascii=False)
                    else:
                        val_str = str(val)
                    out.append(f"{key}: {val_str}")
            else:
                out.append(repr(block))
        return out
    return json.dumps(content, ensure_ascii=False, indent=2).splitlines()


def _try_load_font(size: int) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\simhei.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if not p.is_file():
            continue
        try:
            return ImageFont.truetype(str(p), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_line_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> List[str]:
    if max_width <= 8:
        return [text] if text else [""]
    out: List[str] = []
    current = ""
    for ch in text:
        trial = current + ch
        bbox = draw.textbbox((0, 0), trial, font=font)
        w = bbox[2] - bbox[0]
        if w <= max_width or not current:
            current = trial
        else:
            out.append(current)
            current = ch
    if current:
        out.append(current)
    return out if out else [""]


def _layout_sidebar_lines(
    draw: ImageDraw.ImageDraw,
    logical_lines: Sequence[str],
    font: ImageFont.ImageFont,
    max_width: int,
    max_pixel_height: int,
    line_gap: int,
) -> Tuple[List[str], int]:
    wrapped: List[str] = []
    for ln in logical_lines:
        wrapped.extend(_wrap_line_to_width(draw, ln, font, max_width))
    # 按高度裁剪（避免极端超长）
    usable = max(1, max_pixel_height - 24)
    _bb = draw.textbbox((0, 0), "国Ay", font=font)
    line_h = max(12, _bb[3] - _bb[1])
    max_lines = max(1, usable // (line_h + line_gap))
    if len(wrapped) > max_lines:
        wrapped = wrapped[: max_lines - 1] + ["… (以下省略)"]
    return wrapped, line_h


def _build_check_image(
    image_path: Path,
    sidebar_lines: Sequence[str],
    panel_width_ratio: float = 0.42,
    min_panel: int = 280,
    max_panel: int = 720,
    margin: int = 12,
    font_size: int = 14,
) -> Image.Image:
    base = Image.open(image_path).convert("RGB")
    w, h = base.size
    panel_w = int(max(min_panel, min(max_panel, w * panel_width_ratio)))
    out = Image.new("RGB", (w + panel_w, h), (255, 255, 255))
    out.paste(base, (0, 0))
    draw = ImageDraw.Draw(out)
    font = _try_load_font(font_size)
    x0 = w + margin
    max_text_w = panel_w - 2 * margin
    wrapped, line_h = _layout_sidebar_lines(
        draw, sidebar_lines, font, max_text_w, h - 2 * margin, line_gap=4
    )
    y = margin
    for line in wrapped:
        draw.text((x0, y), line, fill=(0, 0, 0), font=font)
        y += line_h + 4
        if y > h - margin:
            break
    draw.rectangle([w, 0, w + panel_w - 1, h - 1], outline=(200, 200, 200), width=1)
    return out


def _resolve_paths(parquet_path: Path, image_dir_name: str) -> Tuple[Path, Path, Path]:
    parquet_path = parquet_path.expanduser().resolve()
    if not parquet_path.is_file():
        raise FileNotFoundError(f"找不到 parquet: {parquet_path}")
    output_base = parquet_path.parent.parent
    image_root = output_base / image_dir_name
    check_dir = output_base / "check"
    return image_root, check_dir, output_base


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="从 parquet 抽样并生成带 content_dict 备注的核对图")
    parser.add_argument(
        "parquet",
        type=Path,
        help="该轮 parquet 文件路径，例如 .../parquet/round_0001.parquet",
    )
    parser.add_argument("-n", "--num-samples", type=int, default=10, help="随机抽样条数（默认 10）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument(
        "--image-dir",
        default="images",
        help="与 output.root_dir 下图像根目录名一致（默认 images）",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.seed is not None:
        random.seed(args.seed)

    image_root, check_dir, output_base = _resolve_paths(args.parquet, args.image_dir)

    table = pq.read_table(args.parquet)
    df = table.to_pandas()
    if df.empty:
        print("parquet 为空", file=sys.stderr)
        return 1
    need = min(args.num_samples, len(df))
    indices = random.sample(range(len(df)), k=need)
    rows = df.iloc[indices]

    check_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for _, row in rows.iterrows():
        sample_id = str(row.get("ID", ""))
        rel = str(row.get("relative_img_path", ""))
        raw_cd = row.get("content_dict", "")
        content = _parse_content_dict(raw_cd)
        lines = _content_dict_to_lines(content)
        header = [f"ID: {sample_id}", f"relative_img_path: {rel}", ""]
        sidebar = header + lines

        img_path = image_root / rel
        if not img_path.is_file():
            print(f"[skip] 图像不存在: {img_path}", file=sys.stderr)
            safe_id = sample_id.replace("/", "_") or "unknown"
            out_path = check_dir / f"{safe_id}_MISSING.png"
            # 仍生成一张仅备注图便于对照
            placeholder = Image.new("RGB", (400, 300), (240, 240, 240))
            ph_draw = ImageDraw.Draw(placeholder)
            ph_font = _try_load_font(16)
            y = 8
            for ln in sidebar[:40]:
                ph_draw.text((8, y), ln[:120], fill=(0, 0, 0), font=ph_font)
                y += 22
            placeholder.save(out_path)
            continue

        out_name = f"{sample_id.replace('/', '_')}_check.png"
        out_path = check_dir / out_name
        try:
            img = _build_check_image(img_path, sidebar)
            img.save(out_path)
            ok += 1
            print(out_path)
        except Exception as exc:
            print(f"[fail] {sample_id}: {exc}", file=sys.stderr)

    print(f"完成: {ok}/{need} 张写入 {check_dir} (输出根: {output_base})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
