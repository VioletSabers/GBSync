from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def _inject_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize all configured fonts.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/render_config.yaml",
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--font-category",
        type=str,
        default=None,
        help="Font category to preview (e.g. simple/complex).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for per-font previews. Defaults to output/font_preview_<category>/",
    )
    return parser.parse_args()


def _sample_text_by_corpus() -> Dict[str, str]:
    return {
        "chinese": "中文字体预览：宋体 楷书 行书 隶书 手写体 示例",
        "english": "Font preview: The quick brown fox jumps over the lazy dog. 12345",
        "emoji": "😀😄😎🚀✨🎯✅🔥🌟🎉",
        "line_break": "line break marker",
    }


def _load_font_with_fallback(font_path: str, requested_size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(font_path, requested_size)
    except OSError:
        pass
    for size in [20, 24, 28, 32, 36, 40, 44, 48, 56, 64, 96]:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    raise OSError(f"Unable to load font: {font_path}")


def _collect_rows(config, config_dir: Path) -> List[Tuple[str, str, str]]:
    samples = _sample_text_by_corpus()
    rows: List[Tuple[str, str, str]] = []
    for corpus_type, paths in config.fonts_by_corpus.items():
        if corpus_type == "line_break":
            continue
        text = samples.get(corpus_type, "Preview text")
        for p in paths:
            from simple_rendering.config import resolve_config_path

            resolved = str(resolve_config_path(p, config_dir))
            rows.append((corpus_type, resolved, text))
    return rows


def _safe_stem(name: str) -> str:
    safe = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "font_preview"


def _render_single_preview(corpus_type: str, font_path: str, text: str, out_dir: Path) -> Path:
    width = 1500
    height = 260
    image = Image.new("RGB", (width, height), "#111111")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default()

    font_file = Path(font_path).name
    draw.text((20, 18), f"[{corpus_type}] {font_file}", fill="#A0E7E5", font=title_font)

    try:
        font = _load_font_with_fallback(font_path, 64 if corpus_type != "emoji" else 56)
        draw.text((20, 90), text, fill="#FFFFFF", font=font, embedded_color=True)
    except OSError:
        draw.text((20, 120), "(font load failed)", fill="#FF8A80", font=title_font)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{_safe_stem(Path(font_path).stem)}.png"
    out_path = out_dir / out_name
    # Avoid overwriting when same stem appears in different corpus types.
    if out_path.exists():
        out_name = f"{_safe_stem(Path(font_path).stem)}_{corpus_type}.png"
        out_path = out_dir / out_name
    image.save(out_path, format="PNG")
    return out_path


def main() -> None:
    _inject_src_to_path()
    from simple_rendering.config import load_config

    args = parse_args()
    config = load_config(args.config, font_category_override=args.font_category)
    config_dir = Path(args.config).expanduser().resolve().parent
    rows = _collect_rows(config, config_dir)
    if not rows:
        raise ValueError("No fonts found in selected category.")
    output = args.output or f"output/font_preview_{config.font_category}"
    out_dir = Path(output).expanduser().resolve()
    saved_paths: List[Path] = []
    for corpus_type, font_path, text in rows:
        saved_paths.append(_render_single_preview(corpus_type, font_path, text, out_dir))
    print(f"saved_count: {len(saved_paths)}")
    print(f"saved_dir: {out_dir}")


if __name__ == "__main__":
    main()
