from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps


def load_texture_paths(directory: Path) -> Sequence[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})


def build_glyph_mask(
    text: str,
    font: ImageFont.FreeTypeFont,
    draw: ImageDraw.ImageDraw,
    stroke_width: int = 0,
) -> Tuple[Image.Image, Tuple[int, int]]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font, embedded_color=True)
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return Image.new("L", (1, 1), 0), (0, 0)
    pad = max(8, int(max(width, height) * 0.08))
    mask = Image.new("L", (width + 2 * pad, height + 2 * pad), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.text(
        (pad - left, pad - top),
        text,
        fill=255,
        font=font,
        embedded_color=False,
        stroke_width=max(0, stroke_width),
        stroke_fill=255,
    )
    return mask, (left - pad, top - pad)


def compose_brush_texture(
    glyph_mask: Image.Image,
    brush_textures: Sequence[Path],
    color_rgb: Tuple[int, int, int],
    rng: random.Random,
    stroke_direction_jitter: float,
) -> Image.Image:
    w, h = glyph_mask.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if w <= 0 or h <= 0:
        return layer
    texture = _pick_or_generate_texture(brush_textures, (w, h), rng)
    angle = rng.uniform(-35.0, 35.0) * max(0.0, min(1.0, stroke_direction_jitter + 0.5))
    texture = texture.rotate(angle, expand=False, resample=Image.BICUBIC)
    texture = ImageOps.autocontrast(texture)
    alpha = ImageChops.multiply(glyph_mask, texture)
    alpha = ImageOps.autocontrast(alpha)
    painted = Image.new("RGBA", (w, h), (color_rgb[0], color_rgb[1], color_rgb[2], 255))
    painted.putalpha(alpha)
    return painted


def apply_feibai(
    layer: Image.Image,
    strength: float,
    rng: random.Random,
    stroke_direction_jitter: float,
) -> Image.Image:
    w, h = layer.size
    if w <= 1 or h <= 1:
        return layer
    alpha = layer.getchannel("A")
    noise = Image.effect_noise((w, h), 64.0).convert("L")
    angle = rng.uniform(-45.0, 45.0) * max(0.2, min(1.0, stroke_direction_jitter + 0.4))
    noise = noise.rotate(angle, expand=False, resample=Image.BICUBIC)
    noise = noise.filter(ImageFilter.GaussianBlur(radius=max(0.3, 1.8 * strength)))
    threshold = int(180 + 60 * max(0.0, min(1.0, strength)))
    cut = noise.point(lambda p: 255 if p > threshold else 0)
    cut = cut.filter(ImageFilter.MinFilter(size=3))
    feibai_alpha = ImageChops.subtract(alpha, ImageChops.multiply(alpha, cut))
    out = layer.copy()
    out.putalpha(feibai_alpha)
    return out


def apply_ink_bleed(layer: Image.Image, bleed_radius: float) -> Image.Image:
    if bleed_radius <= 0:
        return layer
    alpha = layer.getchannel("A")
    spread = alpha.filter(ImageFilter.GaussianBlur(radius=bleed_radius))
    spread = spread.point(lambda p: int(min(255, p * 0.72)))
    base = layer.copy()
    base.putalpha(ImageChops.lighter(alpha, spread))
    return base


def apply_edge_damage(layer: Image.Image, damage_strength: float, rng: random.Random) -> Image.Image:
    if damage_strength <= 0:
        return layer
    alpha = layer.getchannel("A")
    rough = Image.effect_noise(alpha.size, 100.0).convert("L")
    rough = rough.filter(ImageFilter.GaussianBlur(radius=1.0 + damage_strength * 2.0))
    threshold = int(140 + 80 * max(0.0, min(1.0, damage_strength)))
    mask = rough.point(lambda p: 255 if p > threshold else 0)
    # Randomly shift edge-noise mask to avoid repeated artifact.
    shift_x = rng.randint(-3, 3)
    shift_y = rng.randint(-3, 3)
    mask = ImageChops.offset(mask, shift_x, shift_y)
    damaged_alpha = ImageChops.subtract(alpha, ImageChops.multiply(alpha, mask))
    out = layer.copy()
    out.putalpha(damaged_alpha)
    return out


def warp_layer(
    layer: Image.Image,
    rng: random.Random,
    rotate_range_deg: float,
    shear_range: float,
    perspective_jitter_ratio: float,
) -> Image.Image:
    rotate = rng.uniform(-rotate_range_deg, rotate_range_deg)
    warped = layer.rotate(rotate, resample=Image.BICUBIC, expand=True)
    shear = rng.uniform(-shear_range, shear_range)
    w, h = warped.size
    out_w = max(1, w + int(abs(shear) * h))
    sheared = warped.transform(
        (out_w, h),
        Image.AFFINE,
        (1, shear, 0, 0, 1, 0),
        resample=Image.BICUBIC,
    )
    w2, h2 = sheared.size
    if w2 <= 1 or h2 <= 1:
        return sheared
    jitter = max(1, int(min(w2, h2) * max(0.0, perspective_jitter_ratio)))
    coeffs = (
        1,
        rng.uniform(-0.03, 0.03),
        rng.uniform(-jitter, jitter),
        rng.uniform(-0.03, 0.03),
        1,
        rng.uniform(-jitter, jitter),
    )
    return sheared.transform((w2, h2), Image.AFFINE, coeffs, resample=Image.BICUBIC)


def merge_with_paper(
    canvas: Image.Image,
    paper_textures: Sequence[Path],
    rng: random.Random,
) -> Image.Image:
    if not paper_textures:
        return canvas
    texture = _pick_or_generate_texture(paper_textures, canvas.size, rng)
    texture_rgb = ImageOps.colorize(texture, black="#D8D3C8", white="#FAF8F3").convert("RGB")
    return Image.blend(canvas, texture_rgb, alpha=0.08)


def _pick_or_generate_texture(paths: Sequence[Path], size: Tuple[int, int], rng: random.Random) -> Image.Image:
    if paths:
        chosen = rng.choice(list(paths))
        with Image.open(chosen) as img:
            gray = img.convert("L")
            return gray.resize(size, Image.BICUBIC)
    return Image.effect_noise(size, 64.0).convert("L")
