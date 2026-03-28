from __future__ import annotations

from pathlib import Path
import random
from typing import Optional

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

from .art_effects import (
    apply_edge_damage,
    apply_feibai,
    apply_ink_bleed,
    build_glyph_mask,
    compose_brush_texture,
    load_texture_paths,
    merge_with_paper,
    warp_layer,
)
from .config import CanvasConfig
from .layout import LayoutResult


def render_image(
    canvas_cfg: CanvasConfig,
    layout_result: LayoutResult,
    background_color: str,
    out_path: Path,
    background_image: Optional[Image.Image] = None,
) -> None:
    if background_image is not None:
        bg = background_image.convert("RGB")
        if bg.size != (canvas_cfg.width, canvas_cfg.height):
            bg = bg.resize((canvas_cfg.width, canvas_cfg.height), Image.Resampling.LANCZOS)
        image = bg
    else:
        image = Image.new("RGB", (canvas_cfg.width, canvas_cfg.height), background_color)
    # Optional paper texture pass for calligraphy pipeline.
    paper_paths = []
    if getattr(layout_result, "placements", None):
        first_effects = getattr(layout_result.placements[0], "effects", None) or {}
        calligraphy_cfg = first_effects.get("calligraphy")
        if isinstance(calligraphy_cfg, dict) and calligraphy_cfg.get("enabled", False):
            raw_paths = calligraphy_cfg.get("paper_texture_paths")
            if isinstance(raw_paths, list):
                paper_paths = [Path(p) for p in raw_paths]
    if paper_paths:
        image = merge_with_paper(image, paper_paths, random.Random(20260321))
    draw = ImageDraw.Draw(image)
    for placed in layout_result.placements:
        _draw_styled_text(image, draw, placed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="PNG")


def _draw_styled_text(image: Image.Image, draw: ImageDraw.ImageDraw, placed) -> None:
    effects = getattr(placed, "effects", None) or {}
    if effects:
        calligraphy_cfg = effects.get("calligraphy")
        if isinstance(calligraphy_cfg, dict) and calligraphy_cfg.get("enabled", False):
            _draw_with_calligraphy_effects(
                image=image, draw=draw, placed=placed, effects=effects, calligraphy_cfg=calligraphy_cfg
            )
        else:
            _draw_with_basic_effects(image=image, draw=draw, placed=placed, effects=effects)
        return
    style = placed.font_style or "normal"
    do_bold = style in {"bold", "bold_italic"}
    do_italic = style in {"italic", "bold_italic"}

    if do_italic:
        left, top, right, bottom = draw.textbbox(
            (0, 0), placed.text, font=placed.font, embedded_color=True
        )
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            return
        # Use a larger safety pad to avoid clipping edges after affine shear.
        pad = max(8, int(getattr(placed.font, "size", 24) * 0.18))
        # Extra margin on all sides keeps anti-aliased edge pixels inside buffer.
        base_w = width + 4 * pad
        base_h = height + 4 * pad
        text_img = Image.new("RGBA", (base_w, base_h), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_img)
        draw_x = 2 * pad - left
        draw_y = 2 * pad - top
        if do_bold:
            stroke = min(4, max(1, int(getattr(placed.font, "size", 24) * 0.02)))
            text_draw.text(
                (draw_x, draw_y),
                placed.text,
                fill=placed.color,
                font=placed.font,
                embedded_color=True,
                stroke_width=stroke,
                stroke_fill=placed.color,
            )
        else:
            text_draw.text(
                (draw_x, draw_y),
                placed.text,
                fill=placed.color,
                font=placed.font,
                embedded_color=True,
            )
        shear = 0.28
        out_w = base_w + int(abs(shear) * base_h) + 2 * pad
        italic_img = text_img.transform(
            (out_w, base_h),
            Image.AFFINE,
            (1, shear, 0, 0, 1, 0),
            resample=Image.BICUBIC,
        )
        # Keep the same glyph anchor behavior as draw.text:
        # glyph-left/top should be (placed.x + left, placed.y + top).
        paste_x = placed.x + left - 2 * pad
        paste_y = placed.y + top - 2 * pad
        image.paste(italic_img, (paste_x, paste_y), italic_img)
        return

    if do_bold:
        # Keep bold subtle for large font sizes.
        stroke = min(4, max(1, int(getattr(placed.font, "size", 24) * 0.02)))
        draw.text(
            (placed.x, placed.y),
            placed.text,
            fill=placed.color,
            font=placed.font,
            embedded_color=True,
            stroke_width=stroke,
            stroke_fill=placed.color,
        )
        return
    draw.text(
        (placed.x, placed.y),
        placed.text,
        fill=placed.color,
        font=placed.font,
        embedded_color=True,
    )


def _draw_with_calligraphy_effects(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    placed,
    effects: dict,
    calligraphy_cfg: dict,
) -> None:
    role = str(getattr(placed, "role", "body"))
    role_scale = 1.0 if role == "title" else 0.65
    seed_value = calligraphy_cfg.get("seed")
    rng = random.Random(int(seed_value)) if seed_value is not None else random.Random()
    font_size = max(1, int(getattr(placed.font, "size", 24)))
    base_stroke = int(max(1, font_size * 0.04 * role_scale))
    glyph_mask, anchor_delta = build_glyph_mask(
        text=placed.text,
        font=placed.font,
        draw=draw,
        stroke_width=base_stroke if "bold" in (placed.font_style or "") else 0,
    )
    if glyph_mask.size == (1, 1):
        return
    color_rgb = ImageColor_getrgb_safe(placed.color)
    brush_paths = [Path(p) for p in calligraphy_cfg.get("brush_texture_paths", [])] if isinstance(calligraphy_cfg.get("brush_texture_paths"), list) else []
    stroke_jitter = float(calligraphy_cfg.get("stroke_direction_jitter", 0.2))
    layer = compose_brush_texture(
        glyph_mask=glyph_mask,
        brush_textures=brush_paths,
        color_rgb=color_rgb,
        rng=rng,
        stroke_direction_jitter=stroke_jitter,
    )
    feibai_strength = float(calligraphy_cfg.get("feibai_strength", 0.35)) * role_scale
    layer = apply_feibai(
        layer=layer,
        strength=max(0.0, min(0.95, feibai_strength)),
        rng=rng,
        stroke_direction_jitter=stroke_jitter,
    )
    ink_bleed_radius = float(calligraphy_cfg.get("ink_bleed_radius", 1.2)) * (0.9 if role == "title" else 0.7)
    layer = apply_ink_bleed(layer=layer, bleed_radius=max(0.0, ink_bleed_radius))
    edge_damage_strength = float(calligraphy_cfg.get("edge_damage_strength", 0.2)) * role_scale
    layer = apply_edge_damage(layer=layer, damage_strength=max(0.0, min(0.95, edge_damage_strength)), rng=rng)
    layer = warp_layer(
        layer=layer,
        rng=rng,
        rotate_range_deg=float(calligraphy_cfg.get("rotate_range_deg", 5.0)) * role_scale,
        shear_range=float(calligraphy_cfg.get("shear_range", 0.1)) * role_scale,
        perspective_jitter_ratio=float(calligraphy_cfg.get("perspective_jitter_ratio", 0.05)) * role_scale,
    )
    left, top = anchor_delta
    paste_x = placed.x + left
    paste_y = placed.y + top
    shadow_cfg = effects.get("shadow", {})
    if isinstance(shadow_cfg, dict) and shadow_cfg.get("enabled", True):
        shadow_layer = _build_shadow_layer(layer, shadow_cfg, font_size)
        shadow_dx = int(round(float(shadow_cfg.get("offset_x", font_size * 0.04))))
        shadow_dy = int(round(float(shadow_cfg.get("offset_y", font_size * 0.06))))
        image.paste(shadow_layer, (paste_x + shadow_dx, paste_y + shadow_dy), shadow_layer)
    image.paste(layer, (paste_x, paste_y), layer)


def _draw_with_basic_effects(image: Image.Image, draw: ImageDraw.ImageDraw, placed, effects: dict) -> None:
    style = placed.font_style or "normal"
    do_bold = style in {"bold", "bold_italic"}
    do_italic = style in {"italic", "bold_italic"}
    left, top, right, bottom = draw.textbbox((0, 0), placed.text, font=placed.font, embedded_color=True)
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return
    font_size = max(1, int(getattr(placed.font, "size", 24)))
    pad = max(8, int(font_size * 0.18))
    layer_w = width + 2 * pad
    layer_h = height + 2 * pad
    text_layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_layer)

    outline_cfg = effects.get("outline", {})
    outline_enabled = isinstance(outline_cfg, dict) and bool(outline_cfg.get("enabled", False))
    outline_ratio = float(outline_cfg.get("width_ratio", 0.06)) if outline_enabled else 0.0
    outline_width = max(0, int(font_size * max(0.0, outline_ratio)))
    outline_color = str(outline_cfg.get("color", "#000000")) if outline_enabled else placed.color

    stroke = max(outline_width, min(4, max(1, int(font_size * 0.02))) if do_bold else 0)
    text_draw.text(
        (pad - left, pad - top),
        placed.text,
        fill=placed.color,
        font=placed.font,
        embedded_color=True,
        stroke_width=stroke,
        stroke_fill=outline_color if stroke > 0 else placed.color,
    )

    if do_italic:
        text_layer = _apply_shear(text_layer, shear=float(effects.get("italic_shear", 0.24)))
    distortion_cfg = effects.get("distortion", {})
    if isinstance(distortion_cfg, dict) and distortion_cfg.get("enabled", False):
        text_layer = _apply_distortion(text_layer, distortion_cfg)
    if isinstance(effects.get("texture"), dict) and effects.get("texture", {}).get("enabled", True):
        text_layer = _apply_texture(text_layer, effects["texture"])

    paste_x = placed.x + left - pad
    paste_y = placed.y + top - pad

    shadow_cfg = effects.get("shadow", {})
    if isinstance(shadow_cfg, dict) and shadow_cfg.get("enabled", False):
        shadow_layer = _build_shadow_layer(text_layer, shadow_cfg, font_size)
        shadow_dx = int(round(float(shadow_cfg.get("offset_x", font_size * 0.05))))
        shadow_dy = int(round(float(shadow_cfg.get("offset_y", font_size * 0.08))))
        image.paste(shadow_layer, (paste_x + shadow_dx, paste_y + shadow_dy), shadow_layer)

    image.paste(text_layer, (paste_x, paste_y), text_layer)

    reflection_cfg = effects.get("reflection", {})
    if isinstance(reflection_cfg, dict) and reflection_cfg.get("enabled", False):
        _draw_reflection_effect(
            image=image,
            text_layer=text_layer,
            paste_x=paste_x,
            paste_y=paste_y,
            font_size=font_size,
            reflection_cfg=reflection_cfg,
        )


def _apply_shear(layer: Image.Image, shear: float) -> Image.Image:
    base_w, base_h = layer.size
    out_w = base_w + int(abs(shear) * base_h)
    return layer.transform(
        (out_w, base_h),
        Image.AFFINE,
        (1, shear, 0, 0, 1, 0),
        resample=Image.BICUBIC,
    )


def _apply_distortion(layer: Image.Image, distortion_cfg: dict) -> Image.Image:
    seed_value = distortion_cfg.get("seed")
    rng = random.Random(int(seed_value)) if seed_value is not None else random.Random()
    rotate_range = float(distortion_cfg.get("rotate_range_deg", 6.0))
    shear_range = float(distortion_cfg.get("shear_range", 0.12))
    perspective_jitter = float(distortion_cfg.get("perspective_jitter_ratio", 0.06))

    rotated = layer.rotate(
        rng.uniform(-rotate_range, rotate_range),
        resample=Image.BICUBIC,
        expand=True,
    )
    shear = rng.uniform(-shear_range, shear_range)
    sheared = _apply_shear(rotated, shear=shear)

    w, h = sheared.size
    if w <= 1 or h <= 1:
        return sheared
    jitter = max(1, int(min(w, h) * max(0.0, perspective_jitter)))
    coeffs = (
        1,
        rng.uniform(-0.03, 0.03),
        rng.uniform(-jitter, jitter),
        rng.uniform(-0.03, 0.03),
        1,
        rng.uniform(-jitter, jitter),
    )
    return sheared.transform((w, h), Image.AFFINE, coeffs, resample=Image.BICUBIC)


def _apply_texture(layer: Image.Image, texture_cfg: dict) -> Image.Image:
    noise_strength = float(texture_cfg.get("noise_strength", 0.35))
    grain_size = max(1, int(texture_cfg.get("grain_size", 2)))
    w, h = layer.size
    if w <= 0 or h <= 0:
        return layer
    base_noise = Image.effect_noise((max(1, w // grain_size), max(1, h // grain_size)), 64.0)
    noise = base_noise.resize((w, h), Image.BICUBIC).convert("L")
    noise = ImageOps.autocontrast(noise)
    alpha = layer.getchannel("A")
    blend = Image.blend(alpha, ImageChops.multiply(alpha, noise), max(0.0, min(1.0, noise_strength)))
    out = layer.copy()
    out.putalpha(blend)
    return out


def _build_shadow_layer(layer: Image.Image, shadow_cfg: dict, font_size: int) -> Image.Image:
    blur_radius = float(shadow_cfg.get("blur_radius", max(1.0, font_size * 0.05)))
    color = str(shadow_cfg.get("color", "#000000"))
    alpha_ratio = float(shadow_cfg.get("alpha", 0.45))
    src_alpha = layer.getchannel("A")
    shadow_alpha = src_alpha.point(lambda px: int(max(0, min(255, px * alpha_ratio))))
    r, g, b = ImageColor_getrgb_safe(color)
    shadow = Image.new("RGBA", layer.size, (r, g, b, 0))
    shadow.putalpha(shadow_alpha)
    return shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def _draw_reflection_effect(
    image: Image.Image,
    text_layer: Image.Image,
    paste_x: int,
    paste_y: int,
    font_size: int,
    reflection_cfg: dict,
) -> None:
    alpha_ratio = float(reflection_cfg.get("alpha", 0.24))
    alpha_ratio = max(0.0, min(1.0, alpha_ratio))
    gap_ratio = float(reflection_cfg.get("gap_ratio", 0.08))
    gap = int(round(max(0.0, gap_ratio) * font_size))
    height_ratio = float(reflection_cfg.get("height_ratio", 0.45))
    height_ratio = max(0.1, min(1.0, height_ratio))
    blur_radius = float(reflection_cfg.get("blur_radius", 1.2))

    reflection = ImageOps.flip(text_layer.copy())
    rw, rh = reflection.size
    keep_h = max(1, int(round(rh * height_ratio)))
    reflection = reflection.crop((0, 0, rw, keep_h))
    if blur_radius > 0:
        reflection = reflection.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    alpha = reflection.getchannel("A")
    fade = Image.linear_gradient("L").resize((1, keep_h)).rotate(180, expand=True)
    fade = fade.resize((rw, keep_h), Image.BICUBIC)
    alpha = ImageChops.multiply(alpha, fade)
    alpha = alpha.point(lambda px: int(max(0, min(255, px * alpha_ratio))))
    reflection.putalpha(alpha)

    ref_x = paste_x
    ref_y = paste_y + text_layer.size[1] + gap
    image.paste(reflection, (ref_x, ref_y), reflection)


def ImageColor_getrgb_safe(color: str) -> tuple[int, int, int]:
    # Minimal safe parser for #RRGGBB / #RGB fallback.
    value = color.strip()
    if value.startswith("#"):
        value = value[1:]
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        return (0, 0, 0)
    try:
        return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
    except ValueError:
        return (0, 0, 0)
