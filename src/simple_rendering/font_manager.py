from __future__ import annotations

import copy
import math
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from fontTools.ttLib import TTCollection, TTFont, TTLibError
from PIL import Image, ImageDraw, ImageFont

from .color_utils import contrast_ratio
from .config import CanvasConfig, RenderConfig, resolve_config_path
from .corpus import CorpusItem

# title_body: one sample-wide decision for all title blocks (bold + shadow/effects).
_TITLE_BODY_GLOBAL_TITLE_BOLD = "title_body:title:__global_bold__"
_TITLE_BODY_GLOBAL_TITLE_EFFECTS = "title_body:title:__global_effects__"


@dataclass
class StyledSegment:
    text: str
    corpus_type: str
    font_path: str
    font_name: str
    color: str
    font_size: int
    font_style: str
    role: str = "body"
    effects: Optional[Dict[str, object]] = None


class FontCoverageManager:
    def __init__(self) -> None:
        self._font_cmap_cache: Dict[str, Set[int]] = {}
        self._renderable_cache: Dict[tuple[str, str, int], bool] = {}

    def supports_text(self, font_path: str, text: str) -> bool:
        cmap = self._get_cmap(font_path)
        for ch in text:
            if _skip_coverage_char(ch):
                continue
            if ord(ch) not in cmap:
                return False
        return True

    def is_renderable(self, font_path: str, text: str, requested_size: int) -> bool:
        if not text.strip():
            return True
        cache_key = (font_path, text, requested_size)
        if cache_key in self._renderable_cache:
            return self._renderable_cache[cache_key]

        probe_font = _load_font_with_size_fallback(font_path, requested_size)
        probe_img = Image.new("RGBA", (1024, 256), (255, 255, 255, 0))
        draw = ImageDraw.Draw(probe_img)
        left, top, right, bottom = draw.textbbox(
            (0, 0), text, font=probe_font, embedded_color=True
        )
        ok = (right - left) > 0 and (bottom - top) > 0
        self._renderable_cache[cache_key] = ok
        return ok

    def _get_cmap(self, font_path: str) -> Set[int]:
        if font_path in self._font_cmap_cache:
            return self._font_cmap_cache[font_path]

        codepoints: Set[int] = set()
        try:
            with TTFont(font_path, lazy=True) as font:
                cmap_table = font.getBestCmap() or {}
                codepoints.update(cmap_table.keys())
        except TTLibError as exc:
            try:
                with TTCollection(font_path, lazy=True) as collection:
                    for font in collection.fonts:
                        cmap_table = font.getBestCmap() or {}
                        codepoints.update(cmap_table.keys())
            except TTLibError as coll_exc:
                raise ValueError(f"Failed to load font file: {font_path}") from coll_exc

        self._font_cmap_cache[font_path] = codepoints
        return codepoints


def build_styled_segments(
    sampled_segments: Sequence[CorpusItem],
    config: RenderConfig,
    config_dir: Path,
    font_manager: FontCoverageManager,
    background_color: str,
    rng: random.Random,
    template_name: Optional[str] = None,
    sampled_canvas: Optional[CanvasConfig] = None,
) -> List[StyledSegment]:
    styled: List[StyledSegment] = []
    current_segment: List[CorpusItem] = []
    used_colors: Set[str] = set()
    role_style_cache: Dict[str, Dict[str, object]] = {}
    for item in sampled_segments:
        if item.corpus_type == "line_break":
            styled.extend(
                _style_one_segment(
                    current_segment=current_segment,
                    config=config,
                    config_dir=config_dir,
                    font_manager=font_manager,
                    background_color=background_color,
                    used_colors=used_colors,
                    rng=rng,
                    template_name=template_name,
                    sampled_canvas=sampled_canvas,
                    role_style_cache=role_style_cache,
                )
            )
            current_segment = []
            styled.append(
                StyledSegment(
                    text="\n",
                    corpus_type="line_break",
                    font_path=_pick_any_font_for_line_break(config, config_dir),
                    font_name="line_break",
                    color="#FFFFFF",
                    font_size=config.text.min_font_size,
                    font_style="normal",
                    role="line_break",
                    effects=None,
                )
            )
            continue
        current_segment.append(item)

    styled.extend(
        _style_one_segment(
            current_segment=current_segment,
            config=config,
            config_dir=config_dir,
            font_manager=font_manager,
            background_color=background_color,
            used_colors=used_colors,
            rng=rng,
            template_name=template_name,
            sampled_canvas=sampled_canvas,
            role_style_cache=role_style_cache,
        )
    )
    has_title_role = any(seg.role == "title" for seg in styled)
    has_body_role = any(seg.role == "body" for seg in styled)
    if template_name == "title_body" and (has_title_role or has_body_role):
        # Hard guarantee: titles and bodies each share one color globally in the sample.
        title_color = next((seg.color for seg in styled if seg.role == "title"), None)
        body_color = next((seg.color for seg in styled if seg.role == "body"), None)
        if title_color is not None:
            styled = [
                StyledSegment(
                    text=seg.text,
                    corpus_type=seg.corpus_type,
                    font_path=seg.font_path,
                    font_name=seg.font_name,
                    color=title_color if seg.role == "title" else seg.color,
                    font_size=seg.font_size,
                    font_style=seg.font_style,
                    role=seg.role,
                    effects=seg.effects,
                )
                for seg in styled
            ]
        if body_color is not None:
            styled = [
                StyledSegment(
                    text=seg.text,
                    corpus_type=seg.corpus_type,
                    font_path=seg.font_path,
                    font_name=seg.font_name,
                    color=body_color if seg.role == "body" else seg.color,
                    font_size=seg.font_size,
                    font_style=seg.font_style,
                    role=seg.role,
                    effects=seg.effects,
                )
                for seg in styled
            ]
    return styled


def _needs_injected_ascii_space_between_corpus_items(prev: CorpusItem, curr: CorpusItem) -> bool:
    """
    Force a visible ASCII space between corpus items when the pipeline did not insert one:
    - english | english: always unless boundary already has whitespace
    - emoji | emoji: never
    - english <-> emoji: unless boundary already has whitespace
    Chinese adjacency is unchanged (no automatic spaces).
    """
    if prev.corpus_type == "line_break" or curr.corpus_type == "line_break":
        return False
    pt, ct = prev.corpus_type, curr.corpus_type
    if pt == "emoji" and ct == "emoji":
        return False
    if pt == "english" and ct == "english":
        ps, cs = prev.content or "", curr.content or ""
        if ps and ps[-1].isspace():
            return False
        if cs and cs[0].isspace():
            return False
        return True
    if pt == "emoji" and ct == "english":
        cs = curr.content or ""
        if cs and cs[0].isspace():
            return False
        return True
    if pt == "english" and ct == "emoji":
        ps = prev.content or ""
        if ps and ps[-1].isspace():
            return False
        return True
    return False


def _style_one_segment(
    current_segment: Sequence[CorpusItem],
    config: RenderConfig,
    config_dir: Path,
    font_manager: FontCoverageManager,
    background_color: str,
    used_colors: Set[str],
    rng: random.Random,
    template_name: Optional[str] = None,
    sampled_canvas: Optional[CanvasConfig] = None,
    role_style_cache: Optional[Dict[str, Dict[str, object]]] = None,
) -> List[StyledSegment]:
    if not current_segment:
        return []

    text_items = [item for item in current_segment if item.corpus_type not in {"emoji", "line_break"}]
    if not text_items:
        return []
    base_corpus_type = text_items[0].corpus_type
    base_role = text_items[0].role if getattr(text_items[0], "role", "") else "body"
    role_cfg = _resolve_role_config(config, template_name, base_role)
    # For title_body:
    # - color should stay consistent per role across paragraphs
    # - font should stay consistent per role+corpus_type
    role_style_key = f"{template_name or ''}:{base_role}:{base_corpus_type}"
    role_color_key = f"{template_name or ''}:{base_role}:__color__"
    role_size_key = f"{template_name or ''}:{base_role}:__size__"
    cached = role_style_cache.get(role_style_key) if role_style_cache else None
    if cached is None:
        all_text = " ".join(item.content for item in text_items)
        min_size = int(role_cfg.get("min_font_size", config.text.min_font_size))
        max_size = int(role_cfg.get("max_font_size", config.text.max_font_size))
        if min_size > max_size:
            min_size, max_size = max_size, min_size
        min_size, max_size = _scale_font_size_range_by_canvas(
            min_size=min_size,
            max_size=max_size,
            config=config,
            sampled_canvas=sampled_canvas,
        )
        if template_name == "title_body" and role_style_cache is not None:
            if base_role == "title":
                body_size = role_style_cache.get("title_body:body:__size__", {}).get("font_size")
                if isinstance(body_size, int):
                    min_size = max(min_size, body_size + 10)
            elif base_role == "body":
                title_size = role_style_cache.get("title_body:title:__size__", {}).get("font_size")
                if isinstance(title_size, int):
                    max_size = min(max_size, title_size - 10)
            if min_size > max_size:
                # Keep hard constraint as much as possible in degenerate ranges.
                if base_role == "title":
                    min_size = max_size
                else:
                    max_size = min_size
        base_font_size = rng.randint(min_size, max_size)
        font_whitelist = role_cfg.get("font_whitelist")
        if not isinstance(font_whitelist, list):
            font_whitelist = None
        base_font = _pick_font_for_text(
            corpus_type=base_corpus_type,
            text=all_text,
            requested_size=base_font_size,
            config=config,
            config_dir=config_dir,
            font_manager=font_manager,
            rng=rng,
            font_whitelist=font_whitelist,
        )
        cached_role_color = (
            role_style_cache.get(role_color_key, {}).get("color")
            if role_style_cache is not None
            else None
        )
        if isinstance(cached_role_color, str) and cached_role_color:
            base_color = cached_role_color
        else:
            role_color_pool = role_cfg.get("color_pool")
            if isinstance(role_color_pool, list) and role_color_pool:
                base_color = _pick_color_from_pool(
                    [str(c) for c in role_color_pool],
                    config=config,
                    background_color=background_color,
                    used_colors=used_colors,
                    rng=rng,
                )
            else:
                base_color = _pick_color_for_corpus(
                    base_corpus_type, config, background_color, used_colors, rng
                )
            used_colors.add(base_color)
            if role_style_cache is not None and template_name in {"title_body"}:
                role_style_cache[role_color_key] = {"color": base_color}
        base_effects = _resolve_effects_config(config, template_name, base_role)
        if template_name == "title_body" and base_role == "title" and role_style_cache is not None:
            if _TITLE_BODY_GLOBAL_TITLE_EFFECTS in role_style_cache:
                base_effects = copy.deepcopy(role_style_cache[_TITLE_BODY_GLOBAL_TITLE_EFFECTS])
            else:
                base_effects = _maybe_add_random_title_shadow(
                    base_effects=base_effects,
                    template_name=template_name,
                    role=base_role,
                    font_size=base_font_size,
                    rng=rng,
                )
                role_style_cache[_TITLE_BODY_GLOBAL_TITLE_EFFECTS] = copy.deepcopy(base_effects)
        else:
            base_effects = _maybe_add_random_title_shadow(
                base_effects=base_effects,
                template_name=template_name,
                role=base_role,
                font_size=base_font_size,
                rng=rng,
            )
        if role_style_cache is not None and template_name in {"title_body"}:
            role_style_cache[role_style_key] = {
                "font_size": base_font_size,
                "font": base_font,
                "color": base_color,
                "effects": base_effects,
            }
            role_style_cache[role_size_key] = {"font_size": base_font_size}
    else:
        base_font_size = int(cached["font_size"])
        base_font = str(cached["font"])
        base_color = str(cached["color"])
        base_effects = cached.get("effects") if isinstance(cached, dict) else None

    # Random bold per segment (body); all titles in one image share one bold choice (title_body).
    if template_name == "title_body" and base_role in {"title", "body"}:
        bold_prob = 0.5
    else:
        bold_prob = float(role_cfg.get("bold_probability", 0.1))
    bold_prob = max(0.0, min(1.0, bold_prob))
    if template_name == "title_body" and base_role == "title" and role_style_cache is not None:
        if _TITLE_BODY_GLOBAL_TITLE_BOLD in role_style_cache:
            is_bold = bool(role_style_cache[_TITLE_BODY_GLOBAL_TITLE_BOLD])
        else:
            is_bold = rng.random() < bold_prob
            role_style_cache[_TITLE_BODY_GLOBAL_TITLE_BOLD] = is_bold
    else:
        is_bold = rng.random() < bold_prob
    base_font_style = "bold" if is_bold else "normal"

    out: List[StyledSegment] = []
    chinese_fallback_font = _pick_microsoft_yahei_fallback(
        config=config,
        config_dir=config_dir,
        font_manager=font_manager,
        requested_size=base_font_size,
    )
    symbol_fallback_font = _pick_special_symbol_fallback(
        config=config,
        config_dir=config_dir,
        font_manager=font_manager,
        requested_size=base_font_size,
    )
    prev_corpus_item: Optional[CorpusItem] = None
    for item in current_segment:
        if prev_corpus_item is not None and _needs_injected_ascii_space_between_corpus_items(
            prev_corpus_item, item
        ):
            out.append(
                StyledSegment(
                    text=" ",
                    corpus_type=base_corpus_type,
                    font_path=base_font,
                    font_name=Path(base_font).name,
                    color=base_color,
                    font_size=base_font_size,
                    font_style=base_font_style,
                    role=base_role,
                    effects=base_effects,
                )
            )
        if item.corpus_type == "emoji":
            emoji_font = _pick_font_for_text(
                corpus_type="emoji",
                text=item.content,
                requested_size=base_font_size,
                config=config,
                config_dir=config_dir,
                font_manager=font_manager,
                rng=rng,
            )
            out.append(
                StyledSegment(
                    text=item.content,
                    corpus_type=item.corpus_type,
                    font_path=emoji_font,
                    font_name=Path(emoji_font).name,
                    color=base_color,
                    font_size=base_font_size,
                    font_style="normal",
                    role=base_role,
                    effects=base_effects,
                )
            )
            prev_corpus_item = item
            continue
        else:
            # Character-level fallback chain:
            # base font -> Chinese fallback (Microsoft YaHei) -> special symbol fallback -> skip.
            out.extend(
                _build_text_segments_with_fallback(
                    text=item.content,
                    corpus_type=item.corpus_type,
                    base_font=base_font,
                    chinese_fallback_font=chinese_fallback_font,
                    symbol_fallback_font=symbol_fallback_font,
                    font_manager=font_manager,
                    requested_size=base_font_size,
                    base_color=base_color,
                    base_font_style=base_font_style,
                    base_role=base_role,
                    base_effects=base_effects,
                )
            )
            prev_corpus_item = item
    return out


def _pick_font_for_text(
    corpus_type: str,
    text: str,
    requested_size: int,
    config: RenderConfig,
    config_dir: Path,
    font_manager: FontCoverageManager,
    rng: random.Random,
    font_whitelist: Optional[List[str]] = None,
) -> str:
    candidates = config.fonts_by_corpus[corpus_type]
    resolved_candidates = [str(resolve_config_path(path, config_dir)) for path in candidates]
    if font_whitelist:
        lowered_rules = [rule.lower() for rule in font_whitelist]
        filtered_candidates = [
            path
            for path in resolved_candidates
            if any(rule in Path(path).name.lower() or rule in path.lower() for rule in lowered_rules)
        ]
        if filtered_candidates:
            resolved_candidates = filtered_candidates
    available = [
        path
        for path in resolved_candidates
        if font_manager.supports_text(path, text)
        and font_manager.is_renderable(path, text, requested_size)
    ]
    if not available:
        raise ValueError(f"No font supports full text for corpus_type={corpus_type}, text={text!r}")
    return rng.choice(available)


def _build_text_segments_with_fallback(
    text: str,
    corpus_type: str,
    base_font: str,
    chinese_fallback_font: Optional[str],
    symbol_fallback_font: Optional[str],
    font_manager: FontCoverageManager,
    requested_size: int,
    base_color: str,
    base_font_style: str,
    base_role: str,
    base_effects: Optional[Dict[str, object]],
) -> List[StyledSegment]:
    out: List[StyledSegment] = []
    buffer_chars: List[str] = []
    buffer_font: Optional[str] = None

    def _flush_buffer() -> None:
        nonlocal buffer_chars, buffer_font
        if not buffer_chars or not buffer_font:
            buffer_chars = []
            buffer_font = None
            return
        out.append(
            StyledSegment(
                text="".join(buffer_chars),
                corpus_type=corpus_type,
                font_path=buffer_font,
                font_name=Path(buffer_font).name,
                color=base_color,
                font_size=requested_size,
                font_style=base_font_style,
                role=base_role,
                effects=base_effects,
            )
        )
        buffer_chars = []
        buffer_font = None

    for ch in text:
        if _skip_coverage_char(ch):
            chosen_font = base_font
        elif font_manager.supports_text(base_font, ch) and font_manager.is_renderable(
            base_font, ch, requested_size
        ):
            chosen_font = base_font
        elif (
            _is_cjk_char(ch)
            and chinese_fallback_font
            and font_manager.supports_text(chinese_fallback_font, ch)
            and font_manager.is_renderable(chinese_fallback_font, ch, requested_size)
        ):
            chosen_font = chinese_fallback_font
        elif (
            _is_special_symbol_char(ch)
            and symbol_fallback_font
            and font_manager.supports_text(symbol_fallback_font, ch)
            and font_manager.is_renderable(symbol_fallback_font, ch, requested_size)
        ):
            chosen_font = symbol_fallback_font
        else:
            # If fallback fonts cannot render this character, skip it.
            _flush_buffer()
            continue
        if buffer_font is None:
            buffer_font = chosen_font
            buffer_chars.append(ch)
            continue
        if buffer_font == chosen_font:
            buffer_chars.append(ch)
            continue
        _flush_buffer()
        buffer_font = chosen_font
        buffer_chars.append(ch)
    _flush_buffer()
    return out


def _pick_microsoft_yahei_fallback(
    config: RenderConfig,
    config_dir: Path,
    font_manager: FontCoverageManager,
    requested_size: int,
) -> Optional[str]:
    candidates: List[str] = []
    configured_path = getattr(config, "fallback_chinese_font_path", None)
    if configured_path:
        candidates.append(str(resolve_config_path(str(configured_path), config_dir)))
    for path in config.fonts_by_corpus.get("chinese", []):
        resolved = str(resolve_config_path(path, config_dir))
        name = Path(resolved).name.lower()
        if "yahei" in name or "msyh" in name or "微软雅黑" in name:
            candidates.append(resolved)
    local_common = (config_dir / "../../font/common").resolve()
    extra = [
        local_common / "MicrosoftYaHei.ttf",
        local_common / "msyh.ttc",
        local_common / "msyh.ttf",
        Path("/Library/Fonts/Microsoft YaHei.ttf"),
        Path("/System/Library/Fonts/Supplemental/Microsoft YaHei.ttf"),
    ]
    for p in extra:
        if p.exists():
            candidates.append(str(p.resolve()))
    seen: Set[str] = set()
    unique_candidates: List[str] = []
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        unique_candidates.append(p)
    for font_path in unique_candidates:
        try:
            if font_manager.is_renderable(font_path, "测", requested_size):
                return font_path
        except Exception:
            continue
    return None


def _pick_special_symbol_fallback(
    config: RenderConfig,
    config_dir: Path,
    font_manager: FontCoverageManager,
    requested_size: int,
) -> Optional[str]:
    candidates: List[str] = []
    configured_path = getattr(config, "fallback_symbol_font_path", None)
    if configured_path:
        candidates.append(str(resolve_config_path(str(configured_path), config_dir)))
    # Prefer explicitly configured special/emoji fonts.
    for path in config.fonts_by_corpus.get("emoji", []):
        resolved = str(resolve_config_path(path, config_dir))
        if Path(resolved).exists():
            candidates.append(resolved)
    # Common symbol fonts on macOS.
    extra = [
        Path("/System/Library/Fonts/Apple Symbols.ttf"),
        Path("/System/Library/Fonts/Apple Color Emoji.ttc"),
        Path("/Library/Fonts/Symbola.ttf"),
    ]
    for p in extra:
        if p.exists():
            candidates.append(str(p.resolve()))
    seen: Set[str] = set()
    unique_candidates: List[str] = []
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        unique_candidates.append(p)
    for font_path in unique_candidates:
        try:
            # Probe with representative symbols.
            if (
                font_manager.is_renderable(font_path, "★", requested_size)
                or font_manager.is_renderable(font_path, "✓", requested_size)
                or font_manager.is_renderable(font_path, "→", requested_size)
            ):
                return font_path
        except Exception:
            continue
    return None


def _pick_color_for_corpus(
    corpus_type: str,
    config: RenderConfig,
    background_color: str,
    used_colors: Set[str],
    rng: random.Random,
) -> str:
    _ = corpus_type
    color_pool = config.text.default_text_colors
    contrast_ok_colors = [
        color
        for color in color_pool
        if contrast_ratio(color, background_color) >= config.canvas.min_text_bg_contrast_ratio
    ]
    if not contrast_ok_colors:
        raise ValueError(
            f"No text color meets contrast threshold for background={background_color}"
        )
    fresh_colors = [c for c in contrast_ok_colors if c not in used_colors]
    if fresh_colors:
        return rng.choice(fresh_colors)
    return rng.choice(contrast_ok_colors)


def _pick_color_from_pool(
    color_pool: Sequence[str],
    config: RenderConfig,
    background_color: str,
    used_colors: Set[str],
    rng: random.Random,
) -> str:
    contrast_ok_colors = [
        color
        for color in color_pool
        if contrast_ratio(color, background_color) >= config.canvas.min_text_bg_contrast_ratio
    ]
    if not contrast_ok_colors:
        raise ValueError(f"No role color meets contrast threshold against background={background_color}")
    fresh_colors = [c for c in contrast_ok_colors if c not in used_colors]
    if fresh_colors:
        return rng.choice(fresh_colors)
    return rng.choice(contrast_ok_colors)


def _pick_any_font_for_line_break(config: RenderConfig, config_dir: Path) -> str:
    fonts = config.fonts_by_corpus.get("line_break") or config.fonts_by_corpus.get("english") or []
    if not fonts:
        return ""
    return str(resolve_config_path(fonts[0], config_dir))


def _resolve_role_config(
    config: RenderConfig, template_name: Optional[str], role: str
) -> Dict[str, object]:
    if not template_name or not config.text.style_templates:
        return {}
    template_cfg = config.text.style_templates.get(template_name)
    if not isinstance(template_cfg, dict):
        return {}
    role_cfg = template_cfg.get(role)
    if not isinstance(role_cfg, dict):
        return {}
    return dict(role_cfg)


def _resolve_effects_config(
    config: RenderConfig, template_name: Optional[str], role: str
) -> Optional[Dict[str, object]]:
    if not template_name or not config.text.style_templates:
        return None
    template_cfg = config.text.style_templates.get(template_name)
    if not isinstance(template_cfg, dict):
        return None
    role_cfg = template_cfg.get(role)
    role_effects = role_cfg.get("effects") if isinstance(role_cfg, dict) else None
    template_effects = template_cfg.get("effects")
    resolved: Dict[str, object] = {}
    if isinstance(template_effects, dict):
        resolved.update(template_effects)
    if isinstance(role_effects, dict):
        resolved.update(role_effects)
    return resolved or None


def _maybe_add_random_title_shadow(
    base_effects: Optional[Dict[str, object]],
    template_name: Optional[str],
    role: str,
    font_size: int,
    rng: random.Random,
) -> Optional[Dict[str, object]]:
    # For title_body layout, randomly add shadow effect to title.
    if template_name != "title_body" or role != "title":
        return base_effects
    if rng.random() >= 0.5:
        return base_effects
    effects = dict(base_effects) if isinstance(base_effects, dict) else {}
    shadow_cfg = effects.get("shadow")
    if isinstance(shadow_cfg, dict):
        merged_shadow = dict(shadow_cfg)
        merged_shadow["enabled"] = True
        effects["shadow"] = merged_shadow
        return effects
    effects["shadow"] = {
        "enabled": True,
        "color": "#000000",
        "alpha": 0.32,
        "offset_x": round(max(1.0, font_size * 0.04), 2),
        "offset_y": round(max(1.0, font_size * 0.06), 2),
        "blur_radius": round(max(1.0, font_size * 0.05), 2),
    }
    return effects


def _skip_coverage_char(ch: str) -> bool:
    # Whitespace and emoji join/variation markers do not require dedicated glyph coverage.
    cp = ord(ch)
    if ch.isspace():
        return True
    if cp == 0x200D:  # ZWJ
        return True
    if 0xFE00 <= cp <= 0xFE0F:  # Variation selectors
        return True
    return False


def _is_cjk_char(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF
    )


def _is_special_symbol_char(ch: str) -> bool:
    cp = ord(ch)
    cat = unicodedata.category(ch)
    if cat.startswith("S"):
        return True
    # Extra symbol-heavy blocks.
    return (
        0x2190 <= cp <= 0x21FF  # arrows
        or 0x2200 <= cp <= 0x22FF  # math operators
        or 0x2460 <= cp <= 0x24FF  # enclosed alphanumerics
        or 0x25A0 <= cp <= 0x25FF  # geometric shapes
        or 0x2600 <= cp <= 0x26FF  # misc symbols
        or 0x2700 <= cp <= 0x27BF  # dingbats
    )


def _load_font_with_size_fallback(font_path: str, requested_size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(font_path, requested_size)
    except OSError as exc:
        if "invalid pixel size" not in str(exc):
            raise
    candidate_sizes = [20, 24, 28, 32, 36, 40, 44, 48, 56, 64, 96, 160]
    for size in sorted(candidate_sizes, key=lambda x: abs(x - requested_size)):
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    raise OSError(f"Unable to load font with a valid size: {font_path}")


def _scale_font_size_range_by_canvas(
    min_size: int,
    max_size: int,
    config: RenderConfig,
    sampled_canvas: Optional[CanvasConfig],
) -> tuple[int, int]:
    """
    Treat configured min/max font sizes as values when canvas area is 1024*1024.
    Use isotropic scaling factor derived from area so equivalent-area resize
    preserves configured perceived font size.
    """
    if sampled_canvas is None:
        return max(1, min_size), max(1, max_size)
    if sampled_canvas.width <= 0 or sampled_canvas.height <= 0:
        return max(1, min_size), max(1, max_size)
    base_area = 1024.0 * 1024.0
    area = float(sampled_canvas.width) * float(sampled_canvas.height)
    scale = math.sqrt(area / base_area)
    scale = max(0.1, scale)
    scaled_min = max(1, int(round(min_size * scale)))
    scaled_max = max(1, int(round(max_size * scale)))
    if scaled_min > scaled_max:
        scaled_min, scaled_max = scaled_max, scaled_min
    return scaled_min, scaled_max
