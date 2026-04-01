from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from .config import CanvasConfig, TextConfig
from .font_manager import StyledSegment

LEADING_PUNCTUATION = set("，。！？；：、,.!?;:)]}）】》”’")
TITLE_END_PUNCTUATION = set("，。！？；：、,.!?;:)]}）】》」』”’\"'…—-")


def _sample_segment_line_cap(text_cfg: TextConfig, rng: random.Random) -> int:
    lo = max(1, text_cfg.min_lines_cap_per_segment)
    hi = max(lo, text_cfg.max_lines_cap_per_segment)
    return rng.randint(lo, hi)


def _segment_line_caps_for(
    n: int, text_cfg: TextConfig, rng: random.Random
) -> List[int]:
    return [_sample_segment_line_cap(text_cfg, rng) for _ in range(n)]


def _exit_mixed_line_placements(
    placements: List[PlacedText], allow_partial_layout: bool
) -> Optional[List[PlacedText]]:
    """Strict: empty placements -> None. Partial: always return the list (may be empty)."""
    if allow_partial_layout:
        return placements
    return placements if placements else None


def _fail_mixed_line_placements(
    placements: List[PlacedText], allow_partial_layout: bool
) -> Optional[List[PlacedText]]:
    """On hard failure: partial keeps what was already placed; strict returns None."""
    if allow_partial_layout:
        return placements
    return None


@dataclass
class PlacedText:
    text: str
    x: int
    y: int
    color: str
    font: ImageFont.FreeTypeFont
    font_style: str
    role: str = "body"
    effects: Optional[Dict[str, object]] = None
    # When set (e.g. "ls"), (x, y) is the Pillow text anchor; default None uses legacy top-left.
    anchor: Optional[str] = None
    # Logical paragraph within this layout (line_break / commit boundaries); used for parquet export.
    paragraph_index: int = 0
    corpus_type: str = ""
    # Logical base font for this item; should remain stable even if per-char fallback happened.
    base_font_name: str = ""


@dataclass
class LayoutResult:
    font_size: int
    placements: List[PlacedText]
    rendering_text: str
    layout_variant: str
    rendering_text_meta_info: List[dict]
    template_name: Optional[str] = None


@dataclass
class _LineItem:
    text: str
    x: int
    width: int
    left: int
    top: int
    bottom: int
    color: str
    font: ImageFont.FreeTypeFont
    font_style: str
    corpus_type: str
    base_font_name: str
    role: str
    effects: Optional[Dict[str, object]]


@dataclass
class _VerticalItem:
    text: str
    y: int
    height: int
    width: int
    left: int
    top: int
    color: str
    font: ImageFont.FreeTypeFont
    font_style: str
    role: str
    effects: Optional[Dict[str, object]]
    corpus_type: str = ""


def layout_segments(
    segments: Sequence[StyledSegment],
    canvas: CanvasConfig,
    text_cfg: TextConfig,
    layout_mode: str,
    layout_variant: str = "",
    template_name: Optional[str] = None,
    allow_partial_layout: bool = False,
    rng: Optional[random.Random] = None,
) -> LayoutResult:
    if rng is None:
        rng = random.Random(0)
    measurer_img = Image.new("RGB", (canvas.width, canvas.height))
    draw = ImageDraw.Draw(measurer_img)
    fonts = _build_font_cache(segments)
    if layout_mode in {"mixed_line", "full_text"}:
        variant = layout_variant or "top_left"
        caps = _segment_line_caps_for(len(segments), text_cfg, rng)
        placements = _layout_mixed_line(
            segments,
            fonts,
            draw,
            canvas,
            text_cfg,
            variant,
            allow_partial_layout=allow_partial_layout,
            segment_line_caps=caps,
        )
        placements = _apply_block_vertical_anchor(
            placements, draw, canvas, variant, allow_partial_layout=allow_partial_layout
        )
        rendering_text = "".join(seg.text for seg in segments)
    elif layout_mode == "vertical":
        variant = layout_variant or "top_left@rtl"
        placements = _layout_vertical(segments, fonts, draw, canvas, text_cfg, variant)
        rendering_text = "".join(seg.text for seg in segments)
    elif layout_mode in {"segmented", "title_body"}:
        variant = layout_variant or "top_left"
        paragraph_spacing_override = None
        if layout_mode == "title_body":
            # paragraph_spacing is defined for canvas height=1024 and scales with current height.
            paragraph_spacing_override = max(
                1, int(round(text_cfg.paragraph_spacing * (canvas.height / 1024.0)))
            )
        placements = _layout_segmented(
            segments,
            fonts,
            draw,
            canvas,
            text_cfg,
            variant,
            paragraph_spacing_override=paragraph_spacing_override,
            allow_partial_layout=allow_partial_layout,
            rng=rng,
        )
        placements = _apply_block_vertical_anchor(
            placements, draw, canvas, variant, allow_partial_layout=allow_partial_layout
        )
        rendering_text = "\n".join(seg.text for seg in segments)
    elif layout_mode == "dual_column":
        variant = layout_variant or "top_left"
        placements = _layout_dual_column(
            segments,
            fonts,
            draw,
            canvas,
            text_cfg,
            variant,
            allow_partial_layout=allow_partial_layout,
            rng=rng,
        )
        placements = _apply_block_vertical_anchor(
            placements, draw, canvas, variant, allow_partial_layout=allow_partial_layout
        )
        rendering_text = "\n".join(seg.text for seg in segments)
    elif layout_mode == "title_subtitle":
        variant = layout_variant or "centered"
        placements = _layout_title_subtitle(segments, fonts, draw, canvas, text_cfg, variant)
        rendering_text = "\n".join(seg.text for seg in segments if seg.corpus_type != "line_break")
    else:
        raise ValueError(f"Unsupported layout mode: {layout_mode}")
    if placements is None:
        raise ValueError(
            "Unable to fit text into canvas with current settings. Consider larger canvas or fewer segments."
        )
    max_font_size = max((seg.font_size for seg in segments), default=text_cfg.max_font_size)
    return LayoutResult(
        font_size=max_font_size,
        placements=placements,
        rendering_text=rendering_text,
        layout_variant=variant,
        rendering_text_meta_info=[
            {
                "text": seg.text,
                "color": seg.color,
                "font_name": seg.font_name,
                "font_style": seg.font_style,
                "font_size": seg.font_size,
                "role": seg.role,
            }
            for seg in segments
        ],
        template_name=template_name,
    )


def _build_font_cache(segments: Sequence[StyledSegment]) -> Dict[Tuple[str, int], ImageFont.FreeTypeFont]:
    cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}
    for seg in segments:
        key = (seg.font_path, seg.font_size)
        if key not in cache:
            cache[key] = _load_font_with_size_fallback(seg.font_path, seg.font_size)
    return cache


def _load_font_with_size_fallback(font_path: str, requested_size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(font_path, requested_size)
    except OSError as exc:
        if "invalid pixel size" not in str(exc):
            raise
    # Some bitmap/color fonts (e.g. Apple Color Emoji) only support fixed sizes.
    candidate_sizes = [20, 32, 40, 48, 64, 96, 160]
    for size in sorted(candidate_sizes, key=lambda x: abs(x - requested_size)):
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    raise OSError(f"Unable to load font with a valid size: {font_path}")


def _layout_mixed_line(
    segments: Sequence[StyledSegment],
    fonts: Dict[Tuple[str, int], ImageFont.FreeTypeFont],
    draw: ImageDraw.ImageDraw,
    canvas: CanvasConfig,
    text_cfg: TextConfig,
    variant: str,
    width_ratio_range: Optional[Tuple[float, float]] = None,
    allow_partial_layout: bool = False,
    paragraph_index_offset: int = 0,
    segment_line_caps: Optional[Sequence[int]] = None,
) -> Optional[List[PlacedText]]:
    placements: List[PlacedText] = []
    x = canvas.margin
    y = canvas.margin
    h_align, _, justify = _parse_horizontal_variant(variant)
    full_draw_width = canvas.width - 2 * canvas.margin
    justify_width_mode = "uniform"
    if justify:
        justify_width_mode = _pick_justify_width_mode(variant, canvas, segments)
    uniform_target_width = (
        _sample_justify_target_width(f"{variant}|{canvas.width}|{canvas.height}", full_draw_width)
        if justify
        else full_draw_width
    )
    paragraph_index = 0
    current_target_width = uniform_target_width
    if width_ratio_range is not None:
        min_ratio, max_ratio = width_ratio_range
        current_target_width = _sample_width_by_ratio(
            f"{variant}|{canvas.width}|{canvas.height}|p{paragraph_index}",
            full_draw_width,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
        )
    if justify and justify_width_mode == "per_paragraph":
        current_target_width = _sample_justify_target_width(
            f"{variant}|{canvas.width}|{canvas.height}|p{paragraph_index}",
            full_draw_width,
        )
    current_line: List[_LineItem] = []
    paragraph_lines: List[Tuple[List[_LineItem], int, int]] = []

    def _append_line_to_paragraph(items: List[_LineItem], line_y: int, line_height: int) -> None:
        paragraph_lines.append((list(items), line_y, line_height))

    def _render_line(
        items: List[_LineItem],
        line_y: int,
        line_height: int,
        justify_line: bool,
        line_paragraph_index: int,
    ) -> None:
        line_width = _measure_line_width_with_inline_gaps(items)
        gap_slacks: Optional[List[int]] = None
        if justify_line:
            gap_slacks = _justify_inter_word_slacks(items, current_target_width)
            if gap_slacks is None:
                justify_line = False
        render_width = current_target_width if justify_line and line_width <= current_target_width else line_width
        start_x = _pick_line_start_x(h_align, canvas, render_width, current_target_width)
        baseline_y = _baseline_y_for_line_items(items, line_y)
        if justify_line and gap_slacks is not None:
            x_cursor = start_x
            for idx, item in enumerate(items):
                if idx > 0:
                    x_cursor += _emoji_text_inline_gap(items[idx - 1], item)
                x_draw = x_cursor - item.left
                y_draw, anch = _inline_item_y_and_anchor(item, line_y, line_height, baseline_y)
                placements.append(
                    PlacedText(
                        text=item.text,
                        x=x_draw,
                        y=y_draw,
                        color=item.color,
                        font=item.font,
                        font_style=item.font_style,
                        role=getattr(item, "role", "body"),
                        effects=getattr(item, "effects", None),
                        anchor=anch,
                        paragraph_index=line_paragraph_index,
                        corpus_type=item.corpus_type,
                        base_font_name=item.base_font_name,
                    )
                )
                x_cursor += item.width
                if idx < len(items) - 1:
                    x_cursor += gap_slacks[idx]
        if not justify_line:
            x_cursor = start_x
            for idx, item in enumerate(items):
                if idx > 0:
                    x_cursor += _emoji_text_inline_gap(items[idx - 1], item)
                x_draw = x_cursor - item.left
                y_draw, anch = _inline_item_y_and_anchor(item, line_y, line_height, baseline_y)
                placements.append(
                    PlacedText(
                        text=item.text,
                        x=x_draw,
                        y=y_draw,
                        color=item.color,
                        font=item.font,
                        font_style=item.font_style,
                        role=getattr(item, "role", "body"),
                        effects=getattr(item, "effects", None),
                        anchor=anch,
                        paragraph_index=line_paragraph_index,
                        corpus_type=item.corpus_type,
                        base_font_name=item.base_font_name,
                    )
                )
                x_cursor += item.width

    def _commit_paragraph() -> bool:
        nonlocal y, paragraph_index, current_target_width
        if not paragraph_lines:
            return True
        lines_to_render = list(paragraph_lines)
        paragraph_justify_line = justify
        if justify:
            # For justify mode: if the paragraph wraps to only one line, degrade to h_align
            # (left/center/right from variant) instead of stretching to full target width.
            # If it wraps to 2+ lines, drop the last line for balance, then justify the remainder.
            if len(lines_to_render) < 2:
                paragraph_justify_line = False
            else:
                _, _, dropped_h = lines_to_render.pop()
                y -= dropped_h + text_cfg.line_spacing
        pidx = paragraph_index + paragraph_index_offset
        for items, line_y, line_h in lines_to_render:
            _render_line(
                items,
                line_y,
                line_h,
                justify_line=paragraph_justify_line,
                line_paragraph_index=pidx,
            )
        paragraph_lines.clear()
        paragraph_index += 1
        if width_ratio_range is not None:
            min_ratio, max_ratio = width_ratio_range
            current_target_width = _sample_width_by_ratio(
                f"{variant}|{canvas.width}|{canvas.height}|p{paragraph_index}",
                full_draw_width,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
            )
        elif justify and justify_width_mode == "per_paragraph":
            current_target_width = _sample_justify_target_width(
                f"{variant}|{canvas.width}|{canvas.height}|p{paragraph_index}",
                full_draw_width,
            )
        else:
            current_target_width = uniform_target_width
        return True

    def _emit_buffered_paragraph_lines_if_partial() -> None:
        # paragraph_lines are only copied into placements inside _commit_paragraph. If we hit a
        # vertical overflow before the final commit (common for long single-paragraph full_text),
        # placements would stay empty unless we flush the buffer here.
        if allow_partial_layout and paragraph_lines:
            _commit_paragraph()

    def flush_final_line() -> bool:
        """Flush without per-segment line cap (end of document)."""
        nonlocal y
        if not current_line:
            return True
        line_height = max(item.bottom - item.top for item in current_line)
        if y + line_height > canvas.height - canvas.margin:
            return False
        _append_line_to_paragraph(current_line, y, line_height)
        y += line_height + text_cfg.line_spacing
        current_line.clear()
        return True

    for seg_idx, seg in enumerate(segments):
        cap = (
            segment_line_caps[seg_idx]
            if segment_line_caps is not None and seg_idx < len(segment_line_caps)
            else 10**9
        )
        lines_used = [0]

        def flush_current_line() -> bool:
            nonlocal y
            if not current_line:
                return True
            if lines_used[0] >= cap:
                current_line.clear()
                return True
            line_height = max(item.bottom - item.top for item in current_line)
            if y + line_height > canvas.height - canvas.margin:
                return False
            _append_line_to_paragraph(current_line, y, line_height)
            y += line_height + text_cfg.line_spacing
            lines_used[0] += 1
            current_line.clear()
            return True

        font = fonts[(seg.font_path, seg.font_size)]
        tokens = _split_text_for_line_wrapping(seg.text)
        for token in tokens:
            if lines_used[0] >= cap:
                break
            if token == "\n":
                if not flush_current_line():
                    _emit_buffered_paragraph_lines_if_partial()
                    return _exit_mixed_line_placements(placements, allow_partial_layout)
                if not _commit_paragraph():
                    _emit_buffered_paragraph_lines_if_partial()
                    return _fail_mixed_line_placements(placements, allow_partial_layout)
                y += text_cfg.paragraph_spacing
                x = canvas.margin
                continue
            chunks = _wrap_token_to_width(
                token,
                font,
                draw,
                current_target_width,
                preserve_word=seg.corpus_type == "english",
            )
            if chunks is None:
                _emit_buffered_paragraph_lines_if_partial()
                return _fail_mixed_line_placements(placements, allow_partial_layout)
            for chunk in chunks:
                if seg.corpus_type == "english":
                    # One wrapped line must become word-sized items so justify can insert slack
                    # between words (same convention as _build_segment_line_items).
                    wds = [w for w in chunk.split() if w]
                    if not wds:
                        # Whitespace-only chunks (explicit space StyledSegment / corpus inject) must
                        # not vanish: split() drops them and previously removed all inter-word gaps.
                        units = [" "] if (chunk != "" and all(c.isspace() for c in chunk)) else []
                    else:
                        units = [f"{w} " if j < len(wds) - 1 else w for j, w in enumerate(wds)]
                else:
                    units = list(chunk)
                for unit in units:
                    bbox_anchor = None if seg.corpus_type == "emoji" else "ls"
                    left, top, right, bottom = _measure_text_bbox(draw, unit, font, anchor=bbox_anchor)
                    width = right - left
                    advance = (
                        _horizontal_advance_ls(draw, font, unit, left, right)
                        if bbox_anchor == "ls"
                        else max(width, int(round(draw.textlength(unit, font=font))))
                    )
                    height = bottom - top
                    # PIL may report height 0 for whitespace (e.g. space); only treat as hard
                    # failure for non-whitespace glyphs that should have ink.
                    if unit.strip() and (width <= 0 or height <= 0):
                        _emit_buffered_paragraph_lines_if_partial()
                        return _fail_mixed_line_placements(placements, allow_partial_layout)
                    gap = 0
                    if current_line:
                        gap = _emoji_text_inline_gap(
                            current_line[-1],
                            None,
                            next_text=unit,
                            next_corpus_type=seg.corpus_type,
                            next_font_style=seg.font_style,
                        )
                    remain = canvas.margin + current_target_width - x
                    if gap + advance > remain and x > canvas.margin:
                        if _is_leading_punctuation_unit(unit) and current_line:
                            carry = current_line.pop()
                            # Keep punctuation away from line start by carrying previous glyph down together.
                            if not flush_current_line():
                                _emit_buffered_paragraph_lines_if_partial()
                                return _exit_mixed_line_placements(placements, allow_partial_layout)
                            x = canvas.margin
                            current_line.append(
                                _LineItem(
                                    text=carry.text,
                                    x=x,
                                    width=carry.width,
                                    left=carry.left,
                                    top=carry.top,
                                    bottom=carry.bottom,
                                    color=carry.color,
                                    font=carry.font,
                                    font_style=carry.font_style,
                                    corpus_type=carry.corpus_type,
                                    base_font_name=carry.base_font_name,
                                    role=carry.role,
                                    effects=carry.effects,
                                )
                            )
                            x += carry.width
                            gap = _emoji_text_inline_gap(
                                current_line[-1],
                                None,
                                next_text=unit,
                                next_corpus_type=seg.corpus_type,
                                next_font_style=seg.font_style,
                            )
                        else:
                            if not flush_current_line():
                                _emit_buffered_paragraph_lines_if_partial()
                                return _exit_mixed_line_placements(placements, allow_partial_layout)
                            x = canvas.margin
                            gap = 0
                    x += gap
                    current_line.append(
                        _LineItem(
                            text=unit,
                            x=x,
                            width=advance,
                            left=left,
                            top=top,
                            bottom=bottom,
                            color=seg.color,
                            font=font,
                            font_style=seg.font_style,
                            corpus_type=seg.corpus_type,
                            base_font_name=seg.base_font_name,
                            role=seg.role,
                            effects=seg.effects,
                        )
                    )
                    x += advance
    if not flush_final_line():
        _emit_buffered_paragraph_lines_if_partial()
        return _exit_mixed_line_placements(placements, allow_partial_layout)
    if not _commit_paragraph():
        _emit_buffered_paragraph_lines_if_partial()
        return _fail_mixed_line_placements(placements, allow_partial_layout)

    return placements


def _title_body_block_has_following_body(segments: Sequence[StyledSegment], seg_idx: int) -> bool:
    """
    True when segments form ... title+, line_break, body... starting at seg_idx (any title segment
    in the run). Chinese titles are split into multiple StyledSegments with role=title, so we must
    scan past consecutive title segments before expecting line_break (see font_manager fallback).
    """
    j = seg_idx + 1
    while j < len(segments) and getattr(segments[j], "role", "") == "title":
        j += 1
    if j >= len(segments) or getattr(segments[j], "corpus_type", "") != "line_break":
        return False
    if j + 1 >= len(segments):
        return False
    return getattr(segments[j + 1], "role", "") == "body"


def _build_merged_title_line_items(
    line: str,
    group_segments: Sequence[StyledSegment],
    fonts: Dict[Tuple[str, int], ImageFont.FreeTypeFont],
    draw: ImageDraw.ImageDraw,
) -> Optional[List[_LineItem]]:
    """Place one logical title line using per-segment fonts (Chinese title may be many segments)."""
    if not line:
        return []
    if len(group_segments) == 1:
        seg = group_segments[0]
        font = fonts[(seg.font_path, seg.font_size)]
        return _build_segment_line_items(line, seg, draw, font)
    if group_segments[0].corpus_type == "english":
        seg = group_segments[0]
        font = fonts[(seg.font_path, seg.font_size)]
        return _build_segment_line_items(line, seg, draw, font)
    items: List[_LineItem] = []
    remaining = line
    for seg in group_segments:
        if not remaining:
            break
        font = fonts[(seg.font_path, seg.font_size)]
        st = seg.text
        take = 0
        while take < len(st) and take < len(remaining) and st[take] == remaining[take]:
            take += 1
        if take == 0:
            seg0 = group_segments[0]
            font0 = fonts[(seg0.font_path, seg0.font_size)]
            return _build_segment_line_items(line, seg0, draw, font0)
        sub = remaining[:take]
        sub_items = _build_segment_line_items(sub, seg, draw, font)
        if sub_items is None:
            return None
        items.extend(sub_items)
        remaining = remaining[take:]
    if remaining:
        seg0 = group_segments[0]
        font0 = fonts[(seg0.font_path, seg0.font_size)]
        sub_items = _build_segment_line_items(remaining, seg0, draw, font0)
        if sub_items is None:
            return None
        items.extend(sub_items)
    return items


def _layout_segmented(
    segments: Sequence[StyledSegment],
    fonts: Dict[Tuple[str, int], ImageFont.FreeTypeFont],
    draw: ImageDraw.ImageDraw,
    canvas: CanvasConfig,
    text_cfg: TextConfig,
    variant: str,
    paragraph_spacing_override: Optional[int] = None,
    width_ratio_range: Optional[Tuple[float, float]] = None,
    allow_partial_layout: bool = False,
    rng: Optional[random.Random] = None,
) -> Optional[List[PlacedText]]:
    if rng is None:
        rng = random.Random(0)
    paragraph_spacing = (
        paragraph_spacing_override if paragraph_spacing_override is not None else text_cfg.paragraph_spacing
    )
    placements: List[PlacedText] = []
    h_align, _, justify = _parse_horizontal_variant(variant)
    y = canvas.margin
    full_draw_width = canvas.width - 2 * canvas.margin
    justify_width_mode = "uniform"
    if justify:
        justify_width_mode = _pick_justify_width_mode(variant, canvas, segments)
    uniform_target_width = (
        _sample_justify_target_width(f"{variant}|{canvas.width}|{canvas.height}", full_draw_width)
        if justify
        else full_draw_width
    )
    pending_title_checkpoint: Optional[Tuple[int, int]] = None
    paragraph_idx = 0
    consumed_body_seg_idxs: set[int] = set()
    consumed_title_seg_idxs: set[int] = set()
    skip_body_until_next_title = False
    is_title_body_mode = paragraph_spacing_override is not None

    def _skip_current_paragraph_block() -> None:
        nonlocal y, pending_title_checkpoint, skip_body_until_next_title
        if pending_title_checkpoint is not None:
            p_len, y0 = pending_title_checkpoint
            del placements[p_len:]
            y = y0
        pending_title_checkpoint = None
        skip_body_until_next_title = True

    for seg_idx, seg in enumerate(segments):
        if seg_idx in consumed_body_seg_idxs:
            continue
        if seg_idx in consumed_title_seg_idxs:
            continue
        if skip_body_until_next_title:
            if seg.role == "title":
                skip_body_until_next_title = False
            elif seg.role == "body":
                # A prior failure set skip_body_until_next_title; if we still owe a title rollback
                # (e.g. multi-segment Chinese title with checkpoint on first glyph), drop the orphan
                # title before skipping remaining body segments for this block.
                if pending_title_checkpoint is not None:
                    _skip_current_paragraph_block()
                continue
        if seg.corpus_type == "line_break":
            prev_role = segments[seg_idx - 1].role if seg_idx > 0 else ""
            next_role = segments[seg_idx + 1].role if seg_idx + 1 < len(segments) else ""
            # Keep title->body gap compact, while preserving visible gap between blocks.
            if prev_role == "title" and next_role == "body":
                y += max(6, text_cfg.line_spacing)
            else:
                y += paragraph_spacing
                paragraph_idx += 1
            if y > canvas.height - canvas.margin:
                return _exit_mixed_line_placements(placements, allow_partial_layout)
            continue
        line_target_width = uniform_target_width
        if justify and justify_width_mode == "per_paragraph":
            line_target_width = _sample_justify_target_width(
                f"{variant}|{canvas.width}|{canvas.height}|p{paragraph_idx}",
                full_draw_width,
            )
        if width_ratio_range is not None:
            min_ratio, max_ratio = width_ratio_range
            line_target_width = _sample_width_by_ratio(
                f"{variant}|{canvas.width}|{canvas.height}|p{paragraph_idx}",
                full_draw_width,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
            )
        if seg.role == "body":
            body_group_end = seg_idx
            while (
                body_group_end + 1 < len(segments)
                and segments[body_group_end + 1].role == "body"
                and segments[body_group_end + 1].corpus_type != "line_break"
            ):
                body_group_end += 1
            if body_group_end > seg_idx:
                group_segments = list(segments[seg_idx : body_group_end + 1])
                group_caps = _segment_line_caps_for(len(group_segments), text_cfg, rng)
                group_canvas = CanvasConfig(
                    width=canvas.width,
                    height=canvas.height,
                    min_width=None,
                    max_width=None,
                    min_height=None,
                    max_height=None,
                    short_edge_min=None,
                    short_edge_max=None,
                    aspect_ratios=None,
                    margin=canvas.margin,
                    background_colors=[],
                    min_text_bg_contrast_ratio=canvas.min_text_bg_contrast_ratio,
                    background_images_dir=None,
                    background_image_area_reference=canvas.background_image_area_reference,
                )
                group_placements = _layout_mixed_line(
                    segments=group_segments,
                    fonts=fonts,
                    draw=draw,
                    canvas=group_canvas,
                    text_cfg=text_cfg,
                    variant=variant,
                    width_ratio_range=width_ratio_range,
                    allow_partial_layout=allow_partial_layout,
                    paragraph_index_offset=paragraph_idx,
                    segment_line_caps=group_caps,
                )
                if group_placements is None:
                    if allow_partial_layout and placements:
                        return placements
                    _skip_current_paragraph_block()
                    for k in range(seg_idx + 1, body_group_end + 1):
                        consumed_body_seg_idxs.add(k)
                    continue
                dy = y - canvas.margin
                shifted_group: List[PlacedText] = []
                max_bottom = -1
                body_group_skip_outer = False
                for p in group_placements:
                    shifted = PlacedText(
                        text=p.text,
                        x=p.x,
                        y=p.y + dy,
                        color=p.color,
                        font=p.font,
                        font_style=p.font_style,
                        role=p.role,
                        effects=p.effects,
                        anchor=getattr(p, "anchor", None),
                        paragraph_index=getattr(p, "paragraph_index", 0),
                        corpus_type=getattr(p, "corpus_type", ""),
                        base_font_name=getattr(p, "base_font_name", ""),
                    )
                    l, t, r, b = draw.textbbox((shifted.x, shifted.y), shifted.text, font=shifted.font, embedded_color=True)
                    if b > canvas.height - canvas.margin:
                        if allow_partial_layout:
                            if shifted_group:
                                placements.extend(shifted_group)
                                y = max(y, max_bottom)
                                pending_title_checkpoint = None
                                for k in range(seg_idx + 1, body_group_end + 1):
                                    consumed_body_seg_idxs.add(k)
                                return placements
                            if placements:
                                return placements
                            _skip_current_paragraph_block()
                            for k in range(seg_idx + 1, body_group_end + 1):
                                consumed_body_seg_idxs.add(k)
                            body_group_skip_outer = True
                            break
                        _skip_current_paragraph_block()
                        shifted_group = []
                        break
                    max_bottom = max(max_bottom, b)
                    shifted_group.append(shifted)
                if body_group_skip_outer:
                    continue
                if shifted_group:
                    placements.extend(shifted_group)
                    y = max(y, max_bottom)
                    pending_title_checkpoint = None
                    for k in range(seg_idx + 1, body_group_end + 1):
                        consumed_body_seg_idxs.add(k)
                    continue
        if seg.role == "title":
            title_end = seg_idx
            while title_end + 1 < len(segments) and segments[title_end + 1].role == "title":
                title_end += 1
            group_segments = list(segments[seg_idx : title_end + 1])
            has_following_body = _title_body_block_has_following_body(segments, seg_idx)
            is_first_title_in_block = seg_idx == 0 or segments[seg_idx - 1].role != "title"
            if has_following_body and is_first_title_in_block:
                pending_title_checkpoint = (len(placements), y)
            elif has_following_body and not is_first_title_in_block:
                pass
            else:
                pending_title_checkpoint = None
            merged_font = fonts[(group_segments[0].font_path, group_segments[0].font_size)]
            merged_raw = "".join(s.text for s in group_segments)
            title_text = _strip_trailing_title_punctuation(merged_raw)
            title_line = _truncate_text_to_single_line(
                title_text,
                font=merged_font,
                draw=draw,
                max_width=line_target_width,
                preserve_word=group_segments[0].corpus_type == "english",
            )
            lines = [title_line] if title_line else []
            max_lines_cap = _sample_segment_line_cap(text_cfg, rng)
            if isinstance(lines, list) and lines:
                lines = lines[:max_lines_cap]
            if not lines:
                if pending_title_checkpoint is not None:
                    p_len, y0 = pending_title_checkpoint
                    del placements[p_len:]
                    y = y0
                pending_title_checkpoint = None
                skip_body_until_next_title = True
                for k in range(seg_idx + 1, title_end + 1):
                    consumed_title_seg_idxs.add(k)
                continue
            for line_idx, line in enumerate(lines):
                line_items = _build_merged_title_line_items(line, group_segments, fonts, draw)
                if line_items is None:
                    if allow_partial_layout and placements:
                        return placements
                    _skip_current_paragraph_block()
                    break
                if not line_items:
                    if allow_partial_layout and placements:
                        return placements
                    _skip_current_paragraph_block()
                    break
                width = _measure_line_width_with_inline_gaps(line_items)
                line_height = max(item.bottom - item.top for item in line_items)
                if line and width <= 0:
                    if allow_partial_layout and placements:
                        return placements
                    _skip_current_paragraph_block()
                    break
                if line_height <= 0:
                    if allow_partial_layout and placements:
                        return placements
                    _skip_current_paragraph_block()
                    break
                if width > line_target_width:
                    if allow_partial_layout and placements:
                        return placements
                    _skip_current_paragraph_block()
                    break
                if y + line_height > canvas.height - canvas.margin:
                    if allow_partial_layout and placements:
                        return placements
                    _skip_current_paragraph_block()
                    break
                body_multi_line = False
                justify_line = False
                gap_slacks: Optional[List[int]] = None
                render_width = line_target_width if justify_line and width <= line_target_width else width
                start_x = _pick_line_start_x(h_align, canvas, render_width, line_target_width)
                x_cursor = start_x
                baseline_y = _baseline_y_for_line_items(line_items, y)
                for idx, item in enumerate(line_items):
                    if idx > 0:
                        x_cursor += _emoji_text_inline_gap(line_items[idx - 1], item)
                    y_draw, anch = _inline_item_y_and_anchor(item, y, line_height, baseline_y)
                    placements.append(
                        PlacedText(
                            text=item.text,
                            x=x_cursor - item.left,
                            y=y_draw,
                            color=item.color,
                            font=item.font,
                            font_style=item.font_style,
                            role=item.role,
                            effects=item.effects,
                            anchor=anch,
                            paragraph_index=paragraph_idx,
                            corpus_type=item.corpus_type,
                            base_font_name=item.base_font_name,
                        )
                    )
                    x_cursor += item.width
                    if justify_line and gap_slacks is not None and idx < len(line_items) - 1:
                        x_cursor += gap_slacks[idx]
                y += line_height + text_cfg.line_spacing
            for k in range(seg_idx + 1, title_end + 1):
                consumed_title_seg_idxs.add(k)
            if skip_body_until_next_title:
                continue
            y -= text_cfg.line_spacing
            next_is_line_break = title_end < len(segments) - 1 and segments[title_end + 1].corpus_type == "line_break"
            if title_end < len(segments) - 1 and not next_is_line_break:
                next_role = segments[title_end + 1].role
                if not (seg.role == "body" and next_role == "body"):
                    y += paragraph_spacing
            if y > canvas.height - canvas.margin:
                return _exit_mixed_line_placements(placements, allow_partial_layout)
            continue

        font = fonts[(seg.font_path, seg.font_size)]
        lines = _wrap_text_to_lines(
            seg.text,
            font,
            draw,
            line_target_width,
            preserve_word=seg.corpus_type == "english",
        )
        # For title_body in justify mode: body should first form at least 2 lines,
        # then drop the last line for cleaner visual balance. A single wrapped line uses
        # h_align only (see justify_line below), not full justify.
        if justify and seg.role == "body":
            if lines is None:
                # Keep None semantics: wrapping failed (e.g. unbreakable token too long).
                pass
            elif len(lines) >= 2:
                # Do not drop the last line if that would leave only empty lines (orphan title + no body).
                truncated = lines[:-1]
                if any(s.strip() for s in truncated):
                    lines = truncated
        if seg.role == "body" and isinstance(lines, list):
            lines = [ln for ln in lines if ln.strip()]
        max_lines_cap = _sample_segment_line_cap(text_cfg, rng)
        if isinstance(lines, list) and lines:
            lines = lines[:max_lines_cap]
        if seg.role == "body" and not lines:
            # If body disappears after justify rules, drop the paired title as well.
            if allow_partial_layout and placements:
                return placements
            _skip_current_paragraph_block()
            continue
        if lines is None:
            # Current paragraph cannot be laid out; continue with next paragraph.
            if allow_partial_layout and placements:
                return placements
            _skip_current_paragraph_block()
            continue
        for line_idx, line in enumerate(lines):
            line_items = _build_segment_line_items(
                line=line,
                seg=seg,
                draw=draw,
                font=font,
            )
            if line_items is None:
                if allow_partial_layout and placements:
                    return placements
                _skip_current_paragraph_block()
                break
            if not line_items:
                if allow_partial_layout and placements:
                    return placements
                _skip_current_paragraph_block()
                break
            width = _measure_line_width_with_inline_gaps(line_items)
            line_height = max(item.bottom - item.top for item in line_items)
            if line and width <= 0:
                if allow_partial_layout and placements:
                    return placements
                _skip_current_paragraph_block()
                break
            if line_height <= 0:
                if allow_partial_layout and placements:
                    return placements
                _skip_current_paragraph_block()
                break
            if width > line_target_width:
                if allow_partial_layout and placements:
                    return placements
                _skip_current_paragraph_block()
                break
            if y + line_height > canvas.height - canvas.margin:
                if allow_partial_layout and placements:
                    return placements
                _skip_current_paragraph_block()
                break
            body_multi_line = seg.role == "body" and len(lines) >= 2
            justify_line = justify and seg.role == "body" and body_multi_line and len(line_items) > 1
            gap_slacks: Optional[List[int]] = None
            if justify_line:
                gap_slacks = _justify_inter_word_slacks(line_items, line_target_width)
                if gap_slacks is None:
                    justify_line = False
            # Title should follow normal left/center/right alignment, not justify.
            render_width = line_target_width if justify_line and width <= line_target_width else width
            start_x = _pick_line_start_x(h_align, canvas, render_width, line_target_width)
            x_cursor = start_x
            baseline_y = _baseline_y_for_line_items(line_items, y)
            for idx, item in enumerate(line_items):
                if idx > 0:
                    x_cursor += _emoji_text_inline_gap(line_items[idx - 1], item)
                y_draw, anch = _inline_item_y_and_anchor(item, y, line_height, baseline_y)
                placements.append(
                    PlacedText(
                        text=item.text,
                        x=x_cursor - item.left,
                        y=y_draw,
                        color=item.color,
                        font=item.font,
                        font_style=item.font_style,
                        role=item.role,
                        effects=item.effects,
                        anchor=anch,
                        paragraph_index=paragraph_idx,
                        corpus_type=item.corpus_type,
                        base_font_name=item.base_font_name,
                    )
                )
                x_cursor += item.width
                if justify_line and gap_slacks is not None and idx < len(line_items) - 1:
                    x_cursor += gap_slacks[idx]
            y += line_height + text_cfg.line_spacing
        if skip_body_until_next_title:
            continue
        y -= text_cfg.line_spacing
        if seg.role == "body":
            pending_title_checkpoint = None
        next_is_line_break = seg_idx < len(segments) - 1 and segments[seg_idx + 1].corpus_type == "line_break"
        if seg_idx < len(segments) - 1 and not next_is_line_break:
            next_role = segments[seg_idx + 1].role
            # Keep title_body body units continuous (full_text-like) without extra paragraph gaps.
            if not (seg.role == "body" and next_role == "body"):
                y += paragraph_spacing
        if y > canvas.height - canvas.margin:
            return _exit_mixed_line_placements(placements, allow_partial_layout)
    return placements


def _build_segment_line_items(
    line: str,
    seg: StyledSegment,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
) -> Optional[List[_LineItem]]:
    if not line:
        return []
    if seg.corpus_type == "english":
        # Split into words so justify can distribute spacing between words.
        # Keep one trailing normal space for all but last word to preserve baseline readability.
        words = [w for w in line.split() if w]
        if not words:
            if line != "" and all(c.isspace() for c in line):
                units = [" "]
            else:
                return []
        else:
            units = [f"{w} " if idx < len(words) - 1 else w for idx, w in enumerate(words)]
    else:
        units = list(line)
    items: List[_LineItem] = []
    for unit in units:
        bbox_anchor = None if seg.corpus_type == "emoji" else "ls"
        left, top, right, bottom = _measure_text_bbox(draw, unit, font, anchor=bbox_anchor)
        width = right - left
        advance = (
            _horizontal_advance_ls(draw, font, unit, left, right)
            if bbox_anchor == "ls"
            else max(width, int(round(draw.textlength(unit, font=font))))
        )
        height = bottom - top
        if unit.strip() and (width <= 0 or height <= 0):
            return None
        items.append(
            _LineItem(
                text=unit,
                x=0,
                width=advance,
                left=left,
                top=top,
                bottom=bottom,
                color=seg.color,
                font=font,
                font_style=seg.font_style,
                corpus_type=seg.corpus_type,
                base_font_name=seg.base_font_name,
                role=seg.role,
                effects=seg.effects,
            )
        )
    return items


def _layout_dual_column(
    segments: Sequence[StyledSegment],
    fonts: Dict[Tuple[str, int], ImageFont.FreeTypeFont],
    draw: ImageDraw.ImageDraw,
    canvas: CanvasConfig,
    text_cfg: TextConfig,
    variant: str,
    allow_partial_layout: bool = False,
    rng: Optional[random.Random] = None,
) -> Optional[List[PlacedText]]:
    if rng is None:
        rng = random.Random(0)
    blocks = _split_segments_for_dual_column(segments)
    if not blocks:
        return None
    h_align, v_align, justify = _parse_horizontal_variant(variant)
    # Respect configured variant strictly: no implicit random justify.
    use_justify = justify
    column_variant = f"{v_align}_{h_align}"
    if use_justify:
        column_variant = f"justify_{column_variant}"
    inner_margin = max(8, int(round(text_cfg.dual_column_inner_margin_at_1024 * canvas.height / 1024.0)))
    split_x = canvas.width // 2
    left_canvas = CanvasConfig(
        width=split_x,
        height=canvas.height - 2 * canvas.margin,
        min_width=None,
        max_width=None,
        min_height=None,
        max_height=None,
        short_edge_min=None,
        short_edge_max=None,
        aspect_ratios=None,
        margin=inner_margin,
        background_colors=[],
        min_text_bg_contrast_ratio=canvas.min_text_bg_contrast_ratio,
        background_images_dir=None,
        background_image_area_reference=canvas.background_image_area_reference,
    )
    right_canvas = CanvasConfig(
        width=canvas.width - split_x,
        height=canvas.height - 2 * canvas.margin,
        min_width=None,
        max_width=None,
        min_height=None,
        max_height=None,
        short_edge_min=None,
        short_edge_max=None,
        aspect_ratios=None,
        margin=inner_margin,
        background_colors=[],
        min_text_bg_contrast_ratio=canvas.min_text_bg_contrast_ratio,
        background_images_dir=None,
        background_image_area_reference=canvas.background_image_area_reference,
    )
    left: List[StyledSegment] = []
    right: List[StyledSegment] = []
    for idx, block in enumerate(blocks):
        target = left if idx % 2 == 0 else right
        target.extend(block)
        if block:
            anchor = block[0]
            target.append(
                StyledSegment(
                    text="\n",
                    corpus_type="line_break",
                    font_path=anchor.font_path,
                    font_name=anchor.font_name,
                    color=anchor.color,
                    font_size=anchor.font_size,
                    font_style="normal",
                    role="line_break",
                )
            )
    width_ratio_range = (
        text_cfg.dual_column_write_width_ratio_min,
        text_cfg.dual_column_write_width_ratio_max,
    )
    has_title_role = any(getattr(seg, "role", "") == "title" for seg in segments)
    if has_title_role:
        left_placements = _layout_segmented(
            left,
            fonts,
            draw,
            left_canvas,
            text_cfg,
            column_variant,
            paragraph_spacing_override=text_cfg.paragraph_spacing,
            width_ratio_range=width_ratio_range,
            allow_partial_layout=allow_partial_layout,
            rng=rng,
        )
        right_placements = _layout_segmented(
            right,
            fonts,
            draw,
            right_canvas,
            text_cfg,
            column_variant,
            paragraph_spacing_override=text_cfg.paragraph_spacing,
            width_ratio_range=width_ratio_range,
            allow_partial_layout=allow_partial_layout,
            rng=rng,
        )
    else:
        left_caps = _segment_line_caps_for(len(left), text_cfg, rng)
        right_caps = _segment_line_caps_for(len(right), text_cfg, rng)
        left_placements = _layout_mixed_line(
            left,
            fonts,
            draw,
            left_canvas,
            text_cfg,
            column_variant,
            width_ratio_range=width_ratio_range,
            allow_partial_layout=allow_partial_layout,
            segment_line_caps=left_caps,
        )
        right_placements = _layout_mixed_line(
            right,
            fonts,
            draw,
            right_canvas,
            text_cfg,
            column_variant,
            width_ratio_range=width_ratio_range,
            allow_partial_layout=allow_partial_layout,
            segment_line_caps=right_caps,
        )
    if allow_partial_layout:
        left_placements = left_placements if left_placements is not None else []
        right_placements = right_placements if right_placements is not None else []
        if not left_placements and not right_placements:
            return None
    elif left_placements is None or right_placements is None:
        return None
    x_left = 0
    x_right = split_x
    y_base = canvas.margin
    left_para_max = max((getattr(p, "paragraph_index", 0) for p in left_placements), default=-1)
    right_para_base = left_para_max + 1
    merged = [
        PlacedText(
            text=p.text,
            x=p.x + x_left,
            y=p.y + y_base - left_canvas.margin,
            color=p.color,
            font=p.font,
            font_style=p.font_style,
            role=p.role,
            effects=p.effects,
            anchor=getattr(p, "anchor", None),
            paragraph_index=getattr(p, "paragraph_index", 0),
            corpus_type=getattr(p, "corpus_type", ""),
            base_font_name=getattr(p, "base_font_name", ""),
        )
        for p in left_placements
    ] + [
        PlacedText(
            text=p.text,
            x=p.x + x_right,
            y=p.y + y_base - left_canvas.margin,
            color=p.color,
            font=p.font,
            font_style=p.font_style,
            role=p.role,
            effects=p.effects,
            anchor=getattr(p, "anchor", None),
            paragraph_index=right_para_base + getattr(p, "paragraph_index", 0),
            corpus_type=getattr(p, "corpus_type", ""),
            base_font_name=getattr(p, "base_font_name", ""),
        )
        for p in right_placements
    ]
    return merged


def _split_segments_for_dual_column(segments: Sequence[StyledSegment]) -> List[List[StyledSegment]]:
    rows: List[List[StyledSegment]] = []
    current: List[StyledSegment] = []
    has_title_role = any(getattr(seg, "role", "") == "title" for seg in segments)
    for seg_idx, seg in enumerate(segments):
        if seg.corpus_type == "line_break":
            if not current:
                continue
            if has_title_role:
                prev_role = current[-1].role if current else ""
                next_role = ""
                # find next non-line-break role
                # dual title-body paragraph boundary is body->title
                # title->body line_break stays inside a block
                # (fallback to normal split if roles missing)
                next_idx = seg_idx + 1
                while next_idx < len(segments):
                    nxt = segments[next_idx]
                    if nxt.corpus_type != "line_break":
                        next_role = nxt.role
                        break
                    next_idx += 1
                # Keep title->body line break inside the same block so
                # segmented layout can keep compact intra-paragraph spacing
                # and maintain title/body pairing checkpoint semantics.
                if prev_role == "title" and next_role == "body":
                    current.append(seg)
                    continue
                if prev_role == "body" and next_role == "title":
                    rows.append(current)
                    current = []
            else:
                rows.append(current)
                current = []
            continue
        current.append(seg)
    if current:
        rows.append(current)
    return rows


def _sample_width_by_ratio(
    seed_key: str,
    full_draw_width: int,
    min_ratio: float,
    max_ratio: float,
) -> int:
    min_ratio = max(0.1, min(1.0, float(min_ratio)))
    max_ratio = max(0.1, min(1.0, float(max_ratio)))
    if min_ratio > max_ratio:
        min_ratio, max_ratio = max_ratio, min_ratio
    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)
    ratio = min_ratio + (seed % 1000) / 999.0 * (max_ratio - min_ratio)
    return max(1, int(round(full_draw_width * ratio)))


def _pick_dual_column_justify_mode(canvas: CanvasConfig, segments: Sequence[StyledSegment]) -> bool:
    fingerprint = "".join(seg.text[:8] for seg in segments[:8])
    seed_key = f"dual|{canvas.width}x{canvas.height}|{len(segments)}|{fingerprint}"
    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)
    return seed % 2 == 0


def _layout_title_subtitle(
    segments: Sequence[StyledSegment],
    fonts: Dict[Tuple[str, int], ImageFont.FreeTypeFont],
    draw: ImageDraw.ImageDraw,
    canvas: CanvasConfig,
    text_cfg: TextConfig,
    variant: str,
) -> Optional[List[PlacedText]]:
    _ = variant
    title_segments = [seg for seg in segments if seg.role == "title" and seg.corpus_type != "line_break"]
    subtitle_segments = [seg for seg in segments if seg.role == "subtitle" and seg.corpus_type != "line_break"]
    if not title_segments or not subtitle_segments:
        return None
    title_text = "".join(seg.text for seg in title_segments)
    subtitle_text = "".join(seg.text for seg in subtitle_segments)
    if not title_text.strip() or not subtitle_text.strip():
        return None

    title_anchor = next((seg for seg in title_segments if seg.corpus_type != "emoji"), title_segments[0])
    subtitle_anchor = next(
        (seg for seg in subtitle_segments if seg.corpus_type != "emoji"), subtitle_segments[0]
    )
    title_font = fonts[(title_anchor.font_path, title_anchor.font_size)]
    subtitle_font = fonts[(subtitle_anchor.font_path, subtitle_anchor.font_size)]
    full_width = canvas.width - 2 * canvas.margin
    title_lines = _wrap_text_to_lines(title_text, title_font, draw, full_width, preserve_word=False)
    subtitle_line = _truncate_words_to_single_line(subtitle_text, subtitle_font, draw, full_width)
    if not subtitle_line:
        return None
    subtitle_lines = [subtitle_line]
    if not title_lines:
        return None

    def measure_block(lines: Sequence[str], font: ImageFont.FreeTypeFont, line_spacing: int) -> Tuple[int, int]:
        widths: List[int] = []
        heights: List[int] = []
        for line in lines:
            left, top, right, bottom = _measure_text_bbox(draw, line, font)
            widths.append(max(1, right - left))
            heights.append(max(1, bottom - top))
        total_h = sum(heights) + line_spacing * max(0, len(lines) - 1)
        return max(widths), total_h

    title_w, title_h = measure_block(title_lines, title_font, text_cfg.line_spacing)
    subtitle_w, subtitle_h = measure_block(subtitle_lines, subtitle_font, text_cfg.line_spacing)
    if max(title_w, subtitle_w) > full_width:
        return None

    gap = max(12, int(title_anchor.font_size * 0.18))
    total_h = title_h + gap + subtitle_h
    full_height = canvas.height - 2 * canvas.margin
    if total_h > full_height:
        return None
    y = _pick_random_vertical_start(
        canvas=canvas,
        total_h=total_h,
        title_text=title_text,
        subtitle_text=subtitle_text,
    )
    placements: List[PlacedText] = []

    for line in title_lines:
        left, top, right, bottom = _measure_text_bbox(draw, line, title_font)
        width = right - left
        line_h = bottom - top
        x = canvas.margin + (full_width - width) // 2 - left
        y_draw = y - top
        placements.append(
            PlacedText(
                text=line,
                x=x,
                y=y_draw,
                color=title_anchor.color,
                font=title_font,
                font_style=title_anchor.font_style,
                role=title_anchor.role,
                effects=title_anchor.effects,
                paragraph_index=0,
                corpus_type=title_anchor.corpus_type,
                base_font_name=title_anchor.base_font_name,
            )
        )
        y += line_h + text_cfg.line_spacing
    y += gap - text_cfg.line_spacing

    for line in subtitle_lines:
        left, top, right, bottom = _measure_text_bbox(draw, line, subtitle_font)
        width = right - left
        line_h = bottom - top
        x = canvas.margin + (full_width - width) // 2 - left
        y_draw = y - top
        placements.append(
            PlacedText(
                text=line,
                x=x,
                y=y_draw,
                color=subtitle_anchor.color,
                font=subtitle_font,
                font_style=subtitle_anchor.font_style,
                role=subtitle_anchor.role,
                effects=subtitle_anchor.effects,
                paragraph_index=1,
                corpus_type=subtitle_anchor.corpus_type,
                base_font_name=subtitle_anchor.base_font_name,
            )
        )
        y += line_h + text_cfg.line_spacing
    return placements


def _pick_random_vertical_start(
    canvas: CanvasConfig,
    total_h: int,
    title_text: str,
    subtitle_text: str,
) -> int:
    min_y = canvas.margin
    max_y = canvas.height - canvas.margin - total_h
    if max_y <= min_y:
        return min_y
    payload = f"{title_text}||{subtitle_text}||{canvas.width}x{canvas.height}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16) % (max_y - min_y + 1)
    return min_y + offset


def _truncate_words_to_single_line(
    text: str,
    font: ImageFont.FreeTypeFont,
    draw: ImageDraw.ImageDraw,
    max_width: int,
) -> str:
    # For subtitle: keep a single line and drop overflowing tail words.
    words = [w for w in text.split(" ") if w]
    if not words:
        return text.strip()
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        w, _ = _measure_text(draw, candidate, font)
        if w <= max_width:
            current = candidate
            continue
        break
    return current.strip()


def _layout_vertical(
    segments: Sequence[StyledSegment],
    fonts: Dict[Tuple[str, int], ImageFont.FreeTypeFont],
    draw: ImageDraw.ImageDraw,
    canvas: CanvasConfig,
    text_cfg: TextConfig,
    direction: str,
) -> Optional[List[PlacedText]]:
    h_anchor, direction = _parse_vertical_variant(direction)
    if direction not in {"rtl", "ltr"}:
        raise ValueError(f"Unsupported vertical direction: {direction}")
    placements: List[PlacedText] = []
    paragraph_idx = 0
    y = canvas.margin
    if h_anchor == "left":
        x_cursor = canvas.margin if direction == "ltr" else canvas.width - canvas.margin
    elif h_anchor == "center":
        x_cursor = canvas.width // 2
    else:
        x_cursor = canvas.width - canvas.margin if direction == "rtl" else canvas.margin
    current_col: List[_VerticalItem] = []

    def flush_current_column() -> bool:
        nonlocal y, x_cursor
        if not current_col:
            return True
        col_width = max(item.left + item.width for item in current_col)
        if col_width <= 0:
            return False
        if direction == "rtl":
            x_left = x_cursor - col_width
            if x_left < canvas.margin:
                return False
            next_x_cursor = x_left - text_cfg.paragraph_spacing
        else:
            x_left = x_cursor
            x_right = x_left + col_width
            if x_right > canvas.width - canvas.margin:
                return False
            next_x_cursor = x_right + text_cfg.paragraph_spacing
        pidx = paragraph_idx
        for item in current_col:
            # Align each glyph box horizontally to the center of current column.
            x_draw = x_left + (col_width - (item.left + item.width)) // 2 - item.left
            y_draw = item.y - item.top
            placements.append(
                PlacedText(
                    text=item.text,
                    x=x_draw,
                    y=y_draw,
                    color=item.color,
                    font=item.font,
                    font_style=item.font_style,
                    role=getattr(item, "role", "body"),
                    effects=getattr(item, "effects", None),
                    paragraph_index=pidx,
                    corpus_type=getattr(item, "corpus_type", ""),
                    base_font_name=getattr(item, "base_font_name", ""),
                )
            )
        x_cursor = next_x_cursor
        y = canvas.margin
        current_col.clear()
        return True

    for seg in segments:
        font = fonts[(seg.font_path, seg.font_size)]
        for ch in seg.text:
            if ch == "\n":
                if not flush_current_column():
                    return placements if placements else None
                paragraph_idx += 1
                continue

            left, top, right, bottom = _measure_text_bbox(draw, ch, font)
            width = right - left
            height = bottom - top
            if width <= 0 or height <= 0:
                if ch.isspace():
                    # Keep vertical rhythm for whitespace that may render empty bbox.
                    height = max(1, int(seg.font_size * 0.5))
                    left, top = 0, 0
                else:
                    return None

            remain = canvas.height - canvas.margin - y
            if height > remain and y > canvas.margin:
                if ch in LEADING_PUNCTUATION and current_col:
                    carry = current_col.pop()
                    if not flush_current_column():
                        return placements if placements else None
                    current_col.append(
                        _VerticalItem(
                            text=carry.text,
                            y=canvas.margin,
                            height=carry.height,
                            width=carry.width,
                            left=carry.left,
                            top=carry.top,
                            color=carry.color,
                            font=carry.font,
                            font_style=carry.font_style,
                            role=carry.role,
                            effects=carry.effects,
                            corpus_type=carry.corpus_type,
                        )
                    )
                    y = canvas.margin + carry.height + text_cfg.line_spacing
                if not flush_current_column():
                    return placements if placements else None

            if height > canvas.height - 2 * canvas.margin:
                return placements if placements else None

            current_col.append(
                _VerticalItem(
                    text=ch,
                    y=y,
                    height=height,
                    width=width,
                    left=left,
                    top=top,
                    color=seg.color,
                    font=font,
                    font_style=seg.font_style,
                    role=seg.role,
                    effects=seg.effects,
                    corpus_type=seg.corpus_type,
                )
            )
            y += height + text_cfg.line_spacing

    if not flush_current_column():
        return placements if placements else None
    return placements


def _apply_block_vertical_anchor(
    placements: List[PlacedText],
    draw: ImageDraw.ImageDraw,
    canvas: CanvasConfig,
    variant: str,
    allow_partial_layout: bool = False,
) -> Optional[List[PlacedText]]:
    if not placements:
        return placements
    _, v_anchor, _ = _parse_horizontal_variant(variant)
    if v_anchor == "top":
        return placements

    bbox = _measure_placements_bbox(placements, draw)
    if bbox is None:
        return placements
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    avail_width = canvas.width - 2 * canvas.margin
    avail_height = canvas.height - 2 * canvas.margin
    if width > avail_width or height > avail_height:
        # Partial layouts may exceed the drawable band before all segments are placed;
        # do not invalidate the whole layout (would drop multi-block candidates in resilient build).
        if allow_partial_layout:
            return placements
        return None

    dx, dy = 0, 0
    if v_anchor == "middle":
        target_top = canvas.margin + (avail_height - height) // 2
        dy = target_top - top
    elif v_anchor == "bottom":
        target_top = canvas.height - canvas.margin - height
        dy = target_top - top

    shifted = [
        PlacedText(
            text=item.text,
            x=item.x + dx,
            y=item.y + dy,
            color=item.color,
            font=item.font,
            font_style=item.font_style,
            role=item.role,
            effects=item.effects,
            anchor=getattr(item, "anchor", None),
            paragraph_index=getattr(item, "paragraph_index", 0),
            corpus_type=getattr(item, "corpus_type", ""),
            base_font_name=getattr(item, "base_font_name", ""),
        )
        for item in placements
    ]
    shifted_bbox = _measure_placements_bbox(shifted, draw)
    if shifted_bbox is None:
        return shifted
    s_left, s_top, s_right, s_bottom = shifted_bbox
    if (
        s_left < canvas.margin
        or s_top < canvas.margin
        or s_right > canvas.width - canvas.margin
        or s_bottom > canvas.height - canvas.margin
    ):
        if allow_partial_layout:
            return placements
        return None
    return shifted


def _measure_placements_bbox(
    placements: Sequence[PlacedText], draw: ImageDraw.ImageDraw
) -> Optional[Tuple[int, int, int, int]]:
    if not placements:
        return None
    lefts: List[int] = []
    tops: List[int] = []
    rights: List[int] = []
    bottoms: List[int] = []
    for item in placements:
        kw = {"font": item.font, "embedded_color": True}
        a = getattr(item, "anchor", None)
        if a:
            kw["anchor"] = a
        l, t, r, b = draw.textbbox((item.x, item.y), item.text, **kw)
        lefts.append(l)
        tops.append(t)
        rights.append(r)
        bottoms.append(b)
    return min(lefts), min(tops), max(rights), max(bottoms)


def _pick_line_start_x(h_align: str, canvas: CanvasConfig, line_width: int, target_width: int) -> int:
    full_width = canvas.width - 2 * canvas.margin
    if target_width > full_width:
        raise ValueError("target_width exceeds drawable width")
    if line_width > target_width:
        raise ValueError("line_width exceeds drawable width")
    if h_align == "left":
        base_left = canvas.margin
    elif h_align == "center":
        base_left = canvas.margin + (full_width - target_width) // 2
    elif h_align == "right":
        base_left = canvas.margin + (full_width - target_width)
    else:
        raise ValueError(f"Unsupported horizontal align: {h_align}")
    if h_align == "left":
        return base_left
    if h_align == "center":
        return base_left + (target_width - line_width) // 2
    if h_align == "right":
        return base_left + target_width - line_width
    raise ValueError(f"Unsupported horizontal align: {h_align}")


def _sample_justify_target_width(seed_key: str, full_draw_width: int) -> int:
    ratio_seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)
    ratio = 0.4 + (ratio_seed % 41) / 100.0
    return max(1, int(round(full_draw_width * ratio)))


def _pick_justify_width_mode(variant: str, canvas: CanvasConfig, segments: Sequence[StyledSegment]) -> str:
    text_fingerprint = "".join(seg.text[:8] for seg in segments[:8])
    seed_key = f"{variant}|{canvas.width}|{canvas.height}|{len(segments)}|{text_fingerprint}"
    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)
    return "uniform" if seed % 2 == 0 else "per_paragraph"


def _is_emoji_text(text: str, corpus_type: str) -> bool:
    if corpus_type == "emoji":
        return True
    if len(text) != 1:
        return False
    cp = ord(text)
    return (
        0x1F300 <= cp <= 0x1FAFF
        or 0x2600 <= cp <= 0x26FF
        or 0x2700 <= cp <= 0x27BF
    )


def _emoji_text_inline_gap(
    prev_item: _LineItem,
    curr_item: Optional[_LineItem],
    next_text: str = "",
    next_corpus_type: str = "",
    next_font_style: str = "",
) -> int:
    prev_is_emoji = _is_emoji_text(prev_item.text, prev_item.corpus_type)
    if curr_item is not None:
        curr_is_emoji = _is_emoji_text(curr_item.text, curr_item.corpus_type)
        size = min(prev_item.font.size, curr_item.font.size)
        italic_involved = ("italic" in prev_item.font_style) or ("italic" in curr_item.font_style)
    else:
        curr_is_emoji = _is_emoji_text(next_text, next_corpus_type)
        size = prev_item.font.size
        italic_involved = ("italic" in prev_item.font_style) or ("italic" in next_font_style)
    if prev_is_emoji == curr_is_emoji:
        return 0
    # Keep a small visual buffer only for emoji<->text transitions.
    if italic_involved:
        return max(4, int(round(size * 0.14)))
    return max(2, int(round(size * 0.06)))


def _measure_line_width_with_inline_gaps(items: Sequence[_LineItem]) -> int:
    width = 0
    for idx, item in enumerate(items):
        if idx > 0:
            width += _emoji_text_inline_gap(items[idx - 1], item)
        width += item.width
    return width


def _english_word_count_for_justify(items: Sequence[_LineItem]) -> int:
    """Whitespace-separated tokens from english corpus items only (emoji segments excluded)."""
    parts: List[str] = []
    for it in items:
        if it.corpus_type == "english":
            parts.append(it.text)
    return len([w for w in "".join(parts).split() if w])


def _is_justify_word_unit(item: _LineItem) -> bool:
    """English glyph unit that is not a standalone emoji (emoji may use corpus_type english when split from a line)."""
    if item.corpus_type != "english":
        return False
    core = item.text.strip()
    if not core:
        return False
    return not _is_emoji_text(core, item.corpus_type)


def _justify_inter_word_slacks(
    items: Sequence[_LineItem], target_width: int
) -> Optional[List[int]]:
    """
    Pixel slack to insert after each item (before the next). None means do not justify to target
    (fall back to h_align). Slacks are only placed between adjacent non-emoji english word units
    so emoji boundaries keep a fixed _emoji_text_inline_gap and do not absorb justify stretch.
    """
    n = len(items)
    if n < 2:
        return None
    if _english_word_count_for_justify(items) < 2:
        return None
    line_width = _measure_line_width_with_inline_gaps(items)
    extra = max(0, target_width - line_width)
    eligible = [
        i
        for i in range(n - 1)
        if _is_justify_word_unit(items[i]) and _is_justify_word_unit(items[i + 1])
    ]
    # Prefer stretching only between non-emoji english words; if there is no such gap
    # (e.g. emoji between every word) fall back to all inter-item gaps so justify still
    # reaches target width.
    gap_indices = eligible if eligible else list(range(n - 1))
    if not gap_indices:
        if extra > 0:
            return None
        return [0] * (n - 1)
    base = extra // len(gap_indices)
    rem = extra % len(gap_indices)
    out = [0] * (n - 1)
    for j, i in enumerate(gap_indices):
        out[i] = base + (1 if j < rem else 0)
    return out


def _is_leading_punctuation_unit(text: str) -> bool:
    return len(text) == 1 and text in LEADING_PUNCTUATION


def _parse_horizontal_variant(variant: str) -> Tuple[str, str, bool]:
    # supports: top_left/top_center/top_right/middle_left/.../bottom_right and justify_*
    raw = (variant or "top_left").strip()
    justify = raw.startswith("justify_")
    core = raw[len("justify_") :] if justify else raw
    if core in {"left_aligned", "top_left"}:
        return "left", "top", justify
    if core in {"top_centered", "top_center"}:
        return "center", "top", justify
    if core in {"right_aligned", "top_right"}:
        return "right", "top", justify
    if core in {"center_vertical", "middle_center"}:
        return "center", "middle", justify
    if core == "middle_left":
        return "left", "middle", justify
    if core == "middle_right":
        return "right", "middle", justify
    if core == "bottom_left":
        return "left", "bottom", justify
    if core == "bottom_center":
        return "center", "bottom", justify
    if core == "bottom_right":
        return "right", "bottom", justify
    return "left", "top", justify


def _parse_vertical_variant(variant: str) -> Tuple[str, str]:
    raw = (variant or "top_left@rtl").strip()
    if "@" in raw:
        anchor, direction = raw.split("@", 1)
    else:
        anchor, direction = raw, "rtl"
    anchor = anchor.strip()
    direction = direction.strip() or "rtl"
    if anchor == "top_left":
        return "left", direction
    if anchor == "top_center":
        return "center", direction
    if anchor == "top_right":
        return "right", direction
    if anchor in {"rtl", "ltr"}:
        return "left", anchor
    return "left", direction


def _split_text_for_line_wrapping(text: str) -> List[str]:
    tokens: List[str] = []
    for part in text.split("\n"):
        if part:
            tokens.append(part)
        tokens.append("\n")
    if tokens and tokens[-1] == "\n":
        tokens.pop()
    return tokens


def _strip_trailing_title_punctuation(text: str) -> str:
    out = text.rstrip()
    while out and out[-1] in TITLE_END_PUNCTUATION:
        out = out[:-1].rstrip()
    return out


def _truncate_text_to_single_line(
    text: str,
    font: ImageFont.FreeTypeFont,
    draw: ImageDraw.ImageDraw,
    max_width: int,
    preserve_word: bool = False,
) -> str:
    text = _strip_trailing_title_punctuation(text)
    if not text:
        return ""
    width, _ = _measure_text(draw, text, font)
    if width <= max_width:
        return text
    if preserve_word:
        words = [w for w in text.split(" ") if w]
        if not words:
            return ""
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            w, _ = _measure_text(draw, candidate, font)
            if w <= max_width:
                current = candidate
            else:
                break
        return _strip_trailing_title_punctuation(current)
    # CJK/char mode: keep longest prefix that fits one line.
    low, high = 1, len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid]
        w, _ = _measure_text(draw, candidate, font)
        if w <= max_width:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return _strip_trailing_title_punctuation(best)


def _wrap_text_to_lines(
    text: str,
    font: ImageFont.FreeTypeFont,
    draw: ImageDraw.ImageDraw,
    max_width: int,
    preserve_word: bool = False,
) -> Optional[List[str]]:
    lines: List[str] = []
    for raw_line in text.split("\n"):
        if not raw_line:
            lines.append(" ")
            continue
        if preserve_word:
            wrapped = _wrap_text_to_lines_by_words(raw_line, font, draw, max_width)
            if wrapped is None:
                return None
            lines.extend(wrapped)
            continue
        start = 0
        while start < len(raw_line):
            low = start + 1
            high = len(raw_line)
            best_end = None
            while low <= high:
                mid = (low + high) // 2
                candidate = raw_line[start:mid]
                w, _ = _measure_text(draw, candidate, font)
                if w <= max_width:
                    best_end = mid
                    low = mid + 1
                else:
                    high = mid - 1
            if best_end is None:
                return None
            lines.append(raw_line[start:best_end])
            start = best_end
    if not preserve_word:
        lines = _fix_leading_punctuation(lines)
    return lines


def _fix_leading_punctuation(lines: List[str]) -> List[str]:
    if not lines:
        return lines
    fixed: List[str] = []
    for line in lines:
        cur = line
        while cur and cur[0] in LEADING_PUNCTUATION and fixed and fixed[-1]:
            fixed[-1] = fixed[-1] + cur[0]
            cur = cur[1:]
        if cur:
            fixed.append(cur)
    return fixed


def _wrap_text_to_lines_by_words(
    raw_line: str,
    font: ImageFont.FreeTypeFont,
    draw: ImageDraw.ImageDraw,
    max_width: int,
) -> Optional[List[str]]:
    words = [w for w in raw_line.split(" ") if w]
    if " " not in raw_line:
        # Treat single token as one indivisible word.
        width, _ = _measure_text(draw, raw_line, font)
        if width > max_width:
            return None
        return [raw_line]
    if not words:
        return [" "]
    lines: List[str] = []
    current_line = ""
    for word in words:
        # Do not split a single word across multiple lines.
        word_w, _ = _measure_text(draw, word, font)
        if word_w > max_width:
            # Skip impossible tokens instead of failing the whole paragraph.
            continue
        candidate = word if not current_line else f"{current_line} {word}"
        candidate_w, _ = _measure_text(draw, candidate, font)
        if candidate_w <= max_width:
            current_line = candidate
            continue
        if current_line:
            lines.append(current_line)
        current_line = word
    if current_line:
        lines.append(current_line)
    if not lines:
        return None
    return lines


def _wrap_token_to_width(
    token: str,
    font: ImageFont.FreeTypeFont,
    draw: ImageDraw.ImageDraw,
    max_width: int,
    preserve_word: bool = False,
) -> Optional[List[str]]:
    chunks = _wrap_text_to_lines(token, font, draw, max_width, preserve_word=preserve_word)
    return chunks


def _measure_text(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont
) -> Tuple[int, int]:
    left, top, right, bottom = _measure_text_bbox(draw, text, font)
    return right - left, bottom - top


def _measure_text_bbox(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    anchor: Optional[str] = None,
) -> Tuple[int, int, int, int]:
    kw = {"font": font, "embedded_color": True}
    if anchor:
        kw["anchor"] = anchor
    return draw.textbbox((0, 0), text, **kw)


def _horizontal_advance_ls(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    unit: str,
    bbox_left: int,
    bbox_right: int,
) -> int:
    """
    Horizontal advance after drawing `unit` with anchor ls. Ink bbox often ignores trailing
    spaces; using only max(ink_width, textlength(unit)) can still under-advance so the next
    word is drawn too close. Force word + explicit space width when the token ends with U+0020.
    """
    ink_w = max(0, bbox_right - bbox_left)
    tl = float(draw.textlength(unit, font=font))
    out = max(tl, float(ink_w))
    if unit.endswith(" "):
        base = unit[:-1]
        sp = float(draw.textlength(" ", font=font))
        base_tl = float(draw.textlength(base, font=font)) if base else 0.0
        out = max(out, base_tl + sp)
    rounded = int(round(out))
    return max(1, rounded) if unit else 0


def _baseline_y_for_line_items(items: Sequence[_LineItem], line_y: int) -> int:
    """Shared baseline y for alphabetic text; emoji metrics are excluded so emoji don't skew the line."""
    non_emoji = [it for it in items if it.corpus_type != "emoji"]
    src = non_emoji if non_emoji else list(items)
    return line_y - min(it.top for it in src)


def _inline_item_y_and_anchor(
    item: _LineItem, line_y: int, line_height: int, baseline_y: int
) -> Tuple[int, Optional[str]]:
    # Color emoji fonts don't sit on the Latin baseline with anchor "ls"; center the glyph in the line box.
    if item.corpus_type == "emoji":
        box_h = item.bottom - item.top
        return line_y + (line_height - box_h) // 2 - item.top, None
    return baseline_y, "ls"
