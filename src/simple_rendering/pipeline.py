from __future__ import annotations

import json
import os
import math
import random
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import median
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from tqdm import tqdm

from .config import RenderConfig, load_config, resolve_config_path
from .corpus import (
    CorpusItem,
    build_corpus_pools,
    build_inline_emoji_segments_from_pools,
    build_title_subtitle_segments_from_pools,
    load_corpus_items,
    load_title_corpus_items,
    normalize_english_corpus_unit,
)
from .font_manager import FontCoverageManager, build_styled_segments
from .layout import layout_segments
from .renderer import render_image

EMOJI_CANDIDATES = ["😀", "😄", "😎", "🚀", "✨", "🎯", "✅", "🔥", "🌟", "🎉"]
TITLE_END_PUNCTUATION = set("，。！？；：、,.!?;:)]}）】》」』”’\"'…—-")
WORKER_STATE: Dict = {}
WORKER_FONT_MANAGER: Optional[FontCoverageManager] = None
WORKER_CAPTION_TEMPLATES_L1: Optional[Dict[str, List[str]]] = None
WORKER_CAPTION_TEMPLATES_L2: Optional[Dict[str, Dict[str, List[str]]]] = None
WORKER_CAPTION_TEMPLATES_L3: Optional[Dict[str, Dict[str, List[str]]]] = None
WORKER_CAPTION_TEMPLATES_L4: Optional[Dict[str, Dict[str, List[str]]]] = None
WORKER_LAYOUT_MODE_DESC_L3: Optional[Dict[str, Dict[str, List[str]]]] = None
WORKER_ALIGNMENT_PHRASES: Optional[Dict[str, Dict[str, List[str]]]] = None
WORKER_FONT_MAP: Optional[Dict[str, Dict[str, Any]]] = None
WORKER_COLOR_MAP: Optional[Dict[str, Dict[str, List[str]]]] = None


def _load_caption_templates_L1(templates_path: Path) -> Dict[str, List[str]]:
    """
    Load caption templates JSON file.

    Expected format:
      {
        "zh": ["...{text_all}...", ...],
        "en": ["...{text_all}...", ...]
      }
    """
    try:
        with templates_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"caption templates file not found: {templates_path}") from e
    if not isinstance(data, dict):
        raise ValueError(f"caption templates must be a JSON object: {templates_path}")
    for lang in ("zh", "en"):
        if lang not in data or not isinstance(data[lang], list) or not all(
            isinstance(x, str) for x in data[lang]
        ):
            raise ValueError(f"caption templates[{lang}] must be a list of strings: {templates_path}")
    return {"zh": data["zh"], "en": data["en"]}


def _apply_caption_template(template: str, text_all: str) -> str:
    # Support both "{text_all}" and a raw "text_all" placeholder.
    return template.replace("{text_all}", text_all).replace("text_all", text_all)


def _load_caption_templates_title_body(templates_path: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Load title/body caption templates (used for parquet L2).

    Expected format:
      {
        "zh": {"title": [...], "body": [...]},
        "en": {"title": [...], "body": [...]}
      }
    """
    try:
        with templates_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"caption title/body templates file not found: {templates_path}") from e
    if not isinstance(data, dict):
        raise ValueError(f"caption title/body templates must be a JSON object: {templates_path}")
    out: Dict[str, Dict[str, List[str]]] = {}
    for lang in ("zh", "en"):
        lang_obj = data.get(lang)
        if not isinstance(lang_obj, dict):
            raise ValueError(
                f"caption title/body templates[{lang}] must be an object with title/body lists: {templates_path}"
            )
        for role in ("title", "body"):
            role_list = lang_obj.get(role)
            if not isinstance(role_list, list) or not all(isinstance(x, str) for x in role_list):
                raise ValueError(
                    f"caption title/body templates[{lang}][{role}] must be a list of strings: {templates_path}"
                )
        out[lang] = {"title": lang_obj["title"], "body": lang_obj["body"]}
    return out


def _apply_caption_template_title_body(
    template: str, font_name: str, text_color: str, text_paragraph: str
) -> str:
    # Font/color should not be wrapped by quotes in final captions, even if templates add them.
    t = template
    for ph in ("{font_name}", "{text_color}"):
        t = (
            t.replace(f"“{ph}”", ph)
            .replace(f"\"{ph}\"", ph)
            .replace(f"'{ph}'", ph)
            .replace(f"‘{ph}’", ph)
        )
    return t.replace("{font_name}", font_name).replace("{text_color}", text_color).replace(
        "{text_paragraph}", text_paragraph
    )


def _load_caption_templates_L3_scene(templates_path: Path) -> Dict[str, Dict[str, List[str]]]:
    """Scene + per-paragraph descriptions (L3); same shape as L4 but uses paragraph_desc instead of line_desc."""
    try:
        with templates_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"caption L3 scene templates file not found: {templates_path}") from e
    if not isinstance(data, dict):
        raise ValueError(f"caption L3 scene templates must be a JSON object: {templates_path}")
    required_keys = [
        "scene_bg",
        "scene_bg_image",
        "scene_text_fragment",
        "scene_overall_align",
        "title_intro",
        "title_style",
        "paragraph_style",
        "paragraph_index_intro",
        "paragraph_bridge",
        "paragraph_desc",
        "effect_bold",
        "effect_shadow",
    ]
    out: Dict[str, Dict[str, List[str]]] = {}
    for lang in ("zh", "en"):
        obj = data.get(lang)
        if not isinstance(obj, dict):
            raise ValueError(f"caption L3 scene templates[{lang}] must be an object: {templates_path}")
        for k in required_keys:
            v = obj.get(k)
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise ValueError(f"caption L3 scene templates[{lang}][{k}] must be list[str]: {templates_path}")
        out[lang] = {k: obj[k] for k in required_keys}
    return out


def _load_caption_templates_L4(templates_path: Path) -> Dict[str, Dict[str, List[str]]]:
    try:
        with templates_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"caption L4 templates file not found: {templates_path}") from e
    if not isinstance(data, dict):
        raise ValueError(f"caption L4 templates must be a JSON object: {templates_path}")
    required_keys = [
        "scene_bg",
        "scene_overall_align",
        "title_intro",
        "title_style",
        "body_segment_intro",
        "body_segment_style",
        "line_desc",
        "effect_bold",
        "effect_shadow",
    ]
    out: Dict[str, Dict[str, List[str]]] = {}
    for lang in ("zh", "en"):
        obj = data.get(lang)
        if not isinstance(obj, dict):
            raise ValueError(f"caption L4 templates[{lang}] must be an object: {templates_path}")
        for k in required_keys:
            v = obj.get(k)
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise ValueError(f"caption L4 templates[{lang}][{k}] must be list[str]: {templates_path}")
        out[lang] = {k: obj[k] for k in required_keys}
    return out


def _load_layout_mode_desc_L3(path: Path) -> Dict[str, Dict[str, List[str]]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"layout mode L3 description file not found: {path}") from e
    if not isinstance(data, dict):
        raise ValueError(f"layout mode L3 description must be a JSON object: {path}")
    out: Dict[str, Dict[str, List[str]]] = {}
    for lang in ("zh", "en"):
        obj = data.get(lang)
        if not isinstance(obj, dict):
            raise ValueError(f"layout mode L3 descriptions[{lang}] must be object: {path}")
        for mode in ("full_text", "title_body"):
            ls = obj.get(mode)
            if not isinstance(ls, list) or not all(isinstance(x, str) for x in ls):
                raise ValueError(f"layout mode L3 descriptions[{lang}][{mode}] must be list[str]: {path}")
        out[lang] = {"full_text": obj["full_text"], "title_body": obj["title_body"]}
    return out


def _load_font_map(font_map_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load font_map.json once at startup.

    Expected value schema for each font key:
      {
        "zh": "...",
        "en": "...",
        "style_zh": ["...", ...],
        "style_en": ["...", ...]
      }
    """
    try:
        with font_map_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"font map file not found: {font_map_path}") from e
    if not isinstance(data, dict):
        raise ValueError(f"font map must be a JSON object: {font_map_path}")
    for font_key, meta in data.items():
        if not isinstance(meta, dict):
            raise ValueError(f"font map[{font_key}] must be an object: {font_map_path}")
        if not isinstance(meta.get("zh"), str) or not isinstance(meta.get("en"), str):
            raise ValueError(f"font map[{font_key}] requires string fields 'zh' and 'en': {font_map_path}")
        style_zh = meta.get("style_zh", [])
        style_en = meta.get("style_en", [])
        if not isinstance(style_zh, list) or not all(isinstance(x, str) for x in style_zh):
            raise ValueError(f"font map[{font_key}].style_zh must be list[str]: {font_map_path}")
        if not isinstance(style_en, list) or not all(isinstance(x, str) for x in style_en):
            raise ValueError(f"font map[{font_key}].style_en must be list[str]: {font_map_path}")
    return data


def _pick_font_caption_labels(
    font_name: str, rng: random.Random, font_map: Optional[Dict[str, Dict[str, Any]]]
) -> Tuple[str, str]:
    if not font_map:
        return font_name, font_name
    meta = font_map.get(font_name)
    if not isinstance(meta, dict):
        return font_name, font_name
    zh_base = str(meta.get("zh", font_name))
    en_base = str(meta.get("en", font_name))
    zh_candidates = [zh_base] + [str(x) for x in meta.get("style_zh", []) if str(x)]
    en_candidates = [en_base] + [str(x) for x in meta.get("style_en", []) if str(x)]
    return rng.choice(zh_candidates), rng.choice(en_candidates)


def _load_color_map(color_map_path: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Load color_map.json once at startup.

    Expected schema:
      {
        "#RRGGBB": {
          "zh": ["颜色中文名", ...],
          "en": ["English color name", ...]
        },
        ...
      }
    """
    try:
        with color_map_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"color map file not found: {color_map_path}") from e
    if not isinstance(data, dict):
        raise ValueError(f"color map must be a JSON object: {color_map_path}")
    out: Dict[str, Dict[str, List[str]]] = {}
    for hex_color, names in data.items():
        if not isinstance(hex_color, str):
            raise ValueError(f"color map key must be color hex string: {color_map_path}")
        if not isinstance(names, dict):
            raise ValueError(f"color map[{hex_color}] must be an object: {color_map_path}")
        zh_list = names.get("zh")
        en_list = names.get("en")
        if not isinstance(zh_list, list) or not all(isinstance(x, str) for x in zh_list):
            raise ValueError(f"color map[{hex_color}].zh must be list[str]: {color_map_path}")
        if not isinstance(en_list, list) or not all(isinstance(x, str) for x in en_list):
            raise ValueError(f"color map[{hex_color}].en must be list[str]: {color_map_path}")
        out[hex_color.upper()] = {"zh": zh_list, "en": en_list}
    return out


def _pick_color_caption_labels(
    text_color: str, rng: random.Random, color_map: Optional[Dict[str, Dict[str, List[str]]]]
) -> Tuple[str, str]:
    if not color_map:
        return text_color, text_color
    key = str(text_color).upper()
    names = color_map.get(key)
    if not isinstance(names, dict):
        return text_color, text_color
    zh_candidates = [str(x) for x in names.get("zh", []) if str(x)]
    en_candidates = [str(x) for x in names.get("en", []) if str(x)]
    zh = rng.choice(zh_candidates) if zh_candidates else text_color
    en = rng.choice(en_candidates) if en_candidates else text_color
    return zh, en


def _group_content_by_paragraph(content_list: Sequence[Dict[str, object]]) -> List[Dict[str, str]]:
    """
    Merge rendered line-level content rows into paragraph-level units.
    Split groups when paragraph_index or role changes, so title/body stay separated.
    """
    grouped: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None
    for item in content_list:
        para_idx = int(item.get("paragraph_index", 0) or 0)
        role = str(item.get("role", "body"))
        text = str(item.get("text", ""))
        if (
            current is None
            or int(current["paragraph_index"]) != para_idx
            or str(current["role"]) != role
        ):
            current = {
                "paragraph_index": para_idx,
                "role": role,
                "texts": [],
                "font": str(item.get("font", "")),
                "text_color": str(item.get("text_color", "")),
            }
            grouped.append(current)
        current["texts"].append(text)
    out: List[Dict[str, str]] = []
    for g in grouped:
        lines = [str(t) for t in g["texts"] if str(t)]
        # Keep explicit line-break markers in captions (literal "\n"), without trailing marker.
        paragraph_text = "\\n".join(lines)
        if not paragraph_text:
            continue
        out.append(
            {
                "text_paragraph": paragraph_text,
                "font": str(g["font"]),
                "text_color": str(g["text_color"]),
                "role": str(g["role"]),
            }
        )
    return out


def _group_content_blocks_with_lines(content_list: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Build blocks split by (paragraph_index, role) in order, while preserving line-level texts.
    """
    blocks: List[Dict[str, object]] = []
    cur: Optional[Dict[str, object]] = None
    for item in content_list:
        para_idx = int(item.get("paragraph_index", 0) or 0)
        role = str(item.get("role", "body"))
        text = str(item.get("text", ""))
        if (
            cur is None
            or int(cur["paragraph_index"]) != para_idx
            or str(cur["role"]) != role
        ):
            cur = {
                "paragraph_index": para_idx,
                "role": role,
                "font": str(item.get("font", "")),
                "text_color": str(item.get("text_color", "")),
                "is_bold": bool(item.get("is_bold", False)),
                "has_shadow": bool(item.get("has_shadow", False)),
                "lines": [],
            }
            blocks.append(cur)
        else:
            cur["is_bold"] = bool(cur.get("is_bold", False)) or bool(item.get("is_bold", False))
            cur["has_shadow"] = bool(cur.get("has_shadow", False)) or bool(item.get("has_shadow", False))
        if text:
            cast_lines = cur["lines"]
            assert isinstance(cast_lines, list)
            cast_lines.append(text)
    return blocks


def _layout_variant_to_alignment(layout_mode: str, layout_variant: str) -> Tuple[str, bool]:
    v = (layout_variant or "").strip()
    if layout_mode == "vertical":
        return ("vertical", False)
    if layout_mode == "title_subtitle":
        return ("center", False)
    justify = v.startswith("justify_")
    if justify:
        v = v[len("justify_") :]
    if "@" in v:
        v = v.split("@", 1)[0]
    parts = [p for p in v.split("_") if p]
    h = parts[-1] if parts else "left"
    if h not in {"left", "center", "right"}:
        h = "left"
    return (h, justify)


def _load_alignment_phrases(path: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Load zh/en lists of natural-language phrases per alignment key (left/center/right/vertical).
    Used for L3/L4 captions instead of raw meta strings like \"center\".
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"alignment phrases file not found: {path}") from e
    if not isinstance(data, dict):
        raise ValueError(f"alignment phrases must be a JSON object: {path}")
    out: Dict[str, Dict[str, List[str]]] = {}
    for lang in ("zh", "en"):
        obj = data.get(lang)
        if not isinstance(obj, dict):
            raise ValueError(f"alignment phrases[{lang}] must be an object: {path}")
        inner: Dict[str, List[str]] = {}
        for key in ("left", "center", "right", "vertical"):
            lst = obj.get(key)
            if isinstance(lst, list) and lst and all(isinstance(x, str) for x in lst):
                inner[key] = list(lst)
            else:
                inner[key] = [key]
        out[lang] = inner
    return out


def _caption_placeholder_text(text: str) -> str:
    """Raw text for {title_text} / {paragraph_text} / {line_text}. Quotation marks belong in the template only."""
    return str(text)


def _apply_caption_font_color_template(template: str, font_name: str, text_color: str) -> str:
    """Replace {font_name} / {text_color}; strip decorative quotes around placeholders (same rule as L2)."""
    t = template
    for ph in ("{font_name}", "{text_color}"):
        t = (
            t.replace(f"“{ph}”", ph)
            .replace(f"\"{ph}\"", ph)
            .replace(f"'{ph}'", ph)
            .replace(f"‘{ph}’", ph)
        )
    return t.replace("{font_name}", font_name).replace("{text_color}", text_color)


def _caption_style_labels_for_block(block: Dict[str, object], rng: random.Random) -> Tuple[str, str, str, str]:
    raw_font = str(block.get("font", "") or "")
    raw_color = str(block.get("text_color", "") or "")
    font_zh, font_en = _pick_font_caption_labels(raw_font, rng=rng, font_map=WORKER_FONT_MAP)
    color_zh, color_en = _pick_color_caption_labels(raw_color, rng=rng, color_map=WORKER_COLOR_MAP)
    return font_zh, font_en, color_zh, color_en


def _count_l3_body_blocks_with_text(blocks: Sequence[Dict[str, object]]) -> int:
    n = 0
    for b in blocks:
        if str(b.get("role", "body")) != "body":
            continue
        lines = b.get("lines", [])
        if not isinstance(lines, list):
            continue
        joined = "\\n".join(str(x) for x in lines if str(x))
        if joined.strip():
            n += 1
    return n


def _pick_alignment_caption_phrase(align_key: str, lang: str, rng: random.Random) -> str:
    if WORKER_ALIGNMENT_PHRASES is None:
        return align_key
    phrases = WORKER_ALIGNMENT_PHRASES.get(lang, {}).get(align_key)
    if not phrases:
        return align_key
    return rng.choice(phrases)


def _build_caption_L3(
    content_list: Sequence[Dict[str, object]],
    layout_mode: str,
    layout_variant: str,
    background_color: str,
    has_background_image: bool,
    templates: Dict[str, Dict[str, List[str]]],
    rng: random.Random,
) -> Tuple[str, str]:
    """Scene + layout + per-paragraph body text (paragraph_desc); L4 uses line_desc per line."""
    blocks = _group_content_blocks_with_lines(content_list)
    body_total = _count_l3_body_blocks_with_text(blocks)
    overall_align, overall_justify = _layout_variant_to_alignment(layout_mode, layout_variant)
    align_zh = _pick_alignment_caption_phrase(overall_align, "zh", rng)
    align_en = _pick_alignment_caption_phrase(overall_align, "en", rng)

    def _pick(lang: str, key: str) -> str:
        return rng.choice(templates[lang][key])

    out_zh: List[str] = []
    out_en: List[str] = []
    if has_background_image:
        out_zh.append(_pick("zh", "scene_bg_image"))
        out_en.append(_pick("en", "scene_bg_image"))
    else:
        bg_color_zh, bg_color_en = _pick_color_caption_labels(
            background_color, rng=rng, color_map=WORKER_COLOR_MAP
        )
        out_zh.append(_pick("zh", "scene_bg").replace("{text_bg_color}", bg_color_zh))
        out_en.append(_pick("en", "scene_bg").replace("{text_bg_color}", bg_color_en))
    out_zh.append(_pick("zh", "scene_text_fragment"))
    out_en.append(_pick("en", "scene_text_fragment"))
    overall_justify_clause_zh = "，整体两端对齐" if overall_justify else ""
    overall_justify_clause_en = ", fully justified" if overall_justify else ""
    out_zh.append(
        _pick("zh", "scene_overall_align")
        .replace("{overall_align}", align_zh)
        .replace("{overall_justify_clause}", overall_justify_clause_zh)
    )
    out_en.append(
        _pick("en", "scene_overall_align")
        .replace("{overall_align}", align_en)
        .replace("{overall_justify_clause}", overall_justify_clause_en)
    )
    if WORKER_LAYOUT_MODE_DESC_L3 is not None:
        mode_key = "title_body" if layout_mode == "title_body" else "full_text"
        out_zh.append(rng.choice(WORKER_LAYOUT_MODE_DESC_L3["zh"][mode_key]))
        out_en.append(rng.choice(WORKER_LAYOUT_MODE_DESC_L3["en"][mode_key]))

    body_para_no = 0
    for b in blocks:
        role = str(b.get("role", "body"))
        lines = b.get("lines", [])
        text_paragraph = ""
        if isinstance(lines, list):
            line_texts = [str(x) for x in lines if str(x)]
            text_paragraph = "\\n".join(line_texts)
        title_t = _caption_placeholder_text(text_paragraph)
        if role == "title":
            out_zh.append(
                _pick("zh", "title_intro")
                .replace("{title_text}", title_t)
            )
            out_en.append(
                _pick("en", "title_intro")
                .replace("{title_text}", title_t)
            )
            fz, fe, cz, ce = _caption_style_labels_for_block(b, rng)
            out_zh.append(_apply_caption_font_color_template(_pick("zh", "title_style"), fz, cz))
            out_en.append(_apply_caption_font_color_template(_pick("en", "title_style"), fe, ce))
            if bool(b.get("is_bold", False)):
                out_zh.append(_pick("zh", "effect_bold"))
                out_en.append(_pick("en", "effect_bold"))
            if bool(b.get("has_shadow", False)):
                out_zh.append(_pick("zh", "effect_shadow"))
                out_en.append(_pick("en", "effect_shadow"))
            continue

        if isinstance(lines, list):
            joined = "\\n".join(str(x) for x in lines if str(x))
            if joined.strip():
                body_para_no += 1
                para_t = _caption_placeholder_text(joined)
                if body_para_no >= 2:
                    prev_s = str(body_para_no - 1)
                    next_s = str(body_para_no)
                    out_zh.append(
                        _pick("zh", "paragraph_bridge")
                        .replace("{prev_no}", prev_s)
                        .replace("{next_no}", next_s)
                    )
                    out_en.append(
                        _pick("en", "paragraph_bridge")
                        .replace("{prev_no}", prev_s)
                        .replace("{next_no}", next_s)
                    )
                total_s = str(body_total)
                idx_s = str(body_para_no)
                out_zh.append(
                    _pick("zh", "paragraph_index_intro")
                    .replace("{paragraph_no}", idx_s)
                    .replace("{paragraph_total}", total_s)
                )
                out_en.append(
                    _pick("en", "paragraph_index_intro")
                    .replace("{paragraph_no}", idx_s)
                    .replace("{paragraph_total}", total_s)
                )
                fz, fe, cz, ce = _caption_style_labels_for_block(b, rng)
                out_zh.append(_apply_caption_font_color_template(_pick("zh", "paragraph_style"), fz, cz))
                out_en.append(_apply_caption_font_color_template(_pick("en", "paragraph_style"), fe, ce))
                if bool(b.get("is_bold", False)):
                    out_zh.append(_pick("zh", "effect_bold"))
                    out_en.append(_pick("en", "effect_bold"))
                if bool(b.get("has_shadow", False)):
                    out_zh.append(_pick("zh", "effect_shadow"))
                    out_en.append(_pick("en", "effect_shadow"))
                out_zh.append(
                    _pick("zh", "paragraph_desc")
                    .replace("{paragraph_no}", str(body_para_no))
                    .replace("{paragraph_text}", para_t)
                )
                out_en.append(
                    _pick("en", "paragraph_desc")
                    .replace("{paragraph_no}", str(body_para_no))
                    .replace("{paragraph_text}", para_t)
                )
            else:
                if bool(b.get("is_bold", False)):
                    out_zh.append(_pick("zh", "effect_bold"))
                    out_en.append(_pick("en", "effect_bold"))
                if bool(b.get("has_shadow", False)):
                    out_zh.append(_pick("zh", "effect_shadow"))
                    out_en.append(_pick("en", "effect_shadow"))
        else:
            if bool(b.get("is_bold", False)):
                out_zh.append(_pick("zh", "effect_bold"))
                out_en.append(_pick("en", "effect_bold"))
            if bool(b.get("has_shadow", False)):
                out_zh.append(_pick("zh", "effect_shadow"))
                out_en.append(_pick("en", "effect_shadow"))

    return "".join(out_zh), "".join(out_en)


def _build_caption_L4(
    content_list: Sequence[Dict[str, object]],
    layout_mode: str,
    layout_variant: str,
    background_color: str,
    has_background_image: bool,
    templates: Dict[str, Dict[str, List[str]]],
    rng: random.Random,
) -> Tuple[str, str]:
    """Scene + layout + per-body-segment intro + per-line body text (line_desc)."""
    blocks = _group_content_blocks_with_lines(content_list)
    overall_align, overall_justify = _layout_variant_to_alignment(layout_mode, layout_variant)
    align_zh = _pick_alignment_caption_phrase(overall_align, "zh", rng)
    align_en = _pick_alignment_caption_phrase(overall_align, "en", rng)

    def _pick(lang: str, key: str) -> str:
        return rng.choice(templates[lang][key])

    out_zh: List[str] = []
    out_en: List[str] = []
    if not has_background_image:
        bg_color_zh, bg_color_en = _pick_color_caption_labels(
            background_color, rng=rng, color_map=WORKER_COLOR_MAP
        )
        out_zh.append(_pick("zh", "scene_bg").replace("{text_bg_color}", bg_color_zh))
        out_en.append(_pick("en", "scene_bg").replace("{text_bg_color}", bg_color_en))
    overall_justify_clause_zh = "，整体两端对齐" if overall_justify else ""
    overall_justify_clause_en = ", fully justified" if overall_justify else ""
    out_zh.append(
        _pick("zh", "scene_overall_align")
        .replace("{overall_align}", align_zh)
        .replace("{overall_justify_clause}", overall_justify_clause_zh)
    )
    out_en.append(
        _pick("en", "scene_overall_align")
        .replace("{overall_align}", align_en)
        .replace("{overall_justify_clause}", overall_justify_clause_en)
    )
    if WORKER_LAYOUT_MODE_DESC_L3 is not None:
        mode_key = "title_body" if layout_mode == "title_body" else "full_text"
        out_zh.append(rng.choice(WORKER_LAYOUT_MODE_DESC_L3["zh"][mode_key]))
        out_en.append(rng.choice(WORKER_LAYOUT_MODE_DESC_L3["en"][mode_key]))

    body_para_no = 0
    for b in blocks:
        role = str(b.get("role", "body"))
        lines = b.get("lines", [])
        text_paragraph = ""
        if isinstance(lines, list):
            text_paragraph = "".join(str(x) for x in lines if str(x))
        title_t = _caption_placeholder_text(text_paragraph)
        if role == "title":
            out_zh.append(
                _pick("zh", "title_intro")
                .replace("{title_text}", title_t)
            )
            out_en.append(
                _pick("en", "title_intro")
                .replace("{title_text}", title_t)
            )
            fz, fe, cz, ce = _caption_style_labels_for_block(b, rng)
            out_zh.append(_apply_caption_font_color_template(_pick("zh", "title_style"), fz, cz))
            out_en.append(_apply_caption_font_color_template(_pick("en", "title_style"), fe, ce))
            if bool(b.get("is_bold", False)):
                out_zh.append(_pick("zh", "effect_bold"))
                out_en.append(_pick("en", "effect_bold"))
            if bool(b.get("has_shadow", False)):
                out_zh.append(_pick("zh", "effect_shadow"))
                out_en.append(_pick("en", "effect_shadow"))
            continue

        line_list = lines if isinstance(lines, list) else []
        has_line_content = any(str(x).strip() for x in line_list)
        has_effect = bool(b.get("is_bold", False)) or bool(b.get("has_shadow", False))
        if not has_line_content and not has_effect:
            continue

        body_para_no += 1
        seg_no = str(body_para_no)
        out_zh.append(_pick("zh", "body_segment_intro").replace("{paragraph_no}", seg_no))
        out_en.append(_pick("en", "body_segment_intro").replace("{paragraph_no}", seg_no))
        fz, fe, cz, ce = _caption_style_labels_for_block(b, rng)
        out_zh.append(
            _apply_caption_font_color_template(
                _pick("zh", "body_segment_style").replace("{paragraph_no}", seg_no), fz, cz
            )
        )
        out_en.append(
            _apply_caption_font_color_template(
                _pick("en", "body_segment_style").replace("{paragraph_no}", seg_no), fe, ce
            )
        )

        if bool(b.get("is_bold", False)):
            out_zh.append(_pick("zh", "effect_bold"))
            out_en.append(_pick("en", "effect_bold"))
        if bool(b.get("has_shadow", False)):
            out_zh.append(_pick("zh", "effect_shadow"))
            out_en.append(_pick("en", "effect_shadow"))
        for i, line in enumerate(line_list, start=1):
            txt = str(line)
            if not txt:
                continue
            line_t = _caption_placeholder_text(txt)
            out_zh.append(_pick("zh", "line_desc").replace("{line_no}", str(i)).replace("{line_text}", line_t))
            out_en.append(_pick("en", "line_desc").replace("{line_no}", str(i)).replace("{line_text}", line_t))

    return "".join(out_zh), "".join(out_en)


def _textbbox_for_placed(draw: ImageDraw.ImageDraw, placed) -> Tuple[float, float, float, float]:
    """Match renderer/layout anchor so line clustering aligns with drawn glyphs."""
    kw = {}
    if getattr(placed, "anchor", None):
        kw["anchor"] = placed.anchor
    return draw.textbbox(
        (placed.x, placed.y), placed.text, font=placed.font, embedded_color=True, **kw
    )


def _bounds_for_placed(draw: ImageDraw.ImageDraw, placed) -> Optional[Tuple[float, float, float, float]]:
    """Bounding box for clustering; keeps whitespace glyphs when textbbox is degenerate."""
    left, top, right, bottom = _textbbox_for_placed(draw, placed)
    if right > left and bottom > top:
        return (float(left), float(top), float(right), float(bottom))
    if not placed.text:
        return None
    font = getattr(placed, "font", None)
    if font is None:
        return None
    fs = max(1.0, float(getattr(font, "size", 24) or 24))
    try:
        tw = float(draw.textlength(placed.text, font=font))
    except Exception:
        tw = 0.0
    if tw <= 1e-6:
        tw = max(1.0, 0.25 * fs * len(placed.text))
    x, y = float(placed.x), float(placed.y)
    anch = getattr(placed, "anchor", None)
    if anch == "ls":
        return (x, y - 0.78 * fs, x + tw, y + 0.22 * fs)
    if anch is None:
        return (x, y, x + tw, y + fs)
    return (x, y - 0.78 * fs, x + tw, y + 0.22 * fs)


def _pick_line_style_anchor(group: List[Dict]) -> Dict:
    """Prefer the color that covers the most characters (LTR); tie-break by leftmost span."""
    by_cx = sorted(group, key=lambda i: i["cx"])
    color_weight: Dict[str, int] = defaultdict(int)
    first_pos: Dict[str, int] = {}
    for i, item in enumerate(by_cx):
        c = str(item.get("color", ""))
        color_weight[c] += len(str(item.get("text", "")))
        if c not in first_pos:
            first_pos[c] = i
    best_color = max(
        color_weight,
        key=lambda c: (color_weight[c], -first_pos.get(c, 0)),
    )
    return next((it for it in by_cx if str(it.get("color", "")) == best_color), group[0])


def _pick_line_font_metrics(group: List[Dict]) -> Tuple[str, int, str]:
    """Use the font of non-emoji text on the line; avoid recording Apple Color Emoji for mixed lines."""
    by_cx = sorted(group, key=lambda i: i["cx"])
    non_emoji = [
        it
        for it in by_cx
        if str(it.get("corpus_type", "")) != "emoji" and str(it.get("text", "")).strip()
    ]
    if non_emoji:
        best = min(
            non_emoji,
            key=lambda it: (-len(str(it.get("text", ""))), by_cx.index(it)),
        )
        return (
            str(best.get("font_name", "")),
            int(best.get("font_size", 0) or 0),
            str(best.get("font_style", "normal")),
        )
    if not by_cx:
        return "", 0, "normal"
    best = min(by_cx, key=lambda it: (-len(str(it.get("text", ""))), by_cx.index(it)))
    return (
        str(best.get("font_name", "")),
        int(best.get("font_size", 0) or 0),
        str(best.get("font_style", "normal")),
    )


def run_generation(config_path: str, font_category_override: Optional[str] = None) -> None:
    config = load_config(config_path, font_category_override=font_category_override)
    cfg_path = Path(config_path).expanduser().resolve()
    config_dir = cfg_path.parent

    corpus_items = load_corpus_items(config, config_dir)
    corpus_pools = build_corpus_pools(corpus_items)
    text_corpus_types = tuple(sorted(corpus_pools.keys()))
    source_sampling_weights_by_corpus = _build_source_sampling_weights(config, config_dir, corpus_pools)
    title_corpus_pools: Dict[str, Dict[str, List[str]]] = {}
    title_source_sampling_weights_by_corpus: Dict[str, Dict[str, float]] = {}
    if config.title_corpus_sources:
        title_items = load_title_corpus_items(config, config_dir)
        title_corpus_pools = build_corpus_pools(title_items)
        title_source_sampling_weights_by_corpus = _build_title_source_sampling_weights(
            config, config_dir, title_corpus_pools
        )
    bootstrap_font_manager = FontCoverageManager()
    emoji_candidates = _collect_supported_emojis(config, config_dir, bootstrap_font_manager)
    background_image_paths = _collect_background_image_paths(config, config_dir)
    if not getattr(config, "caption_templates_L1_path", None):
        raise RuntimeError(
            "caption_templates_L1_path is required in config YAML root for parquet caption fields."
        )
    templates_path = Path(config.caption_templates_L1_path).expanduser()
    if not templates_path.is_absolute():
        templates_path = config_dir / templates_path
    templates_path = templates_path.resolve()
    caption_templates_L1 = _load_caption_templates_L1(templates_path)

    if not getattr(config, "caption_templates_L2_path", None):
        raise RuntimeError(
            "caption_templates_L2_path is required in config YAML root for parquet caption L2 fields."
        )
    templates_path_L2 = Path(config.caption_templates_L2_path).expanduser()
    if not templates_path_L2.is_absolute():
        templates_path_L2 = config_dir / templates_path_L2
    templates_path_L2 = templates_path_L2.resolve()
    caption_templates_L2 = _load_caption_templates_title_body(templates_path_L2)
    if not getattr(config, "caption_templates_L3_path", None):
        raise RuntimeError("caption_templates_L3_path is required in config YAML root for parquet caption L3 fields.")
    templates_path_L3 = Path(config.caption_templates_L3_path).expanduser()
    if not templates_path_L3.is_absolute():
        templates_path_L3 = config_dir / templates_path_L3
    templates_path_L3 = templates_path_L3.resolve()
    caption_templates_L3 = _load_caption_templates_L3_scene(templates_path_L3)
    templates_path_L4: Path
    if getattr(config, "caption_templates_L4_path", None):
        templates_path_L4 = Path(str(config.caption_templates_L4_path)).expanduser()
        if not templates_path_L4.is_absolute():
            templates_path_L4 = config_dir / templates_path_L4
    else:
        templates_path_L4 = config_dir.parent / "templates" / "caption_templates_L4.json"
    templates_path_L4 = templates_path_L4.resolve()
    caption_templates_L4 = _load_caption_templates_L4(templates_path_L4)
    layout_mode_desc_L3_path = config_dir.parent / "templates" / "layout_mode_desc_L3.json"
    layout_mode_desc_L3 = _load_layout_mode_desc_L3(layout_mode_desc_L3_path.resolve())
    alignment_phrases_path = config_dir.parent / "templates" / "alignment_phrases.json"
    alignment_phrases = _load_alignment_phrases(alignment_phrases_path.resolve())
    font_map_path = config_dir.parent / "templates" / "font_map.json"
    font_map = _load_font_map(font_map_path.resolve())
    color_map_path = config_dir.parent / "templates" / "color_map.json"
    color_map = _load_color_map(color_map_path.resolve())

    output_root = resolve_config_path(config.output.root_dir, config_dir)
    image_root = output_root / config.output.image_dir
    parquet_root = output_root / config.output.parquet_dir
    image_root.mkdir(parents=True, exist_ok=True)
    parquet_root.mkdir(parents=True, exist_ok=True)
    base_seed = config.seed if config.seed is not None else random.randint(0, 2**31 - 1)
    workers = _resolve_parallel_workers(config)
    worker_state = {
        "config": config,
        "config_dir": str(config_dir),
        "image_root": str(image_root),
        "corpus_pools": corpus_pools,
        "text_corpus_types": text_corpus_types,
        "source_sampling_weights_by_corpus": source_sampling_weights_by_corpus,
        "emoji_candidates": emoji_candidates,
        "background_image_paths": background_image_paths,
        "title_corpus_pools": title_corpus_pools,
        "title_source_sampling_weights_by_corpus": title_source_sampling_weights_by_corpus,
        "caption_templates_L1": caption_templates_L1,
        "caption_templates_L2": caption_templates_L2,
        "caption_templates_L3": caption_templates_L3,
        "caption_templates_L4": caption_templates_L4,
        "layout_mode_desc_L3": layout_mode_desc_L3,
        "alignment_phrases": alignment_phrases,
        "font_map": font_map,
        "color_map": color_map,
    }

    for round_idx in range(1, config.num_rounds + 1):
        rows, _sample_logs = _generate_round_parallel(
            config=config,
            round_idx=round_idx,
            base_seed=base_seed,
            workers=workers,
            worker_state=worker_state,
        )
        round_parquet_path = parquet_root / f"{_format_round_dir_name(round_idx)}.parquet"
        _write_round_parquet(round_parquet_path, rows)
        round_dir_name = _format_round_dir_name(round_idx)
        round_image_dir = image_root / round_dir_name
        print(f"[Round {round_idx:04d}] image_dir: {round_image_dir}")
        print(f"[Round {round_idx:04d}] parquet_path: {round_parquet_path}")


def _generate_round_parallel(
    config: RenderConfig,
    round_idx: int,
    base_seed: int,
    workers: int,
    worker_state: Dict,
) -> Tuple[List[Dict], List[Dict]]:
    rows: List[Dict] = []
    sample_logs: List[Dict] = []
    tasks: List[Tuple[int, int, int]] = []
    for sample_idx in range(1, config.samples_per_round + 1):
        task_seed = base_seed + round_idx * 1_000_003 + sample_idx * 9_973
        tasks.append((round_idx, sample_idx, task_seed))

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(worker_state,),
    ) as executor:
        future_to_sample = {executor.submit(_generate_sample_task, task): task[1] for task in tasks}
        progress = tqdm(total=len(tasks), desc=f"Round {round_idx}", unit="sample")
        for future in as_completed(future_to_sample):
            result = future.result()
            rows.append(result["row"])
            sample_logs.append(result["log"])
            progress.update(1)
        progress.close()

    rows.sort(key=lambda row: row["ID"])
    sample_logs.sort(key=lambda item: item["image_id"])
    return rows, sample_logs


def _init_worker(state: Dict) -> None:
    global WORKER_STATE, WORKER_FONT_MANAGER, WORKER_CAPTION_TEMPLATES_L1, WORKER_CAPTION_TEMPLATES_L2, WORKER_CAPTION_TEMPLATES_L3, WORKER_CAPTION_TEMPLATES_L4, WORKER_LAYOUT_MODE_DESC_L3, WORKER_ALIGNMENT_PHRASES, WORKER_FONT_MAP, WORKER_COLOR_MAP
    WORKER_STATE = state
    WORKER_FONT_MANAGER = FontCoverageManager()
    WORKER_CAPTION_TEMPLATES_L1 = WORKER_STATE.get("caption_templates_L1")
    WORKER_CAPTION_TEMPLATES_L2 = WORKER_STATE.get("caption_templates_L2")
    WORKER_CAPTION_TEMPLATES_L3 = WORKER_STATE.get("caption_templates_L3")
    WORKER_CAPTION_TEMPLATES_L4 = WORKER_STATE.get("caption_templates_L4")
    WORKER_LAYOUT_MODE_DESC_L3 = WORKER_STATE.get("layout_mode_desc_L3")
    WORKER_ALIGNMENT_PHRASES = WORKER_STATE.get("alignment_phrases")
    WORKER_FONT_MAP = WORKER_STATE.get("font_map")
    WORKER_COLOR_MAP = WORKER_STATE.get("color_map")


def _generate_sample_task(task: Tuple[int, int, int]) -> Dict:
    round_idx, sample_idx, seed = task
    return _generate_single_sample(round_idx=round_idx, sample_idx=sample_idx, seed=seed)


def _generate_single_sample(round_idx: int, sample_idx: int, seed: int) -> Dict:
    config: RenderConfig = WORKER_STATE["config"]
    config_dir = Path(WORKER_STATE["config_dir"])
    image_root = Path(WORKER_STATE["image_root"])
    corpus_pools = WORKER_STATE["corpus_pools"]
    text_corpus_types: Sequence[str] = WORKER_STATE["text_corpus_types"]
    source_sampling_weights_by_corpus: Dict[str, Dict[str, float]] = WORKER_STATE[
        "source_sampling_weights_by_corpus"
    ]
    emoji_candidates: Sequence[str] = WORKER_STATE["emoji_candidates"]
    background_image_paths: Sequence[str] = WORKER_STATE.get("background_image_paths", [])
    title_corpus_pools: Dict[str, Dict[str, Sequence[str]]] = WORKER_STATE.get("title_corpus_pools") or {}
    title_source_sampling_weights_by_corpus: Dict[str, Dict[str, float]] = (
        WORKER_STATE.get("title_source_sampling_weights_by_corpus") or {}
    )
    font_manager = WORKER_FONT_MANAGER
    if font_manager is None:
        raise RuntimeError("Worker font manager is not initialized.")

    round_dir = image_root / _format_round_dir_name(round_idx)
    round_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    max_retries = 60
    last_error = None
    background = "#FFFFFF"
    initial_planned_paragraphs: Optional[int] = None
    paragraph_drop_count = 0
    for attempt in range(max_retries):
        try:
            sampled_canvas = _sample_canvas_config(config, rng)
            background_bundle = _pick_background_for_sample(
                background_image_paths=background_image_paths,
                canvas_cfg=config.canvas,
                rng=rng,
            )
            background = background_bundle["background_color"]
            background_image = background_bundle.get("background_image")
            if background_image is not None:
                sampled_canvas = _with_canvas_size(
                    sampled_canvas,
                    width=background_image.width,
                    height=background_image.height,
                )
            template_name = _pick_style_template(config, rng)
            if template_name:
                layout_mode = "title_subtitle"
                layout_variant = "centered"
            else:
                layout_mode = _pick_layout_mode(config, rng)
                if layout_mode == "title_subtitle":
                    layout_mode = "mixed_line"
                layout_variant = _pick_layout_variant(config, layout_mode, rng)
            active_text_corpus_types = _filter_corpus_types_for_layout(layout_mode, text_corpus_types)
            if not active_text_corpus_types and not template_name:
                raise ValueError(f"No corpus types available for layout_mode={layout_mode}")
            active_source_sampling_weights = _resolve_active_source_sampling_weights(
                source_sampling_weights_by_corpus, active_text_corpus_types
            )
            # Gradually reduce complexity on repeated failures to keep generation stable.
            effective_max_segments = max(
                config.text.min_segments_per_image,
                config.text.max_segments_per_image - attempt // 12,
            )
            effective_max_units = max(
                config.text.min_corpus_units_per_segment,
                config.text.max_corpus_units_per_segment - attempt // 10,
            )
            effective_min_units = config.text.min_corpus_units_per_segment
            if layout_variant.startswith("justify_"):
                # When using justify, proactively increase corpus amount on retries
                # so text can form >=2 lines before dropping the final line.
                effective_min_units = min(
                    effective_max_units,
                    config.text.min_corpus_units_per_segment + attempt // 6,
                )
            effective_max_emoji = max(
                _retry_min_emoji_cap(config),
                config.text.max_emojis_between_segments - attempt // 8,
            )
            effective_emoji_prob = max(0.05, config.text.emoji_insert_probability - 0.02 * (attempt // 10))
            no_emoji_hit = rng.random() < float(getattr(config.text, "no_emoji_image_probability", 0.0))
            effective_min_emoji = config.text.min_emojis_between_segments
            if no_emoji_hit:
                effective_emoji_prob = 0.0
                effective_min_emoji = 0
                effective_max_emoji = 0
            if template_name:
                template_cfg = _resolve_template_config(config, template_name)
                segments = build_title_subtitle_segments_from_pools(
                    corpus_pools=corpus_pools,
                    source_sampling_weights_by_corpus=source_sampling_weights_by_corpus,
                    emoji_candidates=emoji_candidates,
                    template_cfg=template_cfg,
                    default_emoji_probability=effective_emoji_prob,
                    default_min_emojis_between_units=effective_min_emoji,
                    default_max_emojis_between_units=effective_max_emoji,
                    rng=rng,
                )
            elif layout_mode == "title_body":
                segments, paragraph_stats = _build_title_body_segments_resilient(
                    corpus_pools=corpus_pools,
                    source_sampling_weights_by_corpus=active_source_sampling_weights,
                    title_corpus_pools=title_corpus_pools,
                    title_source_sampling_weights_by_corpus=title_source_sampling_weights_by_corpus,
                    emoji_candidates=emoji_candidates,
                    emoji_insert_probability=effective_emoji_prob,
                    min_emojis_between_units=effective_min_emoji,
                    max_emojis_between_units=effective_max_emoji,
                    min_segments=config.text.min_segments_per_image,
                    max_segments=effective_max_segments,
                    chinese_min_length=config.text.chinese_min_length,
                    chinese_max_length=config.text.chinese_max_length,
                    english_min_length=config.text.english_min_length,
                    english_max_length=config.text.english_max_length,
                    title_corpus_units_min=config.text.title_corpus_units_min,
                    title_corpus_units_max=config.text.title_corpus_units_max,
                    config=config,
                    config_dir=config_dir,
                    font_manager=font_manager,
                    background_color=background,
                    sampled_canvas=sampled_canvas,
                    layout_variant=layout_variant,
                    rng=rng,
                )
                template_name = "title_body"
                paragraph_drop_count = int(paragraph_stats.get("dropped_paragraphs", 0))
                if initial_planned_paragraphs is None:
                    initial_planned_paragraphs = int(paragraph_stats.get("planned_paragraphs", 0))
            elif layout_mode == "dual_column":
                dual_sub_mode = rng.choice(["full_text", "title_body"])
                if dual_sub_mode == "title_body":
                    segments = _build_title_body_segments_from_pools(
                        corpus_pools=corpus_pools,
                        source_sampling_weights_by_corpus=active_source_sampling_weights,
                        title_corpus_pools=title_corpus_pools,
                        title_source_sampling_weights_by_corpus=title_source_sampling_weights_by_corpus,
                        emoji_candidates=emoji_candidates,
                        emoji_insert_probability=effective_emoji_prob,
                        min_emojis_between_units=effective_min_emoji,
                        max_emojis_between_units=effective_max_emoji,
                        min_segments=config.text.min_segments_per_image,
                        max_segments=effective_max_segments,
                        min_units_per_body=effective_min_units,
                        max_units_per_body=effective_max_units,
                        title_corpus_units_min=config.text.title_corpus_units_min,
                        title_corpus_units_max=config.text.title_corpus_units_max,
                        rng=rng,
                    )
                    template_name = "title_body"
                else:
                    segments, full_text_stats = _build_full_text_segments_resilient(
                        corpus_pools=corpus_pools,
                        text_corpus_types=active_text_corpus_types,
                        source_sampling_weights_by_corpus=active_source_sampling_weights,
                        emoji_candidates=emoji_candidates,
                        min_segments=config.text.min_segments_per_image,
                        max_segments=effective_max_segments,
                        chinese_min_length=config.text.chinese_min_length,
                        chinese_max_length=config.text.chinese_max_length,
                        english_min_length=config.text.english_min_length,
                        english_max_length=config.text.english_max_length,
                        emoji_insert_probability=effective_emoji_prob,
                        min_emojis_between_units=effective_min_emoji,
                        max_emojis_between_units=effective_max_emoji,
                        config=config,
                        config_dir=config_dir,
                        font_manager=font_manager,
                        background_color=background,
                        sampled_canvas=sampled_canvas,
                        layout_mode="full_text",
                        layout_variant=layout_variant,
                        rng=rng,
                    )
                    paragraph_drop_count = int(full_text_stats.get("dropped_paragraphs", 0))
                    if initial_planned_paragraphs is None:
                        initial_planned_paragraphs = int(full_text_stats.get("planned_paragraphs", 0))
            else:
                if layout_mode == "full_text":
                    segments, full_text_stats = _build_full_text_segments_resilient(
                        corpus_pools=corpus_pools,
                        text_corpus_types=active_text_corpus_types,
                        source_sampling_weights_by_corpus=active_source_sampling_weights,
                        emoji_candidates=emoji_candidates,
                        min_segments=config.text.min_segments_per_image,
                        max_segments=effective_max_segments,
                        chinese_min_length=config.text.chinese_min_length,
                        chinese_max_length=config.text.chinese_max_length,
                        english_min_length=config.text.english_min_length,
                        english_max_length=config.text.english_max_length,
                        emoji_insert_probability=effective_emoji_prob,
                        min_emojis_between_units=effective_min_emoji,
                        max_emojis_between_units=effective_max_emoji,
                        config=config,
                        config_dir=config_dir,
                        font_manager=font_manager,
                        background_color=background,
                        sampled_canvas=sampled_canvas,
                        layout_mode=layout_mode,
                        layout_variant=layout_variant,
                        rng=rng,
                    )
                    paragraph_drop_count = int(full_text_stats.get("dropped_paragraphs", 0))
                    if initial_planned_paragraphs is None:
                        initial_planned_paragraphs = int(full_text_stats.get("planned_paragraphs", 0))
                else:
                    segments = build_inline_emoji_segments_from_pools(
                        corpus_pools=corpus_pools,
                        text_corpus_types=active_text_corpus_types,
                        source_sampling_weights_by_corpus=active_source_sampling_weights,
                        emoji_candidates=emoji_candidates,
                        min_segments=config.text.min_segments_per_image,
                        max_segments=effective_max_segments,
                        min_units_per_segment=effective_min_units,
                        max_units_per_segment=effective_max_units,
                        emoji_insert_probability=effective_emoji_prob,
                        min_emojis_between_units=effective_min_emoji,
                        max_emojis_between_units=effective_max_emoji,
                        rng=rng,
                    )
            if attempt >= 20:
                segments = _truncate_segments_for_retry(segments, attempt)
            planned_paragraphs = _count_planned_paragraphs(
                segments=segments,
                layout_mode=layout_mode,
                template_name=template_name,
            )
            if initial_planned_paragraphs is None:
                initial_planned_paragraphs = planned_paragraphs
            styled_segments = build_styled_segments(
                sampled_segments=segments,
                config=config,
                config_dir=config_dir,
                font_manager=font_manager,
                background_color=background,
                rng=rng,
                template_name=template_name,
                sampled_canvas=sampled_canvas,
            )
            layout_result = layout_segments(
                segments=styled_segments,
                canvas=sampled_canvas,
                text_cfg=config.text,
                layout_mode=layout_mode,
                layout_variant=layout_variant,
                template_name=template_name,
                allow_partial_layout=True,
                rng=rng,
            )
            if not _has_rendered_text(layout_result):
                raise ValueError("No rendered text placements; resample.")
            image_id = _build_image_id(sample_idx, rng)
            image_path = round_dir / f"{image_id}.png"
            render_image(
                canvas_cfg=sampled_canvas,
                layout_result=layout_result,
                background_color=background,
                out_path=image_path,
                background_image=background_image,
            )
            relative_img_path = _build_relative_img_path(
                image_path=image_path,
                path_prefix=str(getattr(config.output, "relative_img_path_prefix", "") or ""),
            )
            if WORKER_CAPTION_TEMPLATES_L1 is None:
                raise RuntimeError("Worker caption templates not initialized (caption_templates_L1 missing).")
            caption_template_zh = rng.choice(WORKER_CAPTION_TEMPLATES_L1["zh"])
            caption_template_en = rng.choice(WORKER_CAPTION_TEMPLATES_L1["en"])
            if WORKER_CAPTION_TEMPLATES_L2 is None:
                raise RuntimeError("Worker caption templates not initialized (caption_templates_L2 missing).")
            if WORKER_CAPTION_TEMPLATES_L3 is None:
                raise RuntimeError("Worker caption templates not initialized (caption_templates_L3 missing).")
            if WORKER_CAPTION_TEMPLATES_L4 is None:
                raise RuntimeError("Worker caption templates not initialized (caption_templates_L4 missing).")
            caption_template_zh_title = rng.choice(WORKER_CAPTION_TEMPLATES_L2["zh"]["title"])
            caption_template_zh_body = rng.choice(WORKER_CAPTION_TEMPLATES_L2["zh"]["body"])
            caption_template_en_title = rng.choice(WORKER_CAPTION_TEMPLATES_L2["en"]["title"])
            caption_template_en_body = rng.choice(WORKER_CAPTION_TEMPLATES_L2["en"]["body"])
            row = _build_parquet_row(
                image_id=image_id,
                relative_img_path=relative_img_path,
                sampled_canvas=sampled_canvas,
                layout_mode=layout_mode,
                layout_result=layout_result,
                styled_segments=styled_segments,
                background=background,
                has_background_image=background_image is not None,
                template_name=template_name,
                caption_template_zh=caption_template_zh,
                caption_template_en=caption_template_en,
                caption_template_zh_title=caption_template_zh_title,
                caption_template_zh_body=caption_template_zh_body,
                caption_template_en_title=caption_template_en_title,
                caption_template_en_body=caption_template_en_body,
                rng=rng,
            )
            actual_paragraphs = _count_actual_paragraphs(
                layout_result=layout_result,
                layout_mode=layout_mode,
                template_name=template_name,
            )
            reason = _build_mismatch_reason(
                initial_planned_paragraphs=initial_planned_paragraphs or planned_paragraphs,
                planned_paragraphs_on_success=planned_paragraphs,
                actual_paragraphs=actual_paragraphs,
                attempt=attempt,
                dropped_paragraphs=paragraph_drop_count,
            )
            log_row = {
                "image_id": image_id,
                "planned_paragraphs_initial": int(initial_planned_paragraphs or planned_paragraphs),
                "planned_paragraphs_success_attempt": int(planned_paragraphs),
                "actual_paragraphs": int(actual_paragraphs),
                "layout_mode": layout_mode,
                "layout_variant": layout_result.layout_variant,
                "attempt": int(attempt),
                "mismatch_reason": reason,
                "corpus_build_dropped_segments": int(paragraph_drop_count),
                "planned_paragraph_texts": _planned_paragraph_texts_for_log(
                    segments, layout_mode, template_name
                ),
            }
            return {"row": row, "log": log_row}
        except ValueError as exc:
            last_error = exc
            continue
    fallback = _generate_single_sample_fallback(
        config=config,
        corpus_pools=corpus_pools,
        text_corpus_types=text_corpus_types,
        config_dir=config_dir,
        image_root=image_root,
        round_dir=round_dir,
        round_idx=round_idx,
        sample_idx=sample_idx,
        font_manager=font_manager,
        rng=rng,
    )
    if fallback is not None:
        planned_txt = fallback.pop("_planned_paragraph_texts_log", [])
        fallback["_generation_log"] = {
            "image_id": fallback["ID"],
            "planned_paragraphs_initial": int(initial_planned_paragraphs or 1),
            "planned_paragraphs_success_attempt": 1,
            "actual_paragraphs": 1,
            "layout_mode": "fallback",
            "layout_variant": "fallback",
            "attempt": int(max_retries),
            "mismatch_reason": "主流程重试失败，使用fallback最小样本",
            "planned_paragraph_texts": planned_txt,
        }
        return {"row": fallback, "log": fallback["_generation_log"]}
    raise RuntimeError(f"Failed to generate sample after {max_retries} retries: {last_error}")


def _count_planned_paragraphs(
    segments: Sequence[CorpusItem],
    layout_mode: str,
    template_name: Optional[str],
) -> int:
    if layout_mode == "title_body" or template_name == "title_body":
        return sum(1 for item in segments if getattr(item, "role", "") == "title")
    text_seen = any(getattr(item, "corpus_type", "") != "line_break" for item in segments)
    if not text_seen:
        return 0
    line_breaks = sum(1 for item in segments if getattr(item, "corpus_type", "") == "line_break")
    return line_breaks + 1


def _planned_paragraph_texts_for_log(
    segments: Sequence[CorpusItem],
    layout_mode: str,
    template_name: Optional[str],
) -> List[Dict[str, object]]:
    """
    Per-paragraph source text as sampled (before layout/render), for generation_log.jsonl.
    title_body: one entry per title block with title + merged body string.
    Other modes: entries split on line_break corpus items, full paragraph text in "text".
    """
    if layout_mode == "title_body" or template_name == "title_body":
        out: List[Dict[str, object]] = []
        i = 0
        para_idx = 0
        n = len(segments)
        while i < n:
            if getattr(segments[i], "role", "") != "title":
                i += 1
                continue
            title_text = str(segments[i].content)
            i += 1
            body_parts: List[str] = []
            while i < n and getattr(segments[i], "role", "") != "title":
                seg = segments[i]
                if getattr(seg, "corpus_type", "") == "line_break":
                    i += 1
                    continue
                body_parts.append(str(seg.content))
                i += 1
            out.append(
                {
                    "paragraph_index": para_idx,
                    "title": title_text,
                    "body": "".join(body_parts),
                }
            )
            para_idx += 1
        return out

    blobs: List[str] = []
    current: List[str] = []
    for seg in segments:
        if getattr(seg, "corpus_type", "") == "line_break":
            if current:
                blobs.append("".join(current))
                current = []
            continue
        current.append(str(seg.content))
    if current:
        blobs.append("".join(current))
    return [{"paragraph_index": idx, "text": t} for idx, t in enumerate(blobs)]


def _count_actual_paragraphs(layout_result, layout_mode: str, template_name: Optional[str]) -> int:
    placements = getattr(layout_result, "placements", None) or []
    if not placements:
        return 0
    if layout_mode == "title_body" or template_name == "title_body":
        count = 0
        prev_title = False
        for placed in placements:
            is_title = getattr(placed, "role", "") == "title"
            if is_title and not prev_title:
                count += 1
            prev_title = is_title
        return count
    centers: List[float] = []
    for placed in placements:
        top = float(getattr(placed, "y", 0))
        font = getattr(placed, "font", None)
        size = float(getattr(font, "size", 0) or 0)
        h = max(1.0, size * 0.8)
        centers.append(top + h * 0.5)
    if not centers:
        return 0
    centers.sort()
    line_tol = 10.0
    line_centers: List[float] = []
    for c in centers:
        if not line_centers or abs(c - line_centers[-1]) > line_tol:
            line_centers.append(c)
        else:
            line_centers[-1] = (line_centers[-1] + c) * 0.5
    if len(line_centers) <= 1:
        return 1
    gaps = [max(0.0, line_centers[i + 1] - line_centers[i]) for i in range(len(line_centers) - 1)]
    threshold = max(18.0, median(gaps) * 2.2) if gaps else 18.0
    breaks = sum(1 for g in gaps if g > threshold)
    return 1 + breaks


def _build_mismatch_reason(
    initial_planned_paragraphs: int,
    planned_paragraphs_on_success: int,
    actual_paragraphs: int,
    attempt: int,
    dropped_paragraphs: int = 0,
) -> str:
    if actual_paragraphs == initial_planned_paragraphs:
        if dropped_paragraphs <= 0:
            return ""
        return "有段落替换语料重试5次仍失败，已丢弃该段"
    reasons: List[str] = []
    if attempt > 0 or planned_paragraphs_on_success != initial_planned_paragraphs:
        reasons.append("重试后重采样，计划段落数发生变化")
    if dropped_paragraphs > 0:
        reasons.append(f"有{dropped_paragraphs}段替换语料重试5次仍失败，被丢弃")
    if actual_paragraphs < planned_paragraphs_on_success:
        reasons.append("排版阶段部分段落未成功落版（画布高度或段落跳过）")
    elif actual_paragraphs > planned_paragraphs_on_success:
        reasons.append("实际段落按OCR聚类统计，可能受大间距分段影响")
    if not reasons:
        reasons.append("计划与实际统计口径不同")
    return "；".join(reasons)


def _build_relative_img_path(image_path: Path, path_prefix: str) -> str:
    round_dir_name = image_path.parent.name
    image_stem = image_path.stem
    if path_prefix:
        return f"{path_prefix}/{round_dir_name}.zip/{image_stem}"
    return f"{round_dir_name}.zip/{image_stem}"


def _format_round_dir_name(round_idx: int) -> str:
    return f"{round_idx:06d}"


def _build_image_id(sample_idx: int, rng: random.Random) -> str:
    # Keep deterministic under seeded rng while using UUID naming format.
    return f"{sample_idx:08d}_{uuid.UUID(int=rng.getrandbits(128))}"


def _resolve_parallel_workers(config: RenderConfig) -> int:
    if config.parallel_workers is not None:
        return config.parallel_workers
    cpu_count = os.cpu_count() or 4
    return max(1, min(8, cpu_count))


def _collect_supported_emojis(
    config: RenderConfig, config_dir: Path, font_manager: FontCoverageManager
) -> List[str]:
    # Single-codepoint emoji/symbol ranges that are commonly rendered by color emoji fonts.
    ranges = [
        (0x1F300, 0x1F5FF),
        (0x1F600, 0x1F64F),
        (0x1F680, 0x1F6FF),
        (0x1F700, 0x1F77F),
        (0x1F780, 0x1F7FF),
        (0x1F800, 0x1F8FF),
        (0x1F900, 0x1F9FF),
        (0x1FA00, 0x1FAFF),
        (0x2600, 0x26FF),
        (0x2700, 0x27BF),
    ]
    emoji_font_paths = [
        str(resolve_config_path(path, config_dir)) for path in config.fonts_by_corpus.get("emoji", [])
    ]
    supported: List[str] = []
    for start, end in ranges:
        for codepoint in range(start, end + 1):
            ch = chr(codepoint)
            ok = any(
                font_manager.supports_text(font_path, ch)
                and font_manager.is_renderable(font_path, ch, config.text.max_font_size)
                for font_path in emoji_font_paths
            )
            if ok:
                supported.append(ch)
    if supported:
        return supported
    return EMOJI_CANDIDATES


def _truncate_segments_for_retry(segments, attempt: int):
    # Additional safety valve for very long units (e.g., long news sentences).
    max_cn_len = max(6, 28 - attempt // 6)
    max_en_words = max(3, 10 - attempt // 8)
    truncated = []
    for item in segments:
        role = getattr(item, "role", "body")
        if item.corpus_type == "chinese":
            truncated.append(
                type(item)(
                    content=item.content[:max_cn_len],
                    corpus_type=item.corpus_type,
                    source_path=item.source_path,
                    role=role,
                )
            )
        elif item.corpus_type == "english":
            words = item.content.split()
            truncated_text = " ".join(words[:max_en_words]) if words else item.content[:max_cn_len]
            truncated.append(
                type(item)(
                    content=truncated_text,
                    corpus_type=item.corpus_type,
                    source_path=item.source_path,
                    role=role,
                )
            )
        else:
            truncated.append(item)
    return truncated


def _generate_single_sample_fallback(
    config: RenderConfig,
    corpus_pools: Dict[str, Dict[str, Sequence[str]]],
    text_corpus_types: Sequence[str],
    config_dir: Path,
    image_root: Path,
    round_dir: Path,
    round_idx: int,
    sample_idx: int,
    font_manager: FontCoverageManager,
    rng: random.Random,
) -> Optional[Dict]:
    # Last-resort fallback to avoid full round failure on rare complex-font layout cases.
    layout_mode = _pick_layout_mode(config, rng)
    if layout_mode == "title_subtitle":
        layout_mode = "mixed_line"
    layout_variant = _pick_layout_variant(config, layout_mode, rng)
    active_text_corpus_types = _filter_corpus_types_for_layout(layout_mode, text_corpus_types)
    for corpus_type in active_text_corpus_types:
        units_by_source = corpus_pools.get(corpus_type) or {}
        if not units_by_source:
            continue
        source_path = rng.choice(list(units_by_source.keys()))
        units = list(units_by_source.get(source_path, []))
        if not units:
            continue
        if corpus_type == "chinese":
            raw = rng.choice(units)
            text = raw[:10]
        else:
            # english jsonl 常为「一词一行」：从池中多次抽样再拼接，避免 fallback 只有单个 token。
            parts: List[str] = []
            for _ in range(8):
                u = str(rng.choice(units)).strip()
                if u:
                    parts.append(u.split()[0])
            text = " ".join(parts) if parts else "text"
        segments = [CorpusItem(content=text, corpus_type=corpus_type, source_path="__fallback__")]
        try:
            sampled_canvas = _sample_canvas_config(config, rng)
            background_bundle = _pick_background_for_sample(
                background_image_paths=WORKER_STATE.get("background_image_paths", []),
                canvas_cfg=config.canvas,
                rng=rng,
            )
            background = background_bundle["background_color"]
            background_image = background_bundle.get("background_image")
            if background_image is not None:
                sampled_canvas = _with_canvas_size(
                    sampled_canvas,
                    width=background_image.width,
                    height=background_image.height,
                )
            styled_segments = build_styled_segments(
                sampled_segments=segments,
                config=config,
                config_dir=config_dir,
                font_manager=font_manager,
                background_color=background,
                rng=rng,
                template_name=None,
                sampled_canvas=sampled_canvas,
            )
            layout_result = layout_segments(
                segments=styled_segments,
                canvas=sampled_canvas,
                text_cfg=config.text,
                layout_mode=layout_mode,
                layout_variant=layout_variant,
                template_name=None,
                allow_partial_layout=True,
                rng=rng,
            )
            if not _has_rendered_text(layout_result):
                continue
            image_id = _build_image_id(sample_idx, rng)
            image_path = round_dir / f"{image_id}.png"
            render_image(
                canvas_cfg=sampled_canvas,
                layout_result=layout_result,
                background_color=background,
                out_path=image_path,
                background_image=background_image,
            )
            relative_img_path = _build_relative_img_path(
                image_path=image_path,
                path_prefix=str(getattr(config.output, "relative_img_path_prefix", "") or ""),
            )
            if WORKER_CAPTION_TEMPLATES_L1 is None:
                raise RuntimeError("Worker caption templates not initialized (caption_templates_L1 missing).")
            caption_template_zh = rng.choice(WORKER_CAPTION_TEMPLATES_L1["zh"])
            caption_template_en = rng.choice(WORKER_CAPTION_TEMPLATES_L1["en"])
            if WORKER_CAPTION_TEMPLATES_L2 is None:
                raise RuntimeError("Worker caption templates not initialized (caption_templates_L2 missing).")
            if WORKER_CAPTION_TEMPLATES_L3 is None:
                raise RuntimeError("Worker caption templates not initialized (caption_templates_L3 missing).")
            if WORKER_CAPTION_TEMPLATES_L4 is None:
                raise RuntimeError("Worker caption templates not initialized (caption_templates_L4 missing).")
            caption_template_zh_title = rng.choice(WORKER_CAPTION_TEMPLATES_L2["zh"]["title"])
            caption_template_zh_body = rng.choice(WORKER_CAPTION_TEMPLATES_L2["zh"]["body"])
            caption_template_en_title = rng.choice(WORKER_CAPTION_TEMPLATES_L2["en"]["title"])
            caption_template_en_body = rng.choice(WORKER_CAPTION_TEMPLATES_L2["en"]["body"])
            row = _build_parquet_row(
                image_id=image_id,
                relative_img_path=relative_img_path,
                sampled_canvas=sampled_canvas,
                layout_mode=layout_mode,
                layout_result=layout_result,
                styled_segments=styled_segments,
                background=background,
                has_background_image=background_image is not None,
                template_name=None,
                caption_template_zh=caption_template_zh,
                caption_template_en=caption_template_en,
                caption_template_zh_title=caption_template_zh_title,
                caption_template_zh_body=caption_template_zh_body,
                caption_template_en_title=caption_template_en_title,
                caption_template_en_body=caption_template_en_body,
                rng=rng,
            )
            row["_planned_paragraph_texts_log"] = _planned_paragraph_texts_for_log(
                segments, layout_mode, None
            )
            return row
        except Exception:
            continue
    return None


def _write_round_parquet(parquet_path: Path, rows: List[Dict]) -> None:
    for row in rows:
        if "_generation_log" in row:
            row.pop("_generation_log", None)
    for row in rows:
        sid = row.get("ID", "__unknown__")
        _warn_if_parquet_json_list_field(sid, row.get("content_dict"), "content_dict")
        _warn_if_parquet_json_list_field(sid, row.get("ocr_attribute"), "ocr_attribute")
    schema = pa.schema(
        [
            ("ID", pa.string()),
            ("relative_img_path", pa.string()),
            ("width", pa.int32()),
            ("height", pa.int32()),
            ("content_dict", pa.string()),
            ("ocr_attribute", pa.string()),
            ("caption_zh_L1", pa.string()),
            ("caption_en_L1", pa.string()),
            ("caption_zh_L2", pa.string()),
            ("caption_en_L2", pa.string()),
            ("caption_zh_L3", pa.string()),
            ("caption_en_L3", pa.string()),
            ("caption_zh_L4", pa.string()),
            ("caption_en_L4", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_path)


def _write_round_log(log_path: Path, sample_logs: Sequence[Dict]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        for row in sample_logs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_parquet_row(
    image_id: str,
    relative_img_path: str,
    sampled_canvas,
    layout_mode: str,
    layout_result,
    styled_segments,
    background: str,
    has_background_image: bool,
    template_name: Optional[str],
    caption_template_zh: str,
    caption_template_en: str,
    caption_template_zh_title: str,
    caption_template_zh_body: str,
    caption_template_en_title: str,
    caption_template_en_body: str,
    rng: random.Random,
) -> Dict:
    ocr_rows = _build_ocr_attributes(
        placements=layout_result.placements,
        canvas_width=sampled_canvas.width,
        canvas_height=sampled_canvas.height,
        layout_mode=layout_mode,
        layout_variant=layout_result.layout_variant,
    )
    content_list, ocr_rows = _build_content_and_ocr_from_rendered_lines(
        placements=layout_result.placements,
        canvas_width=sampled_canvas.width,
        canvas_height=sampled_canvas.height,
        ocr_rows=ocr_rows,
        layout_mode=layout_mode,
        layout_variant=layout_result.layout_variant,
        background=background,
        template_name=template_name,
    )
    paragraph_rows = _group_content_by_paragraph(content_list)
    text_all = "\\n".join(row["text_paragraph"] for row in paragraph_rows)
    caption_zh_L1 = _apply_caption_template(caption_template_zh, text_all)
    caption_en_L1 = _apply_caption_template(caption_template_en, text_all)
    caption_lines_zh_L2: List[str] = []
    caption_lines_en_L2: List[str] = []
    for row in paragraph_rows:
        text_paragraph = row["text_paragraph"]
        role = row["role"]
        raw_font_name = row["font"]
        font_name_zh, font_name_en = _pick_font_caption_labels(raw_font_name, rng=rng, font_map=WORKER_FONT_MAP)
        raw_text_color = row["text_color"]
        text_color_zh, text_color_en = _pick_color_caption_labels(
            raw_text_color, rng=rng, color_map=WORKER_COLOR_MAP
        )
        is_title = role == "title"
        tpl_zh = caption_template_zh_title if is_title else caption_template_zh_body
        tpl_en = caption_template_en_title if is_title else caption_template_en_body
        caption_lines_zh_L2.append(
            _apply_caption_template_title_body(
                tpl_zh, font_name=font_name_zh, text_color=text_color_zh, text_paragraph=text_paragraph
            )
        )
        caption_lines_en_L2.append(
            _apply_caption_template_title_body(
                tpl_en, font_name=font_name_en, text_color=text_color_en, text_paragraph=text_paragraph
            )
        )
    caption_zh_L2 = "\n".join(caption_lines_zh_L2)
    caption_en_L2 = "\n".join(caption_lines_en_L2)

    if WORKER_CAPTION_TEMPLATES_L3 is None:
        raise RuntimeError("Worker caption templates not initialized (caption_templates_L3 missing).")
    caption_zh_L3, caption_en_L3 = _build_caption_L3(
        content_list=content_list,
        layout_mode=layout_mode,
        layout_variant=layout_result.layout_variant,
        background_color=background,
        has_background_image=has_background_image,
        templates=WORKER_CAPTION_TEMPLATES_L3,
        rng=rng,
    )
    if WORKER_CAPTION_TEMPLATES_L4 is None:
        raise RuntimeError("Worker caption templates not initialized (caption_templates_L4 missing).")
    caption_zh_L4, caption_en_L4 = _build_caption_L4(
        content_list=content_list,
        layout_mode=layout_mode,
        layout_variant=layout_result.layout_variant,
        background_color=background,
        has_background_image=has_background_image,
        templates=WORKER_CAPTION_TEMPLATES_L4,
        rng=rng,
    )
    # Store as Python literal string so booleans stay True/False for eval-based consumers.
    # str(list[dict]) also keeps emoji as UTF-8 characters instead of \\U... escapes.
    content_dict = json.dumps(content_list, ensure_ascii=False)
    ocr_attribute = json.dumps(ocr_rows, ensure_ascii=False)
    return {
        "ID": image_id,
        "relative_img_path": relative_img_path,
        "width": sampled_canvas.width,
        "height": sampled_canvas.height,
        "content_dict": content_dict,
        "ocr_attribute": ocr_attribute,
        "caption_zh_L1": caption_zh_L1,
        "caption_en_L1": caption_en_L1,
        "caption_zh_L2": caption_zh_L2,
        "caption_en_L2": caption_en_L2,
        "caption_zh_L3": caption_zh_L3,
        "caption_en_L3": caption_en_L3,
        "caption_zh_L4": caption_zh_L4,
        "caption_en_L4": caption_en_L4,
    }


def _warn_if_parquet_json_list_field(
    sample_id: str, raw: Optional[str], field_name: str
) -> None:
    """Validate JSON (or legacy repr) list fields; used for content_dict and ocr_attribute."""
    if raw is None:
        warnings.warn(f"[{sample_id}] {field_name} is missing.", stacklevel=2)
        return
    parsed: object
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = eval(raw, {"__builtins__": {}}, {})
        except Exception as exc:
            warnings.warn(f"[{sample_id}] {field_name} parse failed: {exc}", stacklevel=2)
            return
    if not isinstance(parsed, list):
        warnings.warn(f"[{sample_id}] {field_name} parse result is not a list.", stacklevel=2)


def _build_content_list_from_paragraphs(
    styled_segments,
    layout_mode: str,
    layout_variant: str,
    background: str,
    template_name: Optional[str],
) -> List[Dict]:
    paragraphs: List[List] = []
    current: List = []
    for seg in styled_segments:
        if getattr(seg, "corpus_type", "") == "line_break":
            if current:
                paragraphs.append(current)
                current = []
            continue
        current.append(seg)
    if current:
        paragraphs.append(current)

    content_list: List[Dict] = []
    layout_format = f"{layout_mode}:{layout_variant}"
    for para in paragraphs:
        text = "".join(getattr(seg, "text", "") for seg in para)
        if not text:
            continue
        anchor = None
        for seg in para:
            if getattr(seg, "corpus_type", "") not in {"emoji", "line_break"}:
                anchor = seg
                break
        if anchor is None:
            anchor = para[0]
        font_style = str(getattr(anchor, "font_style", "normal"))
        content_list.append(
            {
                "text": text,
                "font": getattr(anchor, "font_name", ""),
                "font_size": int(getattr(anchor, "font_size", 0)),
                "is_italic": "italic" in font_style,
                "is_bold": "bold" in font_style,
                "text_color": getattr(anchor, "color", ""),
                "background_color": background,
                "layout_format": layout_format,
                "role": getattr(anchor, "role", "body"),
                "template_name": template_name or "",
            }
        )
    return content_list


def _build_content_list_from_ocr_rows(
    placements,
    canvas_width: int,
    canvas_height: int,
    ocr_rows: Sequence[Dict],
    layout_mode: str,
    layout_variant: str,
    background: str,
    template_name: Optional[str],
) -> List[Dict]:
    if not ocr_rows:
        return []
    style_rows = _build_grouped_style_rows(
        placements=placements,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        layout_mode=layout_mode,
        layout_variant=layout_variant,
    )
    layout_format = f"{layout_mode}:{layout_variant}"
    out: List[Dict] = []
    for idx, row in enumerate(ocr_rows):
        text = str(row.get("text", ""))
        if not text.strip():
            continue
        style = style_rows[idx] if idx < len(style_rows) else {}
        font_style = str(style.get("font_style", "normal"))
        out.append(
            {
                "text": text,
                "font": str(style.get("font_name", "")),
                "font_size": int(style.get("font_size", 0) or 0),
                "is_italic": "italic" in font_style,
                "is_bold": "bold" in font_style,
                "text_color": str(style.get("color", "")),
                "background_color": background,
                "layout_format": layout_format,
                "role": str(style.get("role", "body")),
                "template_name": template_name or "",
                "paragraph_index": int(style.get("paragraph_index", 0)),
            }
        )
    return out


def _build_content_and_ocr_from_rendered_lines(
    placements,
    canvas_width: int,
    canvas_height: int,
    ocr_rows: Sequence[Dict],
    layout_mode: str,
    layout_variant: str,
    background: str,
    template_name: Optional[str],
) -> Tuple[List[Dict], List[Dict]]:
    if not ocr_rows:
        return [], []
    style_rows = _build_grouped_style_rows(
        placements=placements,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        layout_mode=layout_mode,
        layout_variant=layout_variant,
    )
    line_rows: List[Dict] = []
    for idx, row in enumerate(ocr_rows):
        text = str(row.get("text", ""))
        if not text.strip():
            continue
        style = style_rows[idx] if idx < len(style_rows) else {}
        font_style = str(style.get("font_style", "normal"))
        line_rows.append(
            {
                "text": text,
                "bbox": row.get("bbox", []),
                "direction": row.get("direction", "horizontal_ltr"),
                "font": str(style.get("font_name", "")),
                "font_size": int(style.get("font_size", 0) or 0),
                "is_italic": "italic" in font_style,
                "is_bold": "bold" in font_style,
                "text_color": str(style.get("color", "")),
                "background_color": background,
                "layout_format": f"{layout_mode}:{layout_variant}",
                "role": str(style.get("role", "body")),
                "template_name": template_name or "",
                "paragraph_index": int(style.get("paragraph_index", 0)),
                "has_shadow": bool(style.get("has_shadow", False)),
            }
        )
    content_lines = [
        {
            "paragraph_index": int(row["paragraph_index"]),
            "text": row["text"],
            "font": row["font"],
            "font_size": int(row["font_size"]),
            "is_italic": bool(row["is_italic"]),
            "is_bold": bool(row["is_bold"]),
            "text_color": row["text_color"],
            "background_color": row["background_color"],
            "layout_format": row["layout_format"],
            "role": row["role"],
            "template_name": row["template_name"],
            "has_shadow": bool(row.get("has_shadow", False)),
        }
        for row in line_rows
    ]
    ocr_out = [
        {
            "text": row["text"],
            "bbox": row["bbox"],
            "direction": row["direction"],
            "paragraph_index": int(row["paragraph_index"]),
        }
        for row in line_rows
    ]
    return content_lines, ocr_out


def _build_grouped_style_rows(
    placements,
    canvas_width: int,
    canvas_height: int,
    layout_mode: str,
    layout_variant: str,
) -> List[Dict]:
    if not placements:
        return []
    draw = ImageDraw.Draw(Image.new("RGB", (max(1, canvas_width), max(1, canvas_height))))
    measured_items: List[Dict] = []
    for placed in placements:
        bb = _bounds_for_placed(draw, placed)
        if bb is None:
            continue
        left, top, right, bottom = bb
        measured_items.append(
            {
                "text": placed.text,
                "left": float(left),
                "top": float(top),
                "right": float(right),
                "bottom": float(bottom),
                "cx": (left + right) / 2.0,
                "cy": (top + bottom) / 2.0,
                "font_name": str(
                    getattr(placed, "base_font_name", "")
                    or (
                        getattr(placed, "font", None).path.split("/")[-1]
                        if getattr(getattr(placed, "font", None), "path", None)
                        else ""
                    )
                ),
                "font_size": int(getattr(getattr(placed, "font", None), "size", 0) or 0),
                "font_style": str(getattr(placed, "font_style", "normal")),
                "color": str(getattr(placed, "color", "")),
                "role": str(getattr(placed, "role", "body")),
                "paragraph_index": int(getattr(placed, "paragraph_index", 0)),
                "corpus_type": str(getattr(placed, "corpus_type", "")),
                "has_shadow": bool(
                    isinstance(getattr(placed, "effects", None), dict)
                    and isinstance(getattr(placed, "effects", None).get("shadow"), dict)
                    and bool(getattr(placed, "effects", None).get("shadow", {}).get("enabled", False))
                ),
            }
        )
    if not measured_items:
        return []
    if layout_mode == "dual_column":
        mid_x = canvas_width / 2.0
        left_items = [item for item in measured_items if item["cx"] < mid_x]
        right_items = [item for item in measured_items if item["cx"] >= mid_x]

        def _groups_for_column(column_items: List[Dict]) -> List[List[Dict]]:
            if not column_items:
                return []
            tol_base = median(item["bottom"] - item["top"] for item in column_items)
            tolerance = max(4.0, float(tol_base) * 0.6)
            col_groups = _cluster_by_axis(column_items, axis="cy", tolerance=tolerance)
            col_groups.sort(key=lambda g: sum(item["cy"] for item in g) / len(g))
            for group in col_groups:
                group.sort(key=lambda item: item["cx"])
            return col_groups

        groups = _groups_for_column(left_items) + _groups_for_column(right_items)
    elif layout_mode == "vertical":
        axis = "cx"
        tol_base = median(item["right"] - item["left"] for item in measured_items)
        tolerance = max(4.0, float(tol_base) * 0.6)
        groups = _cluster_by_axis(measured_items, axis=axis, tolerance=tolerance)
        reverse_cols = layout_variant == "rtl"
        groups.sort(key=lambda g: sum(item["cx"] for item in g) / len(g), reverse=reverse_cols)
        for group in groups:
            group.sort(key=lambda item: item["cy"])
    else:
        axis = "cy"
        tol_base = median(item["bottom"] - item["top"] for item in measured_items)
        tolerance = max(4.0, float(tol_base) * 0.6)
        groups = _cluster_by_axis(measured_items, axis=axis, tolerance=tolerance)
        groups.sort(key=lambda g: sum(item["cy"] for item in g) / len(g))
        for group in groups:
            group.sort(key=lambda item: item["cx"])
    out: List[Dict] = []
    for group in groups:
        color_anchor = _pick_line_style_anchor(group)
        font_name, font_size, font_style = _pick_line_font_metrics(group)
        para_idx = min(int(item.get("paragraph_index", 0)) for item in group)
        out.append(
            {
                "font_name": font_name,
                "font_size": font_size,
                "font_style": font_style,
                "color": str(color_anchor.get("color", "")),
                "role": str(color_anchor.get("role", "body")),
                "paragraph_index": para_idx,
                "has_shadow": any(bool(item.get("has_shadow", False)) for item in group),
            }
        )
    return out


def _build_ocr_attributes(
    placements,
    canvas_width: int,
    canvas_height: int,
    layout_mode: str,
    layout_variant: str,
) -> List[Dict]:
    if not placements:
        return []
    draw = ImageDraw.Draw(Image.new("RGB", (max(1, canvas_width), max(1, canvas_height))))
    measured_items: List[Dict] = []
    for placed in placements:
        bb = _bounds_for_placed(draw, placed)
        if bb is None:
            continue
        left, top, right, bottom = bb
        measured_items.append(
            {
                "text": placed.text,
                "left": float(left),
                "top": float(top),
                "right": float(right),
                "bottom": float(bottom),
                "cx": (left + right) / 2.0,
                "cy": (top + bottom) / 2.0,
            }
        )
    if not measured_items:
        return []

    if layout_mode == "dual_column":
        mid_x = canvas_width / 2.0
        left_items = [item for item in measured_items if item["cx"] < mid_x]
        right_items = [item for item in measured_items if item["cx"] >= mid_x]

        def _groups_for_column(column_items: List[Dict]) -> List[List[Dict]]:
            if not column_items:
                return []
            tol_base = median(item["bottom"] - item["top"] for item in column_items)
            tolerance = max(4.0, float(tol_base) * 0.6)
            col_groups = _cluster_by_axis(column_items, axis="cy", tolerance=tolerance)
            col_groups.sort(key=lambda g: sum(item["cy"] for item in g) / len(g))
            for group in col_groups:
                group.sort(key=lambda item: item["cx"])
            return col_groups

        groups = _groups_for_column(left_items) + _groups_for_column(right_items)
        direction = "horizontal_ltr"
    elif layout_mode == "vertical":
        axis = "cx"
        tol_base = median(item["right"] - item["left"] for item in measured_items)
        tolerance = max(4.0, float(tol_base) * 0.6)
        groups = _cluster_by_axis(measured_items, axis=axis, tolerance=tolerance)
        reverse_cols = layout_variant == "rtl"
        groups.sort(key=lambda g: sum(item["cx"] for item in g) / len(g), reverse=reverse_cols)
        for group in groups:
            group.sort(key=lambda item: item["cy"])
        direction = "vertical_rtl" if reverse_cols else "vertical_ltr"
    else:
        axis = "cy"
        tol_base = median(item["bottom"] - item["top"] for item in measured_items)
        tolerance = max(4.0, float(tol_base) * 0.6)
        groups = _cluster_by_axis(measured_items, axis=axis, tolerance=tolerance)
        groups.sort(key=lambda g: sum(item["cy"] for item in g) / len(g))
        for group in groups:
            group.sort(key=lambda item: item["cx"])
        direction = "horizontal_ltr"

    scale = 1024.0 / float(max(1, canvas_height))
    ocr_rows: List[Dict] = []
    for group in groups:
        text = "".join(item["text"] for item in group)
        left = min(item["left"] for item in group)
        top = min(item["top"] for item in group)
        right = max(item["right"] for item in group)
        bottom = max(item["bottom"] for item in group)
        bbox = [
            [int(round(left * scale)), int(round(top * scale))],
            [int(round(right * scale)), int(round(top * scale))],
            [int(round(right * scale)), int(round(bottom * scale))],
            [int(round(left * scale)), int(round(bottom * scale))],
        ]
        ocr_rows.append({"text": text, "bbox": bbox, "direction": direction})
    return ocr_rows


def _cluster_by_axis(items: Sequence[Dict], axis: str, tolerance: float) -> List[List[Dict]]:
    ordered = sorted(items, key=lambda item: item[axis])
    groups: List[List[Dict]] = []
    for item in ordered:
        if not groups:
            groups.append([item])
            continue
        cur_group = groups[-1]
        cur_center = sum(member[axis] for member in cur_group) / len(cur_group)
        if abs(item[axis] - cur_center) <= tolerance:
            cur_group.append(item)
        else:
            groups.append([item])
    return groups


def _has_rendered_text(layout_result) -> bool:
    placements = getattr(layout_result, "placements", None) or []
    for placed in placements:
        text = getattr(placed, "text", "")
        if isinstance(text, str) and text.strip():
            return True
    return False


def _pick_layout_mode(config: RenderConfig, rng: random.Random) -> str:
    modes = list(config.text.layout_modes)
    if not modes:
        raise ValueError("text.layout_modes cannot be empty.")
    weights_cfg = config.text.layout_mode_weights
    if not weights_cfg:
        return rng.choice(modes)
    weights = [float(weights_cfg.get(mode, 0.0)) for mode in modes]
    if all(w <= 0 for w in weights):
        raise ValueError("text.layout_mode_weights must include at least one positive weight.")
    return rng.choices(modes, weights=weights, k=1)[0]


def _pick_layout_variant(config: RenderConfig, layout_mode: str, rng: random.Random) -> str:
    configured = (config.text.layout_variants_by_mode or {}).get(layout_mode)
    if configured:
        return rng.choice(list(configured))
    if layout_mode in {"mixed_line", "segmented", "full_text", "title_body", "dual_column"}:
        return rng.choice(
            [
                "top_left",
                "top_center",
                "top_right",
                "middle_left",
                "middle_center",
                "middle_right",
                "bottom_left",
                "bottom_center",
                "bottom_right",
                "justify_top_left",
                "justify_middle_center",
                "justify_bottom_right",
            ]
        )
    if layout_mode == "vertical":
        anchor = rng.choice(["top_left", "top_center", "top_right"])
        direction = rng.choice(["rtl", "ltr"])
        return f"{anchor}@{direction}"
    if layout_mode == "title_subtitle":
        return "centered"
    raise ValueError(f"Unsupported layout mode for variant selection: {layout_mode}")


def _filter_corpus_types_for_layout(
    layout_mode: str, text_corpus_types: Sequence[str]
) -> Tuple[str, ...]:
    if layout_mode == "vertical":
        # Vertical layout only uses Chinese corpus by requirement.
        return tuple(corpus_type for corpus_type in text_corpus_types if corpus_type != "english")
    if layout_mode == "title_subtitle":
        return tuple(corpus_type for corpus_type in text_corpus_types if corpus_type in {"chinese", "english"})
    return tuple(text_corpus_types)


def _collect_background_image_paths(config: RenderConfig, config_dir: Path) -> List[str]:
    bg_dir_raw = config.canvas.background_images_dir
    if not bg_dir_raw:
        return []
    bg_dir = resolve_config_path(bg_dir_raw, config_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = [
        str(path)
        for path in bg_dir.iterdir()
        if path.is_file() and path.suffix.lower() in exts
    ]
    if not paths:
        warnings.warn(
            f"No background images found in directory: {bg_dir}. Falling back to solid colors.",
            stacklevel=2,
        )
    return paths


def _pick_background_for_sample(
    background_image_paths: Sequence[str],
    canvas_cfg,
    rng: random.Random,
) -> Dict[str, object]:
    if not background_image_paths:
        if not canvas_cfg.background_colors:
            raise ValueError("No background image and no background color available.")
        return {"background_color": rng.choice(canvas_cfg.background_colors), "background_image": None}

    target_side = int(getattr(canvas_cfg, "background_image_area_reference", 1024))
    target_area = float(target_side * target_side)
    tries = max(1, min(20, len(background_image_paths) * 2))
    for _ in range(tries):
        path = rng.choice(list(background_image_paths))
        with Image.open(path) as src:
            bg = src.convert("RGB")
        src_area = float(bg.width * bg.height)
        if src_area < target_area:
            warnings.warn(
                f"Background image area too small ({int(src_area)} < {int(target_area)}), resampling: {path}",
                stacklevel=2,
            )
            continue
        scale = math.sqrt(target_area / src_area)
        new_w = max(1, int(round(bg.width * scale)))
        new_h = max(1, int(round(bg.height * scale)))
        if new_w != bg.width or new_h != bg.height:
            bg = bg.resize((new_w, new_h), Image.Resampling.LANCZOS)
        dominant_rgb = _compute_dominant_color(bg)
        dominant_hex = "#{:02X}{:02X}{:02X}".format(*dominant_rgb)
        return {"background_color": dominant_hex, "background_image": bg}

    if canvas_cfg.background_colors:
        warnings.warn(
            "No eligible background image met target area; falling back to solid color background.",
            stacklevel=2,
        )
        return {"background_color": rng.choice(canvas_cfg.background_colors), "background_image": None}
    raise ValueError("No eligible background image met target area and no fallback background colors set.")


def _compute_dominant_color(image: Image.Image) -> Tuple[int, int, int]:
    thumb = image.copy()
    thumb.thumbnail((128, 128), Image.Resampling.BILINEAR)
    quantized = thumb.convert("P", palette=Image.ADAPTIVE, colors=8)
    palette = quantized.getpalette() or []
    counts = quantized.getcolors()
    if not counts:
        return (127, 127, 127)
    dominant_idx = max(counts, key=lambda pair: pair[0])[1]
    base = dominant_idx * 3
    if base + 2 >= len(palette):
        return (127, 127, 127)
    return (palette[base], palette[base + 1], palette[base + 2])


def _sample_canvas_config(config: RenderConfig, rng: random.Random):
    base = config.canvas
    ratios = base.aspect_ratios or ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9"]
    if base.short_edge_min is not None and base.short_edge_max is not None:
        short_min = base.short_edge_min
        short_max = base.short_edge_max
    elif (
        base.min_width is not None
        and base.max_width is not None
        and base.min_height is not None
        and base.max_height is not None
    ):
        short_min = min(base.min_width, base.min_height)
        short_max = min(base.max_width, base.max_height)
    else:
        return base
    if short_min > short_max:
        short_min, short_max = short_max, short_min
    short_edge = rng.randint(short_min, short_max)
    ratio_raw = rng.choice(ratios)
    rw, rh = _parse_aspect_ratio(ratio_raw)
    denom = float(min(rw, rh))
    width = int(round(short_edge * rw / denom))
    height = int(round(short_edge * rh / denom))
    return type(base)(
        width=width,
        height=height,
        min_width=base.min_width,
        max_width=base.max_width,
        min_height=base.min_height,
        max_height=base.max_height,
        short_edge_min=base.short_edge_min,
        short_edge_max=base.short_edge_max,
        aspect_ratios=base.aspect_ratios,
        margin=base.margin,
        background_colors=base.background_colors,
        min_text_bg_contrast_ratio=base.min_text_bg_contrast_ratio,
        background_images_dir=base.background_images_dir,
        background_image_area_reference=base.background_image_area_reference,
    )


def _with_canvas_size(canvas_cfg, width: int, height: int):
    return type(canvas_cfg)(
        width=max(1, int(width)),
        height=max(1, int(height)),
        min_width=canvas_cfg.min_width,
        max_width=canvas_cfg.max_width,
        min_height=canvas_cfg.min_height,
        max_height=canvas_cfg.max_height,
        short_edge_min=canvas_cfg.short_edge_min,
        short_edge_max=canvas_cfg.short_edge_max,
        aspect_ratios=canvas_cfg.aspect_ratios,
        margin=canvas_cfg.margin,
        background_colors=canvas_cfg.background_colors,
        min_text_bg_contrast_ratio=canvas_cfg.min_text_bg_contrast_ratio,
        background_images_dir=canvas_cfg.background_images_dir,
        background_image_area_reference=canvas_cfg.background_image_area_reference,
    )


def _parse_aspect_ratio(raw: str) -> Tuple[int, int]:
    if ":" not in raw:
        raise ValueError(f"Invalid aspect ratio: {raw}")
    left, right = raw.split(":", 1)
    w = int(left.strip())
    h = int(right.strip())
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid aspect ratio: {raw}")
    return w, h


def _retry_min_emoji_cap(config: RenderConfig) -> int:
    # Keep consecutive-emoji capability in retries when config allows it.
    if config.text.max_emojis_between_segments >= 2:
        return max(2, config.text.min_emojis_between_segments)
    return config.text.min_emojis_between_segments


def _build_source_sampling_weights(
    config: RenderConfig,
    config_dir: Path,
    corpus_pools: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Dict[str, float]]:
    weights: Dict[str, Dict[str, float]] = {
        corpus_type: {source_path: 1.0 for source_path in units_by_source.keys()}
        for corpus_type, units_by_source in corpus_pools.items()
    }
    for source in config.corpus_sources:
        source_path = str(resolve_config_path(source.path, config_dir))
        by_corpus = weights.get(source.corpus_type)
        if by_corpus is None:
            continue
        if source_path in by_corpus:
            by_corpus[source_path] = float(source.sample_weight)
    return weights


def _build_title_source_sampling_weights(
    config: RenderConfig,
    config_dir: Path,
    title_corpus_pools: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Dict[str, float]]:
    sources = config.title_corpus_sources or []
    if not sources:
        return {}
    weights: Dict[str, Dict[str, float]] = {
        corpus_type: {source_path: 1.0 for source_path in units_by_source.keys()}
        for corpus_type, units_by_source in title_corpus_pools.items()
    }
    for source in sources:
        source_path = str(resolve_config_path(source.path, config_dir))
        by_corpus = weights.get(source.corpus_type)
        if by_corpus is None:
            continue
        if source_path in by_corpus:
            by_corpus[source_path] = float(source.sample_weight)
    return weights


def _resolve_active_source_sampling_weights(
    source_sampling_weights_by_corpus: Dict[str, Dict[str, float]],
    active_text_corpus_types: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    return {
        corpus_type: dict(source_sampling_weights_by_corpus.get(corpus_type, {}))
        for corpus_type in active_text_corpus_types
    }


def _pick_style_template(config: RenderConfig, rng: random.Random) -> Optional[str]:
    weights_cfg = config.text.template_weights
    templates_cfg = config.text.style_templates
    if not weights_cfg or not templates_cfg:
        return None
    names = [name for name in weights_cfg.keys() if name in templates_cfg]
    if not names:
        return None
    weights = [float(weights_cfg.get(name, 0.0)) for name in names]
    if all(w <= 0 for w in weights):
        return None
    return rng.choices(names, weights=weights, k=1)[0]


def _resolve_template_config(config: RenderConfig, template_name: str) -> Dict[str, object]:
    templates_cfg = config.text.style_templates or {}
    cfg = templates_cfg.get(template_name)
    if not isinstance(cfg, dict):
        raise ValueError(f"Unknown style template: {template_name}")
    return dict(cfg)


def _build_title_body_segments_from_pools(
    corpus_pools: Dict[str, Dict[str, Sequence[str]]],
    source_sampling_weights_by_corpus: Dict[str, Dict[str, float]],
    title_corpus_pools: Dict[str, Dict[str, Sequence[str]]],
    title_source_sampling_weights_by_corpus: Dict[str, Dict[str, float]],
    emoji_candidates: Sequence[str],
    emoji_insert_probability: float,
    min_emojis_between_units: int,
    max_emojis_between_units: int,
    min_segments: int,
    max_segments: int,
    min_units_per_body: int,
    max_units_per_body: int,
    title_corpus_units_min: int,
    title_corpus_units_max: int,
    rng: random.Random,
) -> List[CorpusItem]:
    paragraph_count = rng.randint(max(1, min_segments), max(1, max_segments))
    out: List[CorpusItem] = []
    for p in range(paragraph_count):
        title, title_corpus_type, title_source = _sample_concatenated_title(
            title_corpus_pools=title_corpus_pools,
            title_weights_by_corpus=title_source_sampling_weights_by_corpus,
            body_corpus_pools=corpus_pools,
            body_weights_by_corpus=source_sampling_weights_by_corpus,
            min_units=title_corpus_units_min,
            max_units=title_corpus_units_max,
            rng=rng,
        )
        if not title:
            continue
        out.append(CorpusItem(content=title, corpus_type=title_corpus_type, source_path=title_source, role="title"))
        out.append(CorpusItem(content="\n", corpus_type="line_break", source_path="__line_break__", role="line_break"))

        body_corpus_type = rng.choice([c for c in corpus_pools.keys() if c in {"chinese", "english"}] or [title_corpus_type])
        body_units_by_source = corpus_pools.get(body_corpus_type, {})
        if not body_units_by_source:
            continue
        unit_count = rng.randint(max(1, min_units_per_body), max(1, max_units_per_body))
        chunks: List[str] = []
        for _ in range(unit_count):
            body_source = _pick_source_path_for_pipeline(
                body_units_by_source,
                source_sampling_weights_by_corpus.get(body_corpus_type, {}),
                rng,
            )
            raw = rng.choice(list(body_units_by_source[body_source]))
            if body_corpus_type == "english":
                chunks.append(normalize_english_corpus_unit(str(raw)))
            else:
                chunks.append(str(raw).strip())
        body_units = [c for c in chunks if c]
        if body_units:
            emoji_pool = tuple(emoji_candidates)
            for unit_idx, unit in enumerate(body_units):
                out.append(
                    CorpusItem(
                        content=unit,
                        corpus_type=body_corpus_type,
                        source_path="__title_body__",
                        role="body",
                    )
                )
                if unit_idx >= len(body_units) - 1:
                    continue
                emoji_count = max(0, min_emojis_between_units)
                emoji_upper = max(emoji_count, max_emojis_between_units)
                while emoji_count < emoji_upper:
                    if rng.random() >= max(0.0, min(1.0, emoji_insert_probability)):
                        break
                    emoji_count += 1
                if emoji_count == 0 and body_corpus_type == "english":
                    out.append(
                        CorpusItem(
                            content=" ",
                            corpus_type="english",
                            source_path="__inline_space__",
                            role="body",
                        )
                    )
                    continue
                if body_corpus_type == "english":
                    out.append(
                        CorpusItem(
                            content=" ",
                            corpus_type="english",
                            source_path="__inline_space__",
                            role="body",
                        )
                    )
                for _ in range(emoji_count):
                    if not emoji_pool:
                        break
                    out.append(
                        CorpusItem(
                            content=rng.choice(emoji_pool),
                            corpus_type="emoji",
                            source_path="__emoji_injected__",
                            role="body",
                        )
                    )
                if body_corpus_type == "english":
                    out.append(
                        CorpusItem(
                            content=" ",
                            corpus_type="english",
                            source_path="__inline_space__",
                            role="body",
                        )
                    )
        if p < paragraph_count - 1:
            out.append(CorpusItem(content="\n", corpus_type="line_break", source_path="__line_break__", role="line_break"))
    return out


def _build_title_body_segments_resilient(
    corpus_pools: Dict[str, Dict[str, Sequence[str]]],
    source_sampling_weights_by_corpus: Dict[str, Dict[str, float]],
    title_corpus_pools: Dict[str, Dict[str, Sequence[str]]],
    title_source_sampling_weights_by_corpus: Dict[str, Dict[str, float]],
    emoji_candidates: Sequence[str],
    emoji_insert_probability: float,
    min_emojis_between_units: int,
    max_emojis_between_units: int,
    min_segments: int,
    max_segments: int,
    chinese_min_length: int,
    chinese_max_length: int,
    english_min_length: int,
    english_max_length: int,
    title_corpus_units_min: int,
    title_corpus_units_max: int,
    config: RenderConfig,
    config_dir: Path,
    font_manager: FontCoverageManager,
    background_color: str,
    sampled_canvas,
    layout_variant: str,
    rng: random.Random,
) -> Tuple[List[CorpusItem], Dict[str, int]]:
    _ = (config, config_dir, font_manager, background_color, sampled_canvas, layout_variant)
    planned = _sample_title_body_paragraph_count(min_segments, max_segments, rng)
    accepted: List[CorpusItem] = []
    dropped = 0
    for _ in range(planned):
        paragraph_ok = False
        for _retry in range(5):
            para = _sample_one_title_body_paragraph(
                corpus_pools=corpus_pools,
                source_sampling_weights_by_corpus=source_sampling_weights_by_corpus,
                title_corpus_pools=title_corpus_pools,
                title_source_sampling_weights_by_corpus=title_source_sampling_weights_by_corpus,
                emoji_candidates=emoji_candidates,
                emoji_insert_probability=emoji_insert_probability,
                min_emojis_between_units=min_emojis_between_units,
                max_emojis_between_units=max_emojis_between_units,
                chinese_min_length=chinese_min_length,
                chinese_max_length=chinese_max_length,
                english_min_length=english_min_length,
                english_max_length=english_max_length,
                title_corpus_units_min=title_corpus_units_min,
                title_corpus_units_max=title_corpus_units_max,
                rng=rng,
            )
            if not para:
                continue
            candidate = list(accepted)
            if candidate:
                candidate.append(
                    CorpusItem(
                        content="\n",
                        corpus_type="line_break",
                        source_path="__line_break__",
                        role="line_break",
                    )
                )
            candidate.extend(para)
            accepted = candidate
            paragraph_ok = True
            break
        if not paragraph_ok:
            dropped += 1
    return accepted, {"planned_paragraphs": planned, "dropped_paragraphs": dropped}


def _sample_title_body_paragraph_count(min_segments: int, max_segments: int, rng: random.Random) -> int:
    low = max(1, min_segments)
    high = max(low, max_segments)
    return rng.randint(low, high)


def _sample_one_title_body_paragraph(
    corpus_pools: Dict[str, Dict[str, Sequence[str]]],
    source_sampling_weights_by_corpus: Dict[str, Dict[str, float]],
    title_corpus_pools: Dict[str, Dict[str, Sequence[str]]],
    title_source_sampling_weights_by_corpus: Dict[str, Dict[str, float]],
    emoji_candidates: Sequence[str],
    emoji_insert_probability: float,
    min_emojis_between_units: int,
    max_emojis_between_units: int,
    chinese_min_length: int,
    chinese_max_length: int,
    english_min_length: int,
    english_max_length: int,
    title_corpus_units_min: int,
    title_corpus_units_max: int,
    rng: random.Random,
) -> List[CorpusItem]:
    out: List[CorpusItem] = []
    title, title_corpus_type, title_source = _sample_concatenated_title(
        title_corpus_pools=title_corpus_pools,
        title_weights_by_corpus=title_source_sampling_weights_by_corpus,
        body_corpus_pools=corpus_pools,
        body_weights_by_corpus=source_sampling_weights_by_corpus,
        min_units=title_corpus_units_min,
        max_units=title_corpus_units_max,
        rng=rng,
    )
    if not title:
        return []
    out.append(CorpusItem(content=title, corpus_type=title_corpus_type, source_path=title_source, role="title"))
    out.append(CorpusItem(content="\n", corpus_type="line_break", source_path="__line_break__", role="line_break"))
    body_corpus_type = rng.choice([c for c in corpus_pools.keys() if c in {"chinese", "english"}] or [title_corpus_type])
    body_units_by_source = corpus_pools.get(body_corpus_type, {})
    if not body_units_by_source:
        return []
    body_units = _sample_units_to_target_length(
        corpus_type=body_corpus_type,
        units_by_source=body_units_by_source,
        source_sampling_weights=source_sampling_weights_by_corpus.get(body_corpus_type, {}),
        chinese_min_length=chinese_min_length,
        chinese_max_length=chinese_max_length,
        english_min_length=english_min_length,
        english_max_length=english_max_length,
        rng=rng,
    )
    if not body_units:
        return []
    emoji_pool = tuple(emoji_candidates)
    for unit_idx, unit in enumerate(body_units):
        out.append(
            CorpusItem(
                content=unit,
                corpus_type=body_corpus_type,
                source_path="__title_body__",
                role="body",
            )
        )
        if unit_idx >= len(body_units) - 1:
            continue
        emoji_count = max(0, min_emojis_between_units)
        emoji_upper = max(emoji_count, max_emojis_between_units)
        while emoji_count < emoji_upper:
            if rng.random() >= max(0.0, min(1.0, emoji_insert_probability)):
                break
            emoji_count += 1
        if emoji_count == 0 and body_corpus_type == "english":
            out.append(
                CorpusItem(
                    content=" ",
                    corpus_type="english",
                    source_path="__inline_space__",
                    role="body",
                )
            )
            continue
        if body_corpus_type == "english":
            out.append(
                CorpusItem(
                    content=" ",
                    corpus_type="english",
                    source_path="__inline_space__",
                    role="body",
                )
            )
        for _ in range(emoji_count):
            if not emoji_pool:
                break
            out.append(
                CorpusItem(
                    content=rng.choice(emoji_pool),
                    corpus_type="emoji",
                    source_path="__emoji_injected__",
                    role="body",
                )
            )
        if body_corpus_type == "english":
            out.append(
                CorpusItem(
                    content=" ",
                    corpus_type="english",
                    source_path="__inline_space__",
                    role="body",
                )
            )
    return out


def _build_full_text_segments_resilient(
    corpus_pools: Dict[str, Dict[str, Sequence[str]]],
    text_corpus_types: Sequence[str],
    source_sampling_weights_by_corpus: Dict[str, Dict[str, float]],
    emoji_candidates: Sequence[str],
    min_segments: int,
    max_segments: int,
    chinese_min_length: int,
    chinese_max_length: int,
    english_min_length: int,
    english_max_length: int,
    emoji_insert_probability: float,
    min_emojis_between_units: int,
    max_emojis_between_units: int,
    config: RenderConfig,
    config_dir: Path,
    font_manager: FontCoverageManager,
    background_color: str,
    sampled_canvas,
    layout_mode: str,
    layout_variant: str,
    rng: random.Random,
) -> Tuple[List[CorpusItem], Dict[str, int]]:
    _ = (config, config_dir, font_manager, background_color, sampled_canvas, layout_mode, layout_variant)
    planned = _sample_title_body_paragraph_count(min_segments, max_segments, rng)
    accepted: List[CorpusItem] = []
    dropped = 0
    for _ in range(planned):
        seg_ok = False
        for _retry in range(5):
            seg_items = _sample_one_full_text_segment(
                corpus_pools=corpus_pools,
                text_corpus_types=text_corpus_types,
                source_sampling_weights_by_corpus=source_sampling_weights_by_corpus,
                emoji_candidates=emoji_candidates,
                chinese_min_length=chinese_min_length,
                chinese_max_length=chinese_max_length,
                english_min_length=english_min_length,
                english_max_length=english_max_length,
                emoji_insert_probability=emoji_insert_probability,
                min_emojis_between_units=min_emojis_between_units,
                max_emojis_between_units=max_emojis_between_units,
                rng=rng,
            )
            if not seg_items:
                continue
            candidate = list(accepted)
            if candidate:
                candidate.append(
                    CorpusItem(
                        content="\n",
                        corpus_type="line_break",
                        source_path="__line_break__",
                        role="line_break",
                    )
                )
            candidate.extend(seg_items)
            accepted = candidate
            seg_ok = True
            break
        if not seg_ok:
            dropped += 1
    return accepted, {"planned_paragraphs": planned, "dropped_paragraphs": dropped}


def _sample_one_full_text_segment(
    corpus_pools: Dict[str, Dict[str, Sequence[str]]],
    text_corpus_types: Sequence[str],
    source_sampling_weights_by_corpus: Dict[str, Dict[str, float]],
    emoji_candidates: Sequence[str],
    chinese_min_length: int,
    chinese_max_length: int,
    english_min_length: int,
    english_max_length: int,
    emoji_insert_probability: float,
    min_emojis_between_units: int,
    max_emojis_between_units: int,
    rng: random.Random,
) -> List[CorpusItem]:
    corpus_types = [c for c in text_corpus_types if c in corpus_pools]
    if not corpus_types:
        return []
    corpus_type = rng.choice(corpus_types)
    units_by_source = corpus_pools.get(corpus_type, {})
    if not units_by_source:
        return []
    sampled_units = _sample_units_to_target_length(
        corpus_type=corpus_type,
        units_by_source=units_by_source,
        source_sampling_weights=source_sampling_weights_by_corpus.get(corpus_type, {}),
        chinese_min_length=chinese_min_length,
        chinese_max_length=chinese_max_length,
        english_min_length=english_min_length,
        english_max_length=english_max_length,
        rng=rng,
    )
    if not sampled_units:
        return []
    emoji_pool = tuple(emoji_candidates)
    out: List[CorpusItem] = []
    for unit_idx, unit in enumerate(sampled_units):
        out.append(CorpusItem(content=unit, corpus_type=corpus_type, source_path="__full_text__", role="body"))
        if unit_idx >= len(sampled_units) - 1:
            continue
        emoji_count = max(0, min_emojis_between_units)
        emoji_upper = max(emoji_count, max_emojis_between_units)
        while emoji_count < emoji_upper:
            if rng.random() >= max(0.0, min(1.0, emoji_insert_probability)):
                break
            emoji_count += 1
        if emoji_count == 0 and corpus_type == "english":
            out.append(CorpusItem(content=" ", corpus_type="english", source_path="__inline_space__", role="body"))
            continue
        if corpus_type == "english":
            out.append(CorpusItem(content=" ", corpus_type="english", source_path="__inline_space__", role="body"))
        for _ in range(emoji_count):
            if not emoji_pool:
                break
            out.append(
                CorpusItem(
                    content=rng.choice(emoji_pool),
                    corpus_type="emoji",
                    source_path="__emoji_injected__",
                    role="body",
                )
            )
        if corpus_type == "english":
            out.append(CorpusItem(content=" ", corpus_type="english", source_path="__inline_space__", role="body"))
    return out


def _sample_units_to_target_length(
    corpus_type: str,
    units_by_source: Dict[str, Sequence[str]],
    source_sampling_weights: Dict[str, float],
    chinese_min_length: int,
    chinese_max_length: int,
    english_min_length: int,
    english_max_length: int,
    rng: random.Random,
) -> List[str]:
    if corpus_type == "english":
        target_len = rng.randint(max(1, english_min_length), max(1, english_max_length))
        units: List[str] = []
        cur_len = 0
        guard = 0
        while cur_len < target_len and guard < 4096:
            guard += 1
            source = _pick_source_path_for_pipeline(units_by_source, source_sampling_weights, rng)
            unit = " ".join(str(rng.choice(list(units_by_source[source]))).strip().split())
            if not unit:
                continue
            add_len = len(unit) if not units else 1 + len(unit)
            if cur_len + add_len < target_len:
                units.append(unit)
                cur_len += add_len
                continue
            remaining = target_len - cur_len
            if not units:
                units.append(unit[:remaining])
            elif remaining > 1:
                units.append(unit[: remaining - 1])
            break
        return [u for u in units if u]
    target_len = rng.randint(max(1, chinese_min_length), max(1, chinese_max_length))
    units_cn: List[str] = []
    cur_cn = 0
    guard_cn = 0
    while cur_cn < target_len and guard_cn < 4096:
        guard_cn += 1
        source = _pick_source_path_for_pipeline(units_by_source, source_sampling_weights, rng)
        unit = str(rng.choice(list(units_by_source[source])))
        if not unit.strip():
            continue
        if cur_cn + len(unit) < target_len:
            units_cn.append(unit)
            cur_cn += len(unit)
            continue
        remain = target_len - cur_cn
        if remain > 0:
            units_cn.append(unit[:remain])
        break
    return [u for u in units_cn if u]


def _pick_source_path_for_pipeline(
    units_by_source: Dict[str, Sequence[str]],
    source_sampling_weights: Dict[str, float],
    rng: random.Random,
) -> str:
    source_paths = list(units_by_source.keys())
    if not source_paths:
        raise ValueError("units_by_source cannot be empty.")
    weights = [max(0.0, float(source_sampling_weights.get(source_path, 1.0))) for source_path in source_paths]
    if all(weight <= 0 for weight in weights):
        return rng.choice(source_paths)
    return rng.choices(source_paths, weights=weights, k=1)[0]


def _sanitize_title_text(text: str) -> str:
    out = text.strip()
    while out and out[-1] in TITLE_END_PUNCTUATION:
        out = out[:-1].rstrip()
    # Avoid punctuation-only titles that later become empty during layout.
    has_word = any(ch not in TITLE_END_PUNCTUATION and not ch.isspace() for ch in out)
    return out if has_word else ""


def _sample_concatenated_title(
    title_corpus_pools: Dict[str, Dict[str, Sequence[str]]],
    title_weights_by_corpus: Dict[str, Dict[str, float]],
    body_corpus_pools: Dict[str, Dict[str, Sequence[str]]],
    body_weights_by_corpus: Dict[str, Dict[str, float]],
    min_units: int,
    max_units: int,
    rng: random.Random,
) -> Tuple[str, str, str]:
    """
    Build one title string: sample min_units..max_units lines from title_corpus_pools (if configured),
    else fall back to a short snippet from body pools. English units are joined with spaces; Chinese
    concatenated. Layout truncates to one line (_truncate_text_to_single_line); overflow is dropped there.
    Returns (title_text, corpus_type, source_path).
    """
    text_types = ("chinese", "english")
    lo = max(1, min(min_units, max_units))
    hi = max(min_units, max_units)
    if title_corpus_pools:
        available = [t for t in text_types if title_corpus_pools.get(t)]
        if not available:
            return "", "", ""
        title_type = rng.choice(available)
        pools = title_corpus_pools[title_type]
        weights = title_weights_by_corpus.get(title_type, {})
        n = rng.randint(lo, hi)
        parts: List[str] = []
        primary_source = ""
        for _ in range(n):
            sp = _pick_source_path_for_pipeline(pools, weights, rng)
            if not primary_source:
                primary_source = sp
            raw = str(rng.choice(list(pools[sp]))).strip()
            if title_type == "english":
                raw = " ".join(raw.split())
            if raw:
                parts.append(raw)
        if not parts:
            return "", "", ""
        joined = " ".join(parts) if title_type == "english" else "".join(parts)
        cleaned = _sanitize_title_text(joined)
        if not cleaned:
            return "", "", ""
        src_path = primary_source if len(parts) == 1 else "__title_concat__"
        return cleaned, title_type, src_path
    chinese_pool = body_corpus_pools.get("chinese", {})
    english_pool = body_corpus_pools.get("english", {})
    title_type = "chinese" if chinese_pool else ("english" if english_pool else "chinese")
    pools = body_corpus_pools.get(title_type, {})
    if not pools:
        return "", "", ""
    weights = body_weights_by_corpus.get(title_type, {})
    for _ in range(6):
        title_source = _pick_source_path_for_pipeline(pools, weights, rng)
        title_raw = rng.choice(list(pools[title_source]))
        candidate = title_raw.strip()
        if title_type == "english":
            candidate = " ".join(candidate.split()[:6])
        else:
            candidate = candidate[:12]
        title = _sanitize_title_text(candidate)
        if title:
            return title, title_type, title_source
    return "", "", ""
