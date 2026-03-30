from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class CorpusSource:
    path: str
    corpus_type: str
    sample_weight: float = 1.0


@dataclass
class CanvasConfig:
    width: int
    height: int
    min_width: Optional[int]
    max_width: Optional[int]
    min_height: Optional[int]
    max_height: Optional[int]
    short_edge_min: Optional[int]
    short_edge_max: Optional[int]
    aspect_ratios: Optional[List[str]]
    margin: int
    background_colors: List[str]
    min_text_bg_contrast_ratio: float
    background_images_dir: Optional[str]
    background_image_area_reference: int


@dataclass
class TextConfig:
    max_font_size: int
    min_font_size: int
    line_spacing: int
    paragraph_spacing: int
    min_segments_per_image: int
    max_segments_per_image: int
    min_corpus_units_per_segment: int
    max_corpus_units_per_segment: int
    chinese_min_length: int
    chinese_max_length: int
    english_min_length: int
    english_max_length: int
    emoji_insert_probability: float
    no_emoji_image_probability: float
    min_emojis_between_segments: int
    max_emojis_between_segments: int
    layout_modes: List[str]
    layout_mode_weights: Optional[Dict[str, float]]
    layout_variants_by_mode: Optional[Dict[str, List[str]]]
    default_text_colors: List[str]
    template_weights: Optional[Dict[str, float]]
    style_templates: Optional[Dict[str, dict]]
    dual_column_inner_margin_at_1024: int
    dual_column_write_width_ratio_min: float
    dual_column_write_width_ratio_max: float
    title_corpus_units_min: int
    title_corpus_units_max: int
    # Per StyledSegment: random max visual lines before truncating remainder (layout efficiency).
    min_lines_cap_per_segment: int
    max_lines_cap_per_segment: int


@dataclass
class OutputConfig:
    root_dir: str
    image_dir: str
    parquet_dir: str


@dataclass
class ArtAssetsConfig:
    brush_textures_dir: str
    paper_textures_dir: Optional[str]
    feibai_strength_range: List[float]
    ink_bleed_radius_range: List[float]
    edge_damage_strength_range: List[float]
    stroke_direction_jitter: float


@dataclass
class RenderConfig:
    seed: Optional[int]
    parallel_workers: Optional[int]
    font_category: str
    num_rounds: int
    samples_per_round: int
    canvas: CanvasConfig
    text: TextConfig
    output: OutputConfig
    corpus_sources: List[CorpusSource]
    fonts_by_corpus: Dict[str, List[str]]
    colors_by_corpus: Dict[str, List[str]]
    art_assets: Optional[ArtAssetsConfig] = None
    title_corpus_sources: Optional[List[CorpusSource]] = None
    # Caption template file path (e.g. "templates/caption_templates_L1.json").
    # Relative paths are resolved relative to the config YAML directory.
    caption_templates_L1_path: Optional[str] = None
    # Caption L2 template file path (e.g. "templates/caption_templates_L2.json").
    # Relative paths are resolved relative to the config YAML directory.
    caption_templates_L2_path: Optional[str] = None
    # Caption L3 template file path (e.g. "templates/caption_templates_L3.json").
    # Relative paths are resolved relative to the config YAML directory.
    caption_templates_L3_path: Optional[str] = None


def _require_keys(payload: dict, keys: List[str], scope: str) -> None:
    missing = [k for k in keys if k not in payload]
    if missing:
        raise ValueError(f"Missing keys in {scope}: {missing}")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def load_config(config_path: str, font_category_override: Optional[str] = None) -> RenderConfig:
    path = Path(config_path).expanduser().resolve()
    raw = _load_yaml(path)

    _require_keys(
        raw,
        [
            "num_rounds",
            "samples_per_round",
            "canvas",
            "text",
            "output",
            "corpus_sources",
            "colors_by_corpus",
        ],
        "root",
    )
    if "fonts_by_corpus" not in raw and "fonts_by_corpus_by_category" not in raw:
        raise ValueError(
            "One of 'fonts_by_corpus' or 'fonts_by_corpus_by_category' must be provided."
        )

    canvas_raw = raw["canvas"]
    text_raw = raw["text"]
    output_raw = raw["output"]
    shared_palette_hexes = _load_shared_palette_hexes(raw, base_dir=path.parent)

    _require_keys(canvas_raw, ["margin", "min_text_bg_contrast_ratio"], "canvas")
    if "background_colors" not in canvas_raw and "background_images_dir" not in canvas_raw:
        raise ValueError("canvas requires at least one of background_colors or background_images_dir.")
    _require_keys(
        text_raw,
        [
            "min_font_size",
            "line_spacing",
            "paragraph_spacing",
            "min_segments_per_image",
            "max_segments_per_image",
            "min_corpus_units_per_segment",
            "max_corpus_units_per_segment",
            "emoji_insert_probability",
            "min_emojis_between_segments",
            "max_emojis_between_segments",
            "layout_modes",
            "default_text_colors",
        ],
        "text",
    )
    _require_keys(output_raw, ["root_dir", "image_dir", "parquet_dir"], "output")

    sources: List[CorpusSource] = []
    for i, source in enumerate(raw["corpus_sources"]):
        if not isinstance(source, dict):
            raise ValueError(f"corpus_sources[{i}] must be a mapping.")
        _require_keys(source, ["path", "corpus_type"], f"corpus_sources[{i}]")
        weight = float(source.get("sample_weight", 1.0))
        if weight < 0:
            raise ValueError(f"corpus_sources[{i}].sample_weight must be >= 0.")
        sources.append(
            CorpusSource(
                path=source["path"],
                corpus_type=source["corpus_type"],
                sample_weight=weight,
            )
        )

    title_sources: Optional[List[CorpusSource]] = None
    title_raw_list = raw.get("title_corpus_sources")
    if title_raw_list is not None:
        if not isinstance(title_raw_list, list):
            raise ValueError("title_corpus_sources must be a list when provided.")
        parsed_title: List[CorpusSource] = []
        for i, source in enumerate(title_raw_list):
            if not isinstance(source, dict):
                raise ValueError(f"title_corpus_sources[{i}] must be a mapping.")
            _require_keys(source, ["path", "corpus_type"], f"title_corpus_sources[{i}]")
            weight = float(source.get("sample_weight", 1.0))
            if weight < 0:
                raise ValueError(f"title_corpus_sources[{i}].sample_weight must be >= 0.")
            parsed_title.append(
                CorpusSource(
                    path=source["path"],
                    corpus_type=source["corpus_type"],
                    sample_weight=weight,
                )
            )
        title_sources = parsed_title or None

    caption_templates_L1_path_raw = raw.get("caption_templates_L1_path")
    caption_templates_L1_path: Optional[str]
    if caption_templates_L1_path_raw is None:
        caption_templates_L1_path = None
    else:
        if not isinstance(caption_templates_L1_path_raw, str):
            raise ValueError("caption_templates_L1_path must be a string when provided.")
        caption_templates_L1_path = caption_templates_L1_path_raw.strip()
        if not caption_templates_L1_path:
            raise ValueError("caption_templates_L1_path cannot be empty when provided.")

    caption_templates_L2_path_raw = raw.get("caption_templates_L2_path")
    caption_templates_L2_path: Optional[str]
    if caption_templates_L2_path_raw is None:
        caption_templates_L2_path = None
    else:
        if not isinstance(caption_templates_L2_path_raw, str):
            raise ValueError("caption_templates_L2_path must be a string when provided.")
        caption_templates_L2_path = caption_templates_L2_path_raw.strip()
        if not caption_templates_L2_path:
            raise ValueError("caption_templates_L2_path cannot be empty when provided.")

    caption_templates_L3_path_raw = raw.get("caption_templates_L3_path")
    caption_templates_L3_path: Optional[str]
    if caption_templates_L3_path_raw is None:
        caption_templates_L3_path = None
    else:
        if not isinstance(caption_templates_L3_path_raw, str):
            raise ValueError("caption_templates_L3_path must be a string when provided.")
        caption_templates_L3_path = caption_templates_L3_path_raw.strip()
        if not caption_templates_L3_path:
            raise ValueError("caption_templates_L3_path cannot be empty when provided.")

    if text_raw["min_segments_per_image"] > text_raw["max_segments_per_image"]:
        raise ValueError("text.min_segments_per_image must be <= text.max_segments_per_image")
    if text_raw["min_corpus_units_per_segment"] > text_raw["max_corpus_units_per_segment"]:
        raise ValueError(
            "text.min_corpus_units_per_segment must be <= text.max_corpus_units_per_segment"
        )
    chinese_min_length = int(
        text_raw.get(
            "chinese_min_length",
            max(4, int(text_raw["min_corpus_units_per_segment"]) * 6),
        )
    )
    chinese_max_length = int(
        text_raw.get(
            "chinese_max_length",
            max(chinese_min_length, int(text_raw["max_corpus_units_per_segment"]) * 16),
        )
    )
    english_min_length = int(
        text_raw.get(
            "english_min_length",
            max(12, int(text_raw["min_corpus_units_per_segment"]) * 24),
        )
    )
    english_max_length = int(
        text_raw.get(
            "english_max_length",
            max(english_min_length, int(text_raw["max_corpus_units_per_segment"]) * 64),
        )
    )
    if chinese_min_length <= 0 or chinese_max_length <= 0:
        raise ValueError("text.chinese_min_length and text.chinese_max_length must be > 0")
    if english_min_length <= 0 or english_max_length <= 0:
        raise ValueError("text.english_min_length and text.english_max_length must be > 0")
    if chinese_min_length > chinese_max_length:
        raise ValueError("text.chinese_min_length must be <= text.chinese_max_length")
    if english_min_length > english_max_length:
        raise ValueError("text.english_min_length must be <= text.english_max_length")
    if not (0.0 <= float(text_raw["emoji_insert_probability"]) <= 1.0):
        raise ValueError("text.emoji_insert_probability must be in [0.0, 1.0].")
    if not (0.0 <= float(text_raw.get("no_emoji_image_probability", 0.0)) <= 1.0):
        raise ValueError("text.no_emoji_image_probability must be in [0.0, 1.0].")
    title_corpus_units_min = int(text_raw.get("title_corpus_units_min", 1))
    title_corpus_units_max = int(text_raw.get("title_corpus_units_max", 5))
    if title_corpus_units_min < 1 or title_corpus_units_max < 1:
        raise ValueError("text.title_corpus_units_min and title_corpus_units_max must be >= 1.")
    if title_corpus_units_min > title_corpus_units_max:
        raise ValueError("text.title_corpus_units_min must be <= text.title_corpus_units_max.")
    min_lines_cap = int(text_raw.get("min_lines_cap_per_segment", 1))
    max_lines_cap = int(text_raw.get("max_lines_cap_per_segment", 256))
    if min_lines_cap < 1 or max_lines_cap < 1:
        raise ValueError("text.min_lines_cap_per_segment and max_lines_cap_per_segment must be >= 1.")
    if min_lines_cap > max_lines_cap:
        raise ValueError("text.min_lines_cap_per_segment must be <= text.max_lines_cap_per_segment.")
    if text_raw["min_emojis_between_segments"] > text_raw["max_emojis_between_segments"]:
        raise ValueError(
            "text.min_emojis_between_segments must be <= text.max_emojis_between_segments"
        )
    max_font_size = text_raw.get("max_font_size", text_raw.get("default_font_size"))
    if max_font_size is None:
        raise ValueError("text.max_font_size is required (or legacy text.default_font_size).")
    if text_raw["min_font_size"] > max_font_size:
        raise ValueError("text.min_font_size must be <= text.max_font_size")
    parallel_workers = raw.get("parallel_workers")
    if parallel_workers is not None and int(parallel_workers) <= 0:
        raise ValueError("parallel_workers must be > 0 when provided.")

    selected_font_category = font_category_override or raw.get("font_category") or "simple"
    resolved_fonts_by_corpus = _resolve_fonts_by_category(raw, selected_font_category)

    fixed_width_raw = canvas_raw.get("width")
    fixed_height_raw = canvas_raw.get("height")
    min_width_raw = canvas_raw.get("min_width")
    max_width_raw = canvas_raw.get("max_width")
    min_height_raw = canvas_raw.get("min_height")
    max_height_raw = canvas_raw.get("max_height")

    has_fixed = fixed_width_raw is not None and fixed_height_raw is not None
    has_range = (
        min_width_raw is not None
        and max_width_raw is not None
        and min_height_raw is not None
        and max_height_raw is not None
    )
    if not has_fixed and not has_range:
        raise ValueError(
            "canvas requires either width/height or min_width/max_width/min_height/max_height."
        )

    fixed_width = int(fixed_width_raw) if fixed_width_raw is not None else None
    fixed_height = int(fixed_height_raw) if fixed_height_raw is not None else None
    min_width = int(min_width_raw) if min_width_raw is not None else None
    max_width = int(max_width_raw) if max_width_raw is not None else None
    min_height = int(min_height_raw) if min_height_raw is not None else None
    max_height = int(max_height_raw) if max_height_raw is not None else None
    short_edge_min_raw = canvas_raw.get("short_edge_min")
    short_edge_max_raw = canvas_raw.get("short_edge_max")
    short_edge_min = int(short_edge_min_raw) if short_edge_min_raw is not None else None
    short_edge_max = int(short_edge_max_raw) if short_edge_max_raw is not None else None
    aspect_ratios = [str(v) for v in canvas_raw.get("aspect_ratios", [])] if isinstance(canvas_raw.get("aspect_ratios"), list) else None

    config = RenderConfig(
        seed=raw.get("seed"),
        parallel_workers=int(parallel_workers) if parallel_workers is not None else None,
        font_category=selected_font_category,
        num_rounds=int(raw["num_rounds"]),
        samples_per_round=int(raw["samples_per_round"]),
        canvas=CanvasConfig(
            width=fixed_width if fixed_width is not None else int(min_width),  # type: ignore[arg-type]
            height=fixed_height if fixed_height is not None else int(min_height),  # type: ignore[arg-type]
            min_width=min_width,
            max_width=max_width,
            min_height=min_height,
            max_height=max_height,
            short_edge_min=short_edge_min,
            short_edge_max=short_edge_max,
            aspect_ratios=aspect_ratios,
            margin=int(canvas_raw["margin"]),
            background_colors=shared_palette_hexes or list(canvas_raw.get("background_colors", [])),
            min_text_bg_contrast_ratio=float(canvas_raw["min_text_bg_contrast_ratio"]),
            background_images_dir=(
                str(canvas_raw.get("background_images_dir"))
                if canvas_raw.get("background_images_dir")
                else None
            ),
            background_image_area_reference=int(canvas_raw.get("background_image_area_reference", 1024)),
        ),
        text=TextConfig(
            max_font_size=int(max_font_size),
            min_font_size=int(text_raw["min_font_size"]),
            line_spacing=int(text_raw["line_spacing"]),
            paragraph_spacing=int(text_raw["paragraph_spacing"]),
            min_segments_per_image=int(text_raw["min_segments_per_image"]),
            max_segments_per_image=int(text_raw["max_segments_per_image"]),
            min_corpus_units_per_segment=int(text_raw["min_corpus_units_per_segment"]),
            max_corpus_units_per_segment=int(text_raw["max_corpus_units_per_segment"]),
            chinese_min_length=chinese_min_length,
            chinese_max_length=chinese_max_length,
            english_min_length=english_min_length,
            english_max_length=english_max_length,
            emoji_insert_probability=float(text_raw["emoji_insert_probability"]),
            no_emoji_image_probability=float(text_raw.get("no_emoji_image_probability", 0.0)),
            min_emojis_between_segments=int(text_raw["min_emojis_between_segments"]),
            max_emojis_between_segments=int(text_raw["max_emojis_between_segments"]),
            layout_modes=list(text_raw["layout_modes"]),
            layout_mode_weights=_resolve_layout_mode_weights(
                text_raw.get("layout_mode_weights"), list(text_raw["layout_modes"])
            ),
            layout_variants_by_mode=_resolve_layout_variants_by_mode(
                text_raw.get("layout_variants_by_mode")
            ),
            default_text_colors=shared_palette_hexes or list(text_raw["default_text_colors"]),
            template_weights=_resolve_template_weights(text_raw.get("template_weights")),
            style_templates=_resolve_style_templates(text_raw.get("style_templates")),
            dual_column_inner_margin_at_1024=int(
                text_raw.get("dual_column_inner_margin_at_1024", 48)
            ),
            dual_column_write_width_ratio_min=float(
                text_raw.get("dual_column_write_width_ratio_min", 0.70)
            ),
            dual_column_write_width_ratio_max=float(
                text_raw.get("dual_column_write_width_ratio_max", 0.95)
            ),
            title_corpus_units_min=title_corpus_units_min,
            title_corpus_units_max=title_corpus_units_max,
            min_lines_cap_per_segment=min_lines_cap,
            max_lines_cap_per_segment=max_lines_cap,
        ),
        output=OutputConfig(
            root_dir=output_raw["root_dir"],
            image_dir=output_raw["image_dir"],
            parquet_dir=output_raw["parquet_dir"],
        ),
        corpus_sources=sources,
        fonts_by_corpus=resolved_fonts_by_corpus,
        colors_by_corpus=(
            {k: list(shared_palette_hexes) for k in raw["colors_by_corpus"].keys()}
            if shared_palette_hexes
            else {k: list(v) for k, v in raw["colors_by_corpus"].items()}
        ),
        art_assets=_resolve_art_assets(raw.get("art_assets")),
        title_corpus_sources=title_sources,
        caption_templates_L1_path=caption_templates_L1_path,
        caption_templates_L2_path=caption_templates_L2_path,
        caption_templates_L3_path=caption_templates_L3_path,
    )
    _validate_references(config, base_dir=path.parent)
    return config


def _resolve_fonts_by_category(raw: dict, selected_font_category: str) -> Dict[str, List[str]]:
    if "fonts_by_corpus_by_category" in raw:
        by_category = raw["fonts_by_corpus_by_category"]
        if not isinstance(by_category, dict):
            raise ValueError("fonts_by_corpus_by_category must be a mapping.")
        if selected_font_category not in by_category:
            raise ValueError(
                f"Unknown font_category='{selected_font_category}'. "
                f"Available categories: {sorted(by_category.keys())}"
            )
        selected = by_category[selected_font_category]
        if not isinstance(selected, dict):
            raise ValueError(
                f"fonts_by_corpus_by_category['{selected_font_category}'] must be a mapping."
            )
        return {k: list(v) for k, v in selected.items()}

    # Backward compatibility: no category map defined.
    return {k: list(v) for k, v in raw["fonts_by_corpus"].items()}


def _validate_references(config: RenderConfig, base_dir: Path) -> None:
    if config.canvas.width <= 0 or config.canvas.height <= 0:
        raise ValueError("canvas.width and canvas.height must be > 0.")
    if config.canvas.min_width is not None or config.canvas.max_width is not None:
        if config.canvas.min_width is None or config.canvas.max_width is None:
            raise ValueError("canvas.min_width and canvas.max_width must be both provided.")
        if config.canvas.min_width <= 0 or config.canvas.max_width <= 0:
            raise ValueError("canvas min/max width must be > 0.")
        if config.canvas.min_width > config.canvas.max_width:
            raise ValueError("canvas.min_width must be <= canvas.max_width.")
    if config.canvas.min_height is not None or config.canvas.max_height is not None:
        if config.canvas.min_height is None or config.canvas.max_height is None:
            raise ValueError("canvas.min_height and canvas.max_height must be both provided.")
        if config.canvas.min_height <= 0 or config.canvas.max_height <= 0:
            raise ValueError("canvas min/max height must be > 0.")
        if config.canvas.min_height > config.canvas.max_height:
            raise ValueError("canvas.min_height must be <= canvas.max_height.")
    if config.canvas.short_edge_min is not None or config.canvas.short_edge_max is not None:
        if config.canvas.short_edge_min is None or config.canvas.short_edge_max is None:
            raise ValueError("canvas.short_edge_min and canvas.short_edge_max must be both provided.")
        if config.canvas.short_edge_min <= 0 or config.canvas.short_edge_max <= 0:
            raise ValueError("canvas.short_edge_min and canvas.short_edge_max must be > 0.")
        if config.canvas.short_edge_min > config.canvas.short_edge_max:
            raise ValueError("canvas.short_edge_min must be <= canvas.short_edge_max.")
    if config.canvas.aspect_ratios is not None:
        if not config.canvas.aspect_ratios:
            raise ValueError("canvas.aspect_ratios cannot be empty when provided.")
        for ratio in config.canvas.aspect_ratios:
            if ":" not in ratio:
                raise ValueError(f"Invalid canvas.aspect_ratios entry: {ratio}")
            left, right = ratio.split(":", 1)
            try:
                rw = int(left.strip())
                rh = int(right.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid canvas.aspect_ratios entry: {ratio}") from exc
            if rw <= 0 or rh <= 0:
                raise ValueError(f"Invalid canvas.aspect_ratios entry: {ratio}")

    supported_layout_modes = {
        "mixed_line",
        "segmented",
        "vertical",
        "title_subtitle",
        "full_text",
        "title_body",
        "dual_column",
    }
    if not config.text.layout_modes:
        raise ValueError("text.layout_modes cannot be empty.")
    invalid_modes = [m for m in config.text.layout_modes if m not in supported_layout_modes]
    if invalid_modes:
        raise ValueError(f"Unsupported layout modes: {invalid_modes}")
    if config.text.layout_mode_weights is not None:
        unknown_weight_modes = [
            mode for mode in config.text.layout_mode_weights if mode not in supported_layout_modes
        ]
        if unknown_weight_modes:
            raise ValueError(f"Unsupported layout mode weights: {unknown_weight_modes}")
        missing = [mode for mode in config.text.layout_modes if mode not in config.text.layout_mode_weights]
        if missing:
            raise ValueError(f"layout_mode_weights missing modes configured in layout_modes: {missing}")
        has_positive = any(weight > 0 for weight in config.text.layout_mode_weights.values())
        if not has_positive:
            raise ValueError("layout_mode_weights must contain at least one positive weight.")
    if config.text.layout_variants_by_mode is not None:
        for mode, variants in config.text.layout_variants_by_mode.items():
            if mode not in supported_layout_modes:
                raise ValueError(f"Unsupported layout mode in layout_variants_by_mode: {mode}")
            if not isinstance(variants, list) or not variants:
                raise ValueError(f"layout_variants_by_mode[{mode}] must be a non-empty list.")
            for variant in variants:
                if not isinstance(variant, str) or not variant.strip():
                    raise ValueError(f"layout_variants_by_mode[{mode}] contains invalid variant.")
    if config.text.dual_column_inner_margin_at_1024 < 0:
        raise ValueError("text.dual_column_inner_margin_at_1024 must be >= 0.")
    if not (
        0.1 <= config.text.dual_column_write_width_ratio_min <= 1.0
        and 0.1 <= config.text.dual_column_write_width_ratio_max <= 1.0
    ):
        raise ValueError(
            "text.dual_column_write_width_ratio_min/max must be in [0.1, 1.0]."
        )
    if (
        config.text.dual_column_write_width_ratio_min
        > config.text.dual_column_write_width_ratio_max
    ):
        raise ValueError(
            "text.dual_column_write_width_ratio_min must be <= dual_column_write_width_ratio_max."
        )
    if config.text.template_weights is not None:
        has_positive = any(weight > 0 for weight in config.text.template_weights.values())
        if not has_positive:
            raise ValueError("text.template_weights must contain at least one positive weight.")
    if config.text.style_templates is not None:
        for template_name, template_cfg in config.text.style_templates.items():
            if not isinstance(template_cfg, dict):
                raise ValueError(f"text.style_templates.{template_name} must be a mapping.")
            if template_name == "title_subtitle":
                for role in ("title", "subtitle"):
                    role_cfg = template_cfg.get(role)
                    if not isinstance(role_cfg, dict):
                        raise ValueError(f"text.style_templates.{template_name}.{role} must be a mapping.")
                    if "corpus_type" not in role_cfg:
                        raise ValueError(
                            f"text.style_templates.{template_name}.{role}.corpus_type is required."
                        )
        if config.text.template_weights:
            missing_templates = [
                template_name
                for template_name in config.text.template_weights
                if template_name not in config.text.style_templates
            ]
            if missing_templates:
                raise ValueError(
                    f"text.template_weights references undefined templates: {missing_templates}"
                )
    if config.canvas.min_text_bg_contrast_ratio < 1.0 or config.canvas.min_text_bg_contrast_ratio > 21.0:
        raise ValueError("canvas.min_text_bg_contrast_ratio must be in [1.0, 21.0].")
    if config.canvas.background_image_area_reference <= 0:
        raise ValueError("canvas.background_image_area_reference must be > 0.")
    if not config.canvas.background_colors and not config.canvas.background_images_dir:
        raise ValueError("canvas.background_colors and canvas.background_images_dir cannot both be empty.")
    if config.canvas.background_images_dir:
        bg_dir = _resolve_with_base(config.canvas.background_images_dir, base_dir)
        if not bg_dir.exists() or not bg_dir.is_dir():
            raise ValueError(f"canvas.background_images_dir must be an existing directory: {bg_dir}")

    for source in config.corpus_sources:
        source_path = _resolve_with_base(source.path, base_dir)
        if not source_path.exists():
            raise ValueError(f"Corpus source file does not exist: {source_path}")
        if source.corpus_type not in config.fonts_by_corpus:
            raise ValueError(f"No fonts configured for corpus_type={source.corpus_type}")
    source_weights_by_corpus: Dict[str, List[float]] = {}
    for source in config.corpus_sources:
        source_weights_by_corpus.setdefault(source.corpus_type, []).append(source.sample_weight)
    for corpus_type, weights in source_weights_by_corpus.items():
        if all(weight <= 0 for weight in weights):
            raise ValueError(
                f"corpus_sources for corpus_type={corpus_type} must have at least one positive sample_weight."
            )

    if config.title_corpus_sources:
        for source in config.title_corpus_sources:
            title_path = _resolve_with_base(source.path, base_dir)
            if not title_path.exists():
                raise ValueError(f"title_corpus_sources file does not exist: {title_path}")
            if source.corpus_type not in config.fonts_by_corpus:
                raise ValueError(f"No fonts configured for title corpus_type={source.corpus_type}")
        title_weights_by_corpus: Dict[str, List[float]] = {}
        for source in config.title_corpus_sources:
            title_weights_by_corpus.setdefault(source.corpus_type, []).append(source.sample_weight)
        for corpus_type, weights in title_weights_by_corpus.items():
            if all(weight <= 0 for weight in weights):
                raise ValueError(
                    f"title_corpus_sources for corpus_type={corpus_type} must have at least one positive sample_weight."
                )

    for corpus_type, font_paths in config.fonts_by_corpus.items():
        if not font_paths:
            raise ValueError(f"fonts_by_corpus[{corpus_type}] cannot be empty.")
        for font_path in font_paths:
            resolved = _resolve_with_base(font_path, base_dir)
            if not resolved.exists():
                raise ValueError(f"Font path does not exist: {resolved}")
    if config.art_assets is not None:
        brush_dir = _resolve_with_base(config.art_assets.brush_textures_dir, base_dir)
        if not brush_dir.exists() or not brush_dir.is_dir():
            raise ValueError(f"art_assets.brush_textures_dir must be an existing directory: {brush_dir}")
        if not any(path.is_file() for path in brush_dir.iterdir()):
            raise ValueError(f"art_assets.brush_textures_dir has no files: {brush_dir}")
        if config.art_assets.paper_textures_dir:
            paper_dir = _resolve_with_base(config.art_assets.paper_textures_dir, base_dir)
            if not paper_dir.exists() or not paper_dir.is_dir():
                raise ValueError(
                    f"art_assets.paper_textures_dir must be an existing directory: {paper_dir}"
                )
            if not any(path.is_file() for path in paper_dir.iterdir()):
                raise ValueError(f"art_assets.paper_textures_dir has no files: {paper_dir}")


def _resolve_with_base(path: str, base_dir: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()


def _load_shared_palette_hexes(raw: dict, base_dir: Path) -> Optional[List[str]]:
    palette_path = raw.get("shared_color_palette_json")
    if not palette_path:
        return None
    resolved = _resolve_with_base(str(palette_path), base_dir)
    if not resolved.exists():
        raise ValueError(f"shared_color_palette_json does not exist: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse shared_color_palette_json: {resolved}") from exc
    colors = payload.get("colors")
    if not isinstance(colors, list) or not colors:
        raise ValueError("shared_color_palette_json.colors must be a non-empty list.")
    hexes: List[str] = []
    for idx, item in enumerate(colors):
        if not isinstance(item, dict):
            raise ValueError(f"shared_color_palette_json.colors[{idx}] must be a mapping.")
        hex_color = item.get("hex")
        if not isinstance(hex_color, str) or not hex_color.strip():
            raise ValueError(f"shared_color_palette_json.colors[{idx}].hex is required.")
        hexes.append(hex_color.strip().upper())
    return hexes


def resolve_config_path(path: str, base_dir: Path) -> Path:
    return _resolve_with_base(path, base_dir)


def _resolve_layout_mode_weights(
    raw_weights: Optional[dict], layout_modes: List[str]
) -> Optional[Dict[str, float]]:
    if raw_weights is None:
        return None
    if not isinstance(raw_weights, dict):
        raise ValueError("text.layout_mode_weights must be a mapping when provided.")
    resolved: Dict[str, float] = {}
    for mode, weight in raw_weights.items():
        value = float(weight)
        if value < 0:
            raise ValueError("text.layout_mode_weights values must be >= 0.")
        resolved[str(mode)] = value
    for mode in layout_modes:
        if mode not in resolved:
            raise ValueError(f"text.layout_mode_weights missing mode: {mode}")
    return resolved


def _resolve_layout_variants_by_mode(raw: Optional[dict]) -> Optional[Dict[str, List[str]]]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("text.layout_variants_by_mode must be a mapping when provided.")
    resolved: Dict[str, List[str]] = {}
    for mode, variants in raw.items():
        if not isinstance(variants, list) or not variants:
            raise ValueError(f"text.layout_variants_by_mode.{mode} must be a non-empty list.")
        rows: List[str] = []
        for variant in variants:
            value = str(variant).strip()
            if not value:
                raise ValueError(f"text.layout_variants_by_mode.{mode} has empty variant.")
            rows.append(value)
        resolved[str(mode)] = rows
    return resolved


def _resolve_template_weights(raw_weights: Optional[dict]) -> Optional[Dict[str, float]]:
    if raw_weights is None:
        return None
    if not isinstance(raw_weights, dict):
        raise ValueError("text.template_weights must be a mapping when provided.")
    resolved: Dict[str, float] = {}
    for template_name, weight in raw_weights.items():
        value = float(weight)
        if value < 0:
            raise ValueError("text.template_weights values must be >= 0.")
        resolved[str(template_name)] = value
    return resolved


def _resolve_style_templates(raw_templates: Optional[dict]) -> Optional[Dict[str, dict]]:
    if raw_templates is None:
        return None
    if not isinstance(raw_templates, dict):
        raise ValueError("text.style_templates must be a mapping when provided.")
    resolved: Dict[str, dict] = {}
    for template_name, template_cfg in raw_templates.items():
        if not isinstance(template_cfg, dict):
            raise ValueError(f"text.style_templates.{template_name} must be a mapping.")
        resolved[str(template_name)] = dict(template_cfg)
    return resolved


def _resolve_art_assets(raw_assets: Optional[dict]) -> Optional[ArtAssetsConfig]:
    if raw_assets is None:
        return None
    if not isinstance(raw_assets, dict):
        raise ValueError("art_assets must be a mapping when provided.")
    _require_keys(
        raw_assets,
        [
            "brush_textures_dir",
            "feibai_strength_range",
            "ink_bleed_radius_range",
            "edge_damage_strength_range",
            "stroke_direction_jitter",
        ],
        "art_assets",
    )
    feibai = _resolve_numeric_range(raw_assets["feibai_strength_range"], "art_assets.feibai_strength_range")
    bleed = _resolve_numeric_range(raw_assets["ink_bleed_radius_range"], "art_assets.ink_bleed_radius_range")
    edge = _resolve_numeric_range(
        raw_assets["edge_damage_strength_range"], "art_assets.edge_damage_strength_range"
    )
    jitter = float(raw_assets["stroke_direction_jitter"])
    if jitter < 0:
        raise ValueError("art_assets.stroke_direction_jitter must be >= 0.")
    return ArtAssetsConfig(
        brush_textures_dir=str(raw_assets["brush_textures_dir"]),
        paper_textures_dir=(
            str(raw_assets["paper_textures_dir"]) if raw_assets.get("paper_textures_dir") else None
        ),
        feibai_strength_range=feibai,
        ink_bleed_radius_range=bleed,
        edge_damage_strength_range=edge,
        stroke_direction_jitter=jitter,
    )


def _resolve_numeric_range(raw_value, scope: str) -> List[float]:
    if not isinstance(raw_value, list) or len(raw_value) != 2:
        raise ValueError(f"{scope} must be a list of two numbers.")
    left = float(raw_value[0])
    right = float(raw_value[1])
    if left < 0 or right < 0:
        raise ValueError(f"{scope} values must be >= 0.")
    if left > right:
        raise ValueError(f"{scope}[0] must be <= {scope}[1].")
    return [left, right]


