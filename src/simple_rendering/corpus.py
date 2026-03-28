from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from .config import CorpusSource, RenderConfig, resolve_config_path


@dataclass
class CorpusItem:
    content: str
    corpus_type: str
    source_path: str
    role: str = "body"


def load_corpus_items(config: RenderConfig, config_dir: Path) -> List[CorpusItem]:
    items: List[CorpusItem] = []
    for source in config.corpus_sources:
        source_path = resolve_config_path(source.path, config_dir)
        items.extend(_read_jsonl_source(source, source_path))
    if not items:
        raise ValueError("No valid corpus items loaded from configured jsonl files.")
    return items


def load_title_corpus_items(config: RenderConfig, config_dir: Path) -> List[CorpusItem]:
    sources = config.title_corpus_sources or []
    if not sources:
        return []
    items: List[CorpusItem] = []
    for source in sources:
        source_path = resolve_config_path(source.path, config_dir)
        items.extend(_read_jsonl_source(source, source_path))
    if not items:
        raise ValueError("No valid corpus items loaded from title_corpus_sources jsonl files.")
    return items


def sample_segments(
    items: Sequence[CorpusItem],
    min_segments: int,
    max_segments: int,
    rng: random.Random,
) -> List[CorpusItem]:
    if min_segments <= 0:
        raise ValueError("min_segments must be > 0")
    n = rng.randint(min_segments, max_segments)
    return [rng.choice(items) for _ in range(n)]


def build_multi_unit_text_segments(
    items: Sequence[CorpusItem],
    min_segments: int,
    max_segments: int,
    min_units_per_segment: int,
    max_units_per_segment: int,
    rng: random.Random,
) -> List[CorpusItem]:
    if min_segments <= 0:
        raise ValueError("min_segments must be > 0")
    pools: Dict[str, List[CorpusItem]] = {}
    for item in items:
        if item.corpus_type == "emoji":
            continue
        pools.setdefault(item.corpus_type, []).append(item)
    if not pools:
        raise ValueError("No non-emoji corpus items available.")

    segment_count = rng.randint(min_segments, max_segments)
    corpus_types = sorted(pools.keys())
    segments: List[CorpusItem] = []
    for _ in range(segment_count):
        corpus_type = rng.choice(corpus_types)
        unit_count = rng.randint(min_units_per_segment, max_units_per_segment)
        units = [rng.choice(pools[corpus_type]).content for _ in range(unit_count)]
        if corpus_type == "english":
            content = " ".join(units)
        else:
            content = "".join(units)
        segments.append(
            CorpusItem(
                content=content,
                corpus_type=corpus_type,
                source_path="__multi_unit_segment__",
            )
        )
    return segments


def build_inline_emoji_segments(
    items: Sequence[CorpusItem],
    emoji_candidates: Sequence[str],
    min_segments: int,
    max_segments: int,
    min_units_per_segment: int,
    max_units_per_segment: int,
    emoji_insert_probability: float,
    min_emojis_between_units: int,
    max_emojis_between_units: int,
    rng: random.Random,
) -> List[CorpusItem]:
    pools: Dict[str, List[CorpusItem]] = {}
    for item in items:
        if item.corpus_type in {"emoji", "line_break"}:
            continue
        pools.setdefault(item.corpus_type, []).append(item)
    if not pools:
        raise ValueError("No non-emoji corpus items available.")

    segment_count = rng.randint(min_segments, max_segments)
    corpus_types = sorted(pools.keys())
    output: List[CorpusItem] = []

    for seg_idx in range(segment_count):
        corpus_type = rng.choice(corpus_types)
        unit_count = rng.randint(min_units_per_segment, max_units_per_segment)
        for unit_idx in range(unit_count):
            output.append(rng.choice(pools[corpus_type]))
            if unit_idx < unit_count - 1:
                emoji_count = _sample_emoji_count_with_continue_probability(
                    emoji_insert_probability=emoji_insert_probability,
                    min_emojis_between_units=min_emojis_between_units,
                    max_emojis_between_units=max_emojis_between_units,
                    rng=rng,
                )
                if emoji_count == 0 and corpus_type == "english":
                    # Keep adjacent English units visually separable.
                    output.append(
                        CorpusItem(
                            content=" ",
                            corpus_type="english",
                            source_path="__inline_space__",
                        )
                    )
                else:
                    if corpus_type == "english":
                        output.append(
                            CorpusItem(
                                content=" ",
                                corpus_type="english",
                                source_path="__inline_space__",
                            )
                        )
                    for _ in range(emoji_count):
                        output.append(
                            CorpusItem(
                                content=rng.choice(list(emoji_candidates)),
                                corpus_type="emoji",
                                source_path="__emoji_injected__",
                            )
                        )
                    if corpus_type == "english":
                        output.append(
                            CorpusItem(
                                content=" ",
                                corpus_type="english",
                                source_path="__inline_space__",
                            )
                        )
        if seg_idx < segment_count - 1:
            output.append(
                CorpusItem(
                    content="\n",
                    corpus_type="line_break",
                    source_path="__line_break__",
                )
            )
    return output


def build_corpus_pools(items: Sequence[CorpusItem]) -> Dict[str, Dict[str, List[str]]]:
    pools: Dict[str, Dict[str, List[str]]] = {}
    for item in items:
        if item.corpus_type in {"emoji", "line_break"}:
            continue
        pools.setdefault(item.corpus_type, {}).setdefault(item.source_path, []).append(item.content)
    if not pools:
        raise ValueError("No non-emoji corpus items available.")
    return pools


def build_inline_emoji_segments_from_pools(
    corpus_pools: Dict[str, Dict[str, Sequence[str]]],
    text_corpus_types: Sequence[str],
    source_sampling_weights_by_corpus: Dict[str, Dict[str, float]],
    emoji_candidates: Sequence[str],
    min_segments: int,
    max_segments: int,
    min_units_per_segment: int,
    max_units_per_segment: int,
    emoji_insert_probability: float,
    min_emojis_between_units: int,
    max_emojis_between_units: int,
    rng: random.Random,
) -> List[CorpusItem]:
    output: List[CorpusItem] = []
    segment_count = rng.randint(min_segments, max_segments)
    corpus_types = tuple(text_corpus_types)
    emoji_pool = tuple(emoji_candidates)

    for seg_idx in range(segment_count):
        corpus_type = rng.choice(corpus_types)
        units_by_source = corpus_pools[corpus_type]
        unit_count = rng.randint(min_units_per_segment, max_units_per_segment)
        for unit_idx in range(unit_count):
            source_path = _pick_source_path(
                units_by_source=units_by_source,
                source_sampling_weights=source_sampling_weights_by_corpus.get(corpus_type, {}),
                rng=rng,
            )
            units = units_by_source[source_path]
            output.append(
                CorpusItem(
                    content=rng.choice(units),
                    corpus_type=corpus_type,
                    source_path=source_path,
                    role="body",
                )
            )
            if unit_idx < unit_count - 1:
                emoji_count = _sample_emoji_count_with_continue_probability(
                    emoji_insert_probability=emoji_insert_probability,
                    min_emojis_between_units=min_emojis_between_units,
                    max_emojis_between_units=max_emojis_between_units,
                    rng=rng,
                )
                if emoji_count == 0 and corpus_type == "english":
                    output.append(
                        CorpusItem(
                            content=" ",
                            corpus_type="english",
                            source_path="__inline_space__",
                            role="body",
                        )
                    )
                else:
                    if corpus_type == "english":
                        output.append(
                            CorpusItem(
                                content=" ",
                                corpus_type="english",
                                source_path="__inline_space__",
                                role="body",
                            )
                        )
                    for _ in range(emoji_count):
                        output.append(
                            CorpusItem(
                                content=rng.choice(emoji_pool),
                                corpus_type="emoji",
                                source_path="__emoji_injected__",
                                role="body",
                            )
                        )
                    if corpus_type == "english":
                        output.append(
                            CorpusItem(
                                content=" ",
                                corpus_type="english",
                                source_path="__inline_space__",
                                role="body",
                            )
                        )
        if seg_idx < segment_count - 1:
            output.append(
                CorpusItem(
                    content="\n",
                    corpus_type="line_break",
                    source_path="__line_break__",
                    role="line_break",
                )
            )
    return output


def build_title_subtitle_segments_from_pools(
    corpus_pools: Dict[str, Dict[str, Sequence[str]]],
    source_sampling_weights_by_corpus: Dict[str, Dict[str, float]],
    emoji_candidates: Sequence[str],
    template_cfg: Mapping[str, object],
    default_emoji_probability: float,
    default_min_emojis_between_units: int,
    default_max_emojis_between_units: int,
    rng: random.Random,
) -> List[CorpusItem]:
    title_cfg = dict(template_cfg.get("title", {})) if isinstance(template_cfg.get("title"), dict) else {}
    subtitle_cfg = (
        dict(template_cfg.get("subtitle", {})) if isinstance(template_cfg.get("subtitle"), dict) else {}
    )
    title_items = _build_role_units(
        role="title",
        role_cfg=title_cfg,
        fallback_corpus_type="chinese",
        corpus_pools=corpus_pools,
        source_sampling_weights_by_corpus=source_sampling_weights_by_corpus,
        rng=rng,
    )
    subtitle_items = _build_role_units(
        role="subtitle",
        role_cfg=subtitle_cfg,
        fallback_corpus_type="english",
        corpus_pools=corpus_pools,
        source_sampling_weights_by_corpus=source_sampling_weights_by_corpus,
        rng=rng,
    )
    title_with_emoji = _inject_inline_emojis_for_role(
        role_items=title_items,
        role="title",
        role_cfg=title_cfg,
        emoji_candidates=emoji_candidates,
        default_emoji_probability=default_emoji_probability,
        default_min_emojis_between_units=default_min_emojis_between_units,
        default_max_emojis_between_units=default_max_emojis_between_units,
        rng=rng,
    )
    subtitle_with_emoji = _inject_inline_emojis_for_role(
        role_items=subtitle_items,
        role="subtitle",
        role_cfg=subtitle_cfg,
        emoji_candidates=emoji_candidates,
        default_emoji_probability=default_emoji_probability * 0.4,
        default_min_emojis_between_units=default_min_emojis_between_units,
        default_max_emojis_between_units=default_max_emojis_between_units,
        rng=rng,
    )
    output: List[CorpusItem] = []
    output.extend(title_with_emoji)
    output.append(
        CorpusItem(content="\n", corpus_type="line_break", source_path="__line_break__", role="line_break")
    )
    output.extend(subtitle_with_emoji)
    return output


def _compose_role_text(role_items: Sequence[CorpusItem]) -> str:
    if not role_items:
        return ""
    corpus_type = role_items[0].corpus_type
    units = [item.content for item in role_items if item.content]
    if corpus_type == "english":
        return " ".join(unit.strip() for unit in units if unit.strip())
    return "".join(units)


def _sample_emoji_count_with_continue_probability(
    emoji_insert_probability: float,
    min_emojis_between_units: int,
    max_emojis_between_units: int,
    rng: random.Random,
) -> int:
    # Start from configured lower bound (often 0), then keep inserting until stop-probability
    # triggers or upper bound is reached.
    lower = max(0, min_emojis_between_units)
    upper = max(lower, max_emojis_between_units)
    emoji_count = lower
    while emoji_count < upper:
        if rng.random() >= emoji_insert_probability:
            break
        emoji_count += 1
    return emoji_count


def _pick_source_path(
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


def _build_role_units(
    role: str,
    role_cfg: Dict[str, object],
    fallback_corpus_type: str,
    corpus_pools: Dict[str, Dict[str, Sequence[str]]],
    source_sampling_weights_by_corpus: Dict[str, Dict[str, float]],
    rng: random.Random,
) -> List[CorpusItem]:
    corpus_type = str(role_cfg.get("corpus_type", fallback_corpus_type))
    units_by_source = corpus_pools.get(corpus_type)
    if not units_by_source:
        raise ValueError(f"No corpus pool found for role={role}, corpus_type={corpus_type}")
    filtered_units_by_source = _filter_sources_by_rules(units_by_source, role_cfg)
    if not filtered_units_by_source:
        raise ValueError(f"All sources filtered out for role={role}, corpus_type={corpus_type}")
    min_units = int(role_cfg.get("min_units", 2))
    max_units = int(role_cfg.get("max_units", 5))
    if min_units <= 0 or max_units < min_units:
        raise ValueError(f"Invalid min/max units in role config for role={role}")
    unit_count = rng.randint(min_units, max_units)
    role_weights = source_sampling_weights_by_corpus.get(corpus_type, {})
    items: List[CorpusItem] = []
    for _ in range(unit_count):
        source_path = _pick_source_path(
            units_by_source=filtered_units_by_source,
            source_sampling_weights=role_weights,
            rng=rng,
        )
        units = list(filtered_units_by_source[source_path])
        if not units:
            continue
        items.append(
            CorpusItem(
                content=rng.choice(units),
                corpus_type=corpus_type,
                source_path=source_path,
                role=role,
            )
        )
    if not items:
        raise ValueError(f"Failed to build units for role={role}")
    return items


def _inject_inline_emojis_for_role(
    role_items: Sequence[CorpusItem],
    role: str,
    role_cfg: Dict[str, object],
    emoji_candidates: Sequence[str],
    default_emoji_probability: float,
    default_min_emojis_between_units: int,
    default_max_emojis_between_units: int,
    rng: random.Random,
) -> List[CorpusItem]:
    if len(role_items) <= 1:
        return list(role_items)
    emoji_probability = float(role_cfg.get("emoji_insert_probability", default_emoji_probability))
    min_emojis = int(role_cfg.get("min_emojis_between_units", default_min_emojis_between_units))
    max_emojis = int(role_cfg.get("max_emojis_between_units", default_max_emojis_between_units))
    output: List[CorpusItem] = []
    for idx, item in enumerate(role_items):
        output.append(item)
        if idx >= len(role_items) - 1:
            continue
        emoji_count = _sample_emoji_count_with_continue_probability(
            emoji_insert_probability=max(0.0, min(1.0, emoji_probability)),
            min_emojis_between_units=max(0, min_emojis),
            max_emojis_between_units=max(max(0, min_emojis), max_emojis),
            rng=rng,
        )
        if emoji_count == 0 and item.corpus_type == "english":
            output.append(
                CorpusItem(
                    content=" ",
                    corpus_type="english",
                    source_path="__inline_space__",
                    role=role,
                )
            )
            continue
        if item.corpus_type == "english":
            output.append(
                CorpusItem(
                    content=" ",
                    corpus_type="english",
                    source_path="__inline_space__",
                    role=role,
                )
            )
        for _ in range(emoji_count):
            output.append(
                CorpusItem(
                    content=rng.choice(list(emoji_candidates)),
                    corpus_type="emoji",
                    source_path="__emoji_injected__",
                    role=role,
                )
            )
        if item.corpus_type == "english":
            output.append(
                CorpusItem(
                    content=" ",
                    corpus_type="english",
                    source_path="__inline_space__",
                    role=role,
                )
            )
    return output


def _filter_sources_by_rules(
    units_by_source: Dict[str, Sequence[str]],
    role_cfg: Dict[str, object],
) -> Dict[str, Sequence[str]]:
    allow_patterns = role_cfg.get("source_allow_patterns")
    deny_patterns = role_cfg.get("source_deny_patterns")
    allow = [str(p).lower() for p in allow_patterns] if isinstance(allow_patterns, list) else []
    deny = [str(p).lower() for p in deny_patterns] if isinstance(deny_patterns, list) else []
    if not allow and not deny:
        return dict(units_by_source)
    filtered: Dict[str, Sequence[str]] = {}
    for source_path, units in units_by_source.items():
        source_lower = source_path.lower()
        if allow and not any(p in source_lower for p in allow):
            continue
        if deny and any(p in source_lower for p in deny):
            continue
        filtered[source_path] = units
    return filtered


def _read_jsonl_source(source: CorpusSource, source_path: Path) -> List[CorpusItem]:
    loaded: List[CorpusItem] = []
    with source_path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = payload.get("content")
            if not isinstance(content, str):
                continue
            content = content.strip()
            if not content:
                continue
            loaded.append(
                CorpusItem(
                    content=content,
                    corpus_type=source.corpus_type,
                    source_path=str(source_path),
                )
            )
    return loaded
