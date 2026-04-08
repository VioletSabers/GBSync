#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描多个目标文件夹下的 parquet，从 content_dict 中解析列表内各块的 text 字段，
统计中文汉字出现频次，并输出 Excel；再按 GB8105 字表 jsonl 中的顺序输出另一份 Excel。

运行: python scripts/check_parquet_chinese_char_frequency.py
参数在下方「配置区」修改，不使用命令行参数。
"""

from __future__ import annotations

import ast
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pyarrow.parquet as pq
from openpyxl import Workbook
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 配置区：按需修改（不使用命令行参数）
# ---------------------------------------------------------------------------

# 要扫描的根目录列表；在其下递归查找所有 .parquet（可写绝对路径或相对运行目录的路径）
TARGET_FOLDERS: List[Path] = [
    # Path("output/SimpleRendering_1024_round01/config_01_cn_color/parquet"),
]

# 可选：从文本文件额外读取目录列表（每行一个路径，UTF-8，# 开头为注释）。
# 设为 None 表示不使用文件；与 TARGET_FOLDERS 合并后去重。
FOLDER_LIST_FILE: Optional[Path] = None

# 全量汉字频次、GB8105 字表频次 的 Excel 输出路径
OUTPUT_ALL_XLSX = Path("chinese_char_frequency_all.xlsx")
OUTPUT_GB_XLSX = Path("chinese_char_frequency_gb8105.xlsx")

# 8105 通用规范汉字列表（JSONL 或每行一字）；不存在则只生成全量表并打印警告
GB8105_JSONL = Path("sample_data/chinese_gb_8105.jsonl")

# 并行进程数；0 表示自动使用 CPU 核数；1 表示单进程（便于调试）
WORKERS = 0


def _parse_content_dict(raw: Any) -> Any:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    s = raw.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return None


def _is_cjk_unified_or_ext_a(ch: str) -> bool:
    if not ch:
        return False
    o = ord(ch)
    # CJK Unified Ideographs + Extension A（常见「汉字」覆盖范围）
    return (0x4E00 <= o <= 0x9FFF) or (0x3400 <= o <= 0x4DBF)


def _count_chinese_in_text(counter: Counter[str], s: str) -> None:
    for ch in s:
        if _is_cjk_unified_or_ext_a(ch):
            counter[ch] += 1


def _extract_texts_from_content(content: Any) -> List[str]:
    if not isinstance(content, list):
        return []
    out: List[str] = []
    for block in content:
        if isinstance(block, dict):
            t = block.get("text")
            if isinstance(t, str):
                out.append(t)
    return out


def _count_one_parquet(parquet_path: str) -> Dict[str, int]:
    path = Path(parquet_path)
    counter: Counter[str] = Counter()
    try:
        table = pq.read_table(path, columns=["content_dict"])
    except Exception:
        return dict(counter)
    col = table.column(0)
    for chunk in col.chunks:
        for i in range(len(chunk)):
            raw = chunk[i].as_py()
            content = _parse_content_dict(raw)
            for text in _extract_texts_from_content(content):
                _count_chinese_in_text(counter, text)
    return dict(counter)


def _merge_counters(parts: Iterable[Dict[str, int]]) -> Counter[str]:
    total: Counter[str] = Counter()
    for d in parts:
        total.update(d)
    return total


def _collect_parquet_paths(folder_paths: Sequence[Path]) -> List[Path]:
    seen: Set[Path] = set()
    out: List[Path] = []
    for root in folder_paths:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.parquet")):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                out.append(p)
    return out


def _read_folder_list(list_file: Path) -> List[Path]:
    raw = list_file.read_text(encoding="utf-8")
    paths: List[Path] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        paths.append(Path(s).expanduser())
    return paths


def _collect_config_folder_paths() -> List[Path]:
    """合并 TARGET_FOLDERS 与 FOLDER_LIST_FILE，按出现顺序去重。"""
    raw: List[Path] = []
    for p in TARGET_FOLDERS:
        raw.append(p.expanduser())
    if FOLDER_LIST_FILE is not None and FOLDER_LIST_FILE.is_file():
        raw.extend(_read_folder_list(FOLDER_LIST_FILE))
    seen: Set[str] = set()
    out: List[Path] = []
    for p in raw:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _load_gb8105_chars(jsonl_path: Path) -> List[str]:
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    chars: List[str] = []
    for line_no, line in enumerate(lines, start=1):
        s = line.strip()
        if not s:
            continue
        ch: Optional[str] = None
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            ch = s.strip()
            if len(ch) == 1:
                chars.append(ch)
            else:
                raise ValueError(
                    f"{jsonl_path}:{line_no}: 非 JSON 行且不是单字: {line!r}"
                ) from None
            continue
        if isinstance(obj, str) and len(obj) == 1:
            ch = obj
        elif isinstance(obj, dict):
            for key in ("char", "character", "han", "字", "text", "glyph"):
                v = obj.get(key)
                if isinstance(v, str) and len(v) == 1:
                    ch = v
                    break
            if ch is None:
                raise ValueError(f"{jsonl_path}:{line_no}: 无法从对象解析单字: {line!r}")
        else:
            raise ValueError(f"{jsonl_path}:{line_no}: 不支持的 JSON 行: {line!r}")
        chars.append(ch)
    return chars


def _write_all_frequency_xlsx(path: Path, counter: Counter[str]) -> None:
    total_hits = sum(counter.values())
    rows: List[Tuple[str, int, float]] = []
    for ch, cnt in counter.most_common():
        ratio = (cnt / total_hits) if total_hits else 0.0
        rows.append((ch, cnt, ratio))
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "汉字频次"
    ws.append(["汉字", "频次", "占比(相对全部汉字计数字符)"])
    for ch, cnt, ratio in rows:
        ws.append([ch, cnt, ratio])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def _write_gb8105_xlsx(
    path: Path,
    counter: Counter[str],
    gb_chars: Sequence[str],
) -> None:
    total_hits = sum(counter.values())
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "GB8105"
    ws.append(["序号", "汉字", "频次", "占比(相对全部汉字计数字符)", "是否出现"])
    for i, ch in enumerate(gb_chars, start=1):
        cnt = counter.get(ch, 0)
        ratio = (cnt / total_hits) if total_hits else 0.0
        ws.append([i, ch, cnt, ratio, cnt > 0])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def main() -> int:
    folders = _collect_config_folder_paths()
    if not folders:
        print("错误: 未配置任何扫描目录：请设置 TARGET_FOLDERS 或有效的 FOLDER_LIST_FILE。", file=sys.stderr)
        return 1

    parquet_files = _collect_parquet_paths(folders)
    if not parquet_files:
        print("错误: 未找到任何 .parquet 文件。", file=sys.stderr)
        return 1

    str_paths = [str(p) for p in parquet_files]
    auto_workers = max(1, (mp.cpu_count() or 1))
    workers = auto_workers if WORKERS <= 0 else max(1, WORKERS)

    parts: List[Dict[str, int]] = []
    if workers == 1:
        for sp in tqdm(str_paths, desc="parquet", unit="file"):
            parts.append(_count_one_parquet(sp))
    else:
        ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for d in tqdm(
                pool.imap_unordered(_count_one_parquet, str_paths, chunksize=4),
                total=len(str_paths),
                desc="parquet",
                unit="file",
            ):
                parts.append(d)

    total_counter = _merge_counters(parts)

    _write_all_frequency_xlsx(OUTPUT_ALL_XLSX, total_counter)
    print(f"已写入全量频次: {OUTPUT_ALL_XLSX.resolve()}")

    gb_path = GB8105_JSONL
    if not gb_path.is_file():
        print(
            f"警告: 未找到 {gb_path}，跳过 GB8105 表。请放置该文件后重试。",
            file=sys.stderr,
        )
        return 0

    gb_chars = _load_gb8105_chars(gb_path)
    _write_gb8105_xlsx(OUTPUT_GB_XLSX, total_counter, gb_chars)
    print(f"已写入 GB8105 频次: {OUTPUT_GB_XLSX.resolve()}（共 {len(gb_chars)} 字）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
