from __future__ import annotations

import argparse
import sys
from pathlib import Path

def _inject_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simple rendered dataset.")
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
        help="Override font category in config (e.g. simple/complex).",
    )
    return parser.parse_args()


def main() -> None:
    _inject_src_to_path()
    from simple_rendering.pipeline import run_generation

    args = parse_args()
    run_generation(args.config, font_category_override=args.font_category)


if __name__ == "__main__":
    main()
