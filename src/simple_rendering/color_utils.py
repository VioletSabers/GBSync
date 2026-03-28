from __future__ import annotations

from PIL import ImageColor


def contrast_ratio(color_a: str, color_b: str) -> float:
    r1, g1, b1 = ImageColor.getrgb(color_a)
    r2, g2, b2 = ImageColor.getrgb(color_b)
    l1 = _relative_luminance(r1, g1, b1)
    l2 = _relative_luminance(r2, g2, b2)
    bright = max(l1, l2)
    dark = min(l1, l2)
    return (bright + 0.05) / (dark + 0.05)


def _relative_luminance(r: int, g: int, b: int) -> float:
    rs = _srgb_to_linear(r / 255.0)
    gs = _srgb_to_linear(g / 255.0)
    bs = _srgb_to_linear(b / 255.0)
    return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs


def _srgb_to_linear(c: float) -> float:
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4
