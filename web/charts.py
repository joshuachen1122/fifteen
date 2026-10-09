"""Server-rendered SVG line charts for the HTMX dashboard.

Charts are plain SVG strings so HTMX partial swaps need no client-side chart
library. Hover is handled by a delegated script in index.html that reads the
`data-points` attribute.
"""

import json
import math
from dataclasses import dataclass
from datetime import datetime
from html import escape
from typing import Callable, Sequence

WIDTH, HEIGHT = 640, 220
MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM = 72, 20, 14, 28


@dataclass(frozen=True)
class ReferenceLine:
    value: float
    label: str


def _nice_step(span: float, target_ticks: int = 4) -> float:
    raw = span / max(target_ticks, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 2.5, 5, 10):
        if raw <= multiple * magnitude:
            return multiple * magnitude
    return 10 * magnitude


def line_chart_svg(
    points: Sequence[tuple[datetime, float]],
    title: str,
    value_format: Callable[[float], str],
    reference_lines: Sequence[ReferenceLine] = (),
    chart_id: str = "chart",
) -> str:
    """Single-series line chart with optional dashed reference rules (e.g. strike K)."""
    if len(points) < 2:
        return (
            f'<div class="chart-empty" role="img" aria-label="{escape(title)}: waiting for data">'
            "Waiting for data…</div>"
        )

    times = [p[0].timestamp() for p in points]
    values = [p[1] for p in points]
    domain_values = values + [r.value for r in reference_lines]
    lo, hi = min(domain_values), max(domain_values)
    if hi == lo:
        hi, lo = hi + 1, lo - 1
    step = _nice_step(hi - lo)
    lo = math.floor(lo / step) * step
    hi = math.ceil(hi / step) * step
    t0, t1 = times[0], times[-1] if times[-1] > times[0] else times[0] + 1

    plot_w = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_h = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

    def x(t: float) -> float:
        return MARGIN_LEFT + (t - t0) / (t1 - t0) * plot_w

    def y(v: float) -> float:
        return MARGIN_TOP + (hi - v) / (hi - lo) * plot_h

    parts: list[str] = []
    # Gridlines + y labels
    tick = lo
    while tick <= hi + step / 2:
        ty = y(tick)
        parts.append(f'<line class="grid" x1="{MARGIN_LEFT}" x2="{WIDTH - MARGIN_RIGHT}" y1="{ty:.1f}" y2="{ty:.1f}"/>')
        parts.append(
            f'<text class="tick" x="{MARGIN_LEFT - 8}" y="{ty + 4:.1f}" text-anchor="end">{escape(value_format(tick))}</text>'
        )
        tick += step
    # X labels
    multi_day = points[-1][0].date() != points[0][0].date()
    fmt = "%m-%d %H:%M" if multi_day else "%H:%M"
    for i in range(4):
        tx_t = t0 + (t1 - t0) * i / 3
        anchor = "start" if i == 0 else "end" if i == 3 else "middle"
        label = datetime.fromtimestamp(tx_t, tz=points[0][0].tzinfo).strftime(fmt)
        parts.append(f'<text class="tick" x="{x(tx_t):.1f}" y="{HEIGHT - 8}" text-anchor="{anchor}">{label}</text>')
    parts.append(
        f'<line class="axis" x1="{MARGIN_LEFT}" x2="{WIDTH - MARGIN_RIGHT}" y1="{MARGIN_TOP + plot_h}" y2="{MARGIN_TOP + plot_h}"/>'
    )
    # Reference rules
    for ref in reference_lines:
        ry = y(ref.value)
        parts.append(f'<line class="ref" x1="{MARGIN_LEFT}" x2="{WIDTH - MARGIN_RIGHT}" y1="{ry:.1f}" y2="{ry:.1f}"/>')
        parts.append(
            f'<text class="ref-label" x="{WIDTH - MARGIN_RIGHT - 4}" y="{ry - 6:.1f}" text-anchor="end">{escape(ref.label)}</text>'
        )
    # Series
    coords = [(x(t), y(v)) for t, v in zip(times, values)]
    path = " ".join(f"{'M' if i == 0 else 'L'}{cx:.1f},{cy:.1f}" for i, (cx, cy) in enumerate(coords))
    parts.append(f'<path class="series" d="{path}"/>')
    end_x, end_y = coords[-1]
    parts.append(f'<circle class="series-end" cx="{end_x:.1f}" cy="{end_y:.1f}" r="4"/>')
    # Hover layer (positioned by the delegated script)
    parts.append(
        f'<line class="crosshair" x1="0" x2="0" y1="{MARGIN_TOP}" y2="{MARGIN_TOP + plot_h}" visibility="hidden"/>'
        '<circle class="hover-dot" r="5" visibility="hidden"/>'
        f'<rect class="hit" x="{MARGIN_LEFT}" y="{MARGIN_TOP}" width="{plot_w}" height="{plot_h}"/>'
    )

    hover = [
        [round(cx, 1), round(cy, 1), p[0].strftime("%Y-%m-%d %H:%M:%S"), value_format(p[1])]
        for (cx, cy), p in zip(coords, points)
    ]
    return (
        f'<div class="chart" id="{escape(chart_id)}">'
        f'<svg class="viz-line" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{escape(title)}" '
        f"data-points='{escape(json.dumps(hover), quote=True)}'>"
        + "".join(parts)
        + '</svg><div class="tooltip" hidden></div></div>'
    )


def usd(value: float) -> str:
    return f"${value:,.0f}"


def usd_cents(value: float) -> str:
    return f"${value:,.2f}"
