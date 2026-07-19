"""Render editable training-reward data as an SVG comparison chart."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT / "docs" / "results" / "official_airline_reward_trajectories.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "docs" / "assets" / "official_airline_reward_trajectories.svg"
)

WIDTH = 1160
HEIGHT = 640
PLOT_LEFT = 92
PLOT_RIGHT = 1122
PLOT_TOP = 126
PLOT_BOTTOM = 548


@dataclass(frozen=True, slots=True)
class Series:
    key: str
    label: str
    color: str
    values: tuple[tuple[int, float], ...]


SERIES_STYLE = (
    ("e1", "E1: sequence GRPO", "#6D55B5"),
    ("e2", "E2: balanced GRPO", "#D85C85"),
    ("e5", "E5: balanced + process + hindsight", "#A35C3A"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot E1, E2, and merged E5 reward trajectories from CSV."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load_series(path: Path) -> tuple[Series, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"step", *(key for key, _, _ in SERIES_STYLE)}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing CSV columns: {sorted(missing)}")

        values: dict[str, list[tuple[int, float]]] = {
            key: [] for key, _, _ in SERIES_STYLE
        }
        previous_step = 0
        for row in reader:
            step = int(row["step"])
            if step <= previous_step:
                raise ValueError("CSV steps must be strictly increasing")
            previous_step = step

            for key in values:
                raw = (row.get(key) or "").strip()
                if not raw:
                    continue
                reward = float(raw)
                if not math.isfinite(reward):
                    raise ValueError(f"non-finite {key} reward at step {step}")
                values[key].append((step, reward))

    result = tuple(
        Series(key=key, label=label, color=color, values=tuple(values[key]))
        for key, label, color in SERIES_STYLE
    )
    if any(not series.values for series in result):
        raise ValueError("every configured experiment must contain at least one value")
    return result


def _axis_bounds(series: tuple[Series, ...]) -> tuple[int, int, float, float]:
    points = [point for item in series for point in item.values]
    x_min = min(step for step, _ in points)
    x_max = max(step for step, _ in points)
    raw_y_min = min(value for _, value in points)
    raw_y_max = max(value for _, value in points)
    tick = 0.05
    y_min = max(0.0, math.floor((raw_y_min - 0.01) / tick) * tick)
    y_max = math.ceil((raw_y_max + 0.01) / tick) * tick
    return x_min, x_max, y_min, y_max


def _render_svg(series: tuple[Series, ...]) -> str:
    x_min, x_max, y_min, y_max = _axis_bounds(series)
    plot_width = PLOT_RIGHT - PLOT_LEFT
    plot_height = PLOT_BOTTOM - PLOT_TOP

    def x_pos(step: int) -> float:
        return PLOT_LEFT + (step - x_min) / (x_max - x_min) * plot_width

    def y_pos(value: float) -> float:
        return PLOT_BOTTOM - (value - y_min) / (y_max - y_min) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        'aria-labelledby="title description">',
        '<title id="title">Training reward trajectories on official tau2 airline tasks</title>',
        '<desc id="description">Mean rollout reward by global training step for E1, E2, and E5.</desc>',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>',
        'text { font-family: Inter, "Segoe UI", Arial, sans-serif; fill: #20242C; letter-spacing: 0; }',
        '.title { font-size: 27px; font-weight: 650; }',
        '.subtitle { font-size: 14px; fill: #66707D; }',
        '.axis { font-size: 13px; fill: #68717E; }',
        '.axis-title { font-size: 14px; font-weight: 600; fill: #4D5663; }',
        '.legend { font-size: 13px; font-weight: 600; }',
        '</style>',
        '<text class="title" x="92" y="46">Training reward trajectories</text>',
        '<text class="subtitle" x="92" y="73">Official tau2 airline train tasks; mean over 120 rollouts per global step</text>',
    ]

    legend_x = 92
    for item in series:
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="101" x2="{legend_x + 27}" y2="101" '
                f'stroke="{item.color}" stroke-width="3" stroke-linecap="round"/>',
                f'<circle cx="{legend_x + 13.5}" cy="101" r="3" fill="{item.color}"/>',
                f'<text class="legend" x="{legend_x + 36}" y="106">{escape(item.label)}</text>',
            ]
        )
        legend_x += 36 + len(item.label) * 7.2 + 30

    y_tick = y_min
    while y_tick <= y_max + 1e-9:
        y = y_pos(y_tick)
        parts.extend(
            [
                f'<line x1="{PLOT_LEFT}" y1="{y:.2f}" x2="{PLOT_RIGHT}" y2="{y:.2f}" '
                'stroke="#E6E9ED" stroke-width="1"/>',
                f'<text class="axis" x="{PLOT_LEFT - 14}" y="{y + 4:.2f}" '
                f'text-anchor="end">{y_tick:.2f}</text>',
            ]
        )
        y_tick += 0.05

    x_ticks = [1, 5, 10, 15, 20, 25, 30, 35]
    for tick in (tick for tick in x_ticks if x_min <= tick <= x_max):
        x = x_pos(tick)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{PLOT_BOTTOM}" x2="{x:.2f}" y2="{PLOT_BOTTOM + 7}" '
                'stroke="#A9B0B9" stroke-width="1"/>',
                f'<text class="axis" x="{x:.2f}" y="{PLOT_BOTTOM + 28}" '
                f'text-anchor="middle">{tick}</text>',
            ]
        )

    parts.extend(
        [
            f'<line x1="{PLOT_LEFT}" y1="{PLOT_TOP}" x2="{PLOT_LEFT}" y2="{PLOT_BOTTOM}" '
            'stroke="#A9B0B9" stroke-width="1.2"/>',
            f'<line x1="{PLOT_LEFT}" y1="{PLOT_BOTTOM}" x2="{PLOT_RIGHT}" y2="{PLOT_BOTTOM}" '
            'stroke="#A9B0B9" stroke-width="1.2"/>',
            f'<text class="axis-title" x="{(PLOT_LEFT + PLOT_RIGHT) / 2:.2f}" y="603" '
            'text-anchor="middle">Global training step</text>',
            '<text class="axis-title" x="27" y="337" text-anchor="middle" '
            'transform="rotate(-90 27 337)">Mean rollout reward</text>',
        ]
    )

    for item in series:
        points = " ".join(
            f"{x_pos(step):.2f},{y_pos(value):.2f}" for step, value in item.values
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{item.color}" '
            'stroke-width="2.7" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        for step, value in item.values:
            parts.append(
                f'<circle cx="{x_pos(step):.2f}" cy="{y_pos(value):.2f}" r="2.65" '
                f'fill="#FFFFFF" stroke="{item.color}" stroke-width="1.7">'
                f'<title>{escape(item.key.upper())} step {step}: {value:.5f}</title></circle>'
            )

    parts.extend(
        [
            '<text class="subtitle" x="1122" y="626" text-anchor="end">'
            'E5 resumed segments merged by global step; raw values are not smoothed.</text>',
            '</svg>',
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    args = _parse_args()
    series = _load_series(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render_svg(series), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
