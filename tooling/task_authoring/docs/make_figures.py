"""Regenerate the figures in tooling/task_authoring/docs/assets from the pipeline's own outputs.

Every number in the documentation's charts comes from files the pipeline writes, so the figures
can be rebuilt after each run instead of going stale:

    python3 tooling/task_authoring/docs/make_figures.py
    python3 tooling/task_authoring/docs/make_figures.py --render-dir tmp/droid100/review   # + physics check

Inputs (all gitignored, produced by the pipeline):
    tmp/droid100/DROID_clusters.json         select_droid100.py output (stage 1)
    tmp/droid100/interactive_report.json     agent.cli export report (stages 2-4)
    realm/config/tasks/REALM_DROID100/       the exported task configs
    <render-dir>/*.json                      optional: render_review.py per-task records (stage 5)

Each figure is written twice, `<name>-light.svg` and `<name>-dark.svg`, so the Markdown can pick
the one matching the reader's GitHub theme with a <picture> element.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import Polygon, Rectangle

REPO = Path(__file__).resolve().parents[3]
ASSETS = Path(__file__).resolve().parent / "assets"

# Validated categorical slots (blue, orange, aqua) and neutral inks, per theme.
THEMES = {
    "light": {"ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
              "neutral": "#b9b7af", "region": "#f0efec"},
    "dark": {"ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70",
             "neutral": "#5f5e59", "region": "#262624"},
}

#: Decline reasons are free text written by the generator. These rules group them for the chart;
#: the first matching rule wins, so the order encodes precedence.
DECLINE_RULES = [
    ("Failed the physics check", r"^render v\d|the render showed"),
    ("Several objects or steps",
     r"plural|several|two main|chained|three manipulations|two stages|six manipulations|implies several"),
    ("Object can't be identified", r"identified only|unidentified|ambiguous|names no goal|\"the maker\""),
    ("Motion REALM can't score", r"rubric|scor|orientation goal|pouring|stirring"),
    ("No suitable asset", r"catalogue|asset|flat sheet|deformable|the only |honestly|does not pack|told apart|scaled down"),
    ("Fixture or furniture", r"fixture|furniture|appliance"),
]


def classify_decline(reason: str) -> str:
    text = reason.lower()
    for label, pattern in DECLINE_RULES:
        if re.search(pattern, text):
            return label
    return "Other"


def setup(theme: dict) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.5,
        "svg.fonttype": "none",
        "text.color": theme["ink"],
        "axes.labelcolor": theme["ink2"],
        "axes.edgecolor": theme["axis"],
        "xtick.color": theme["muted"],
        "ytick.color": theme["ink2"],
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
    })


def save(fig, name: str, mode: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / f"{name}-{mode}.svg"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15, metadata={"Date": None})
    plt.close(fig)


def hbar(ax, labels, values, colors, theme, total=None, fmt="{:,}"):
    """Horizontal bars, largest on top, value (and share) labelled at the bar end."""
    ys = range(len(labels))[::-1]
    ax.barh(list(ys), values, color=colors, height=0.62, edgecolor="none")
    ax.set_yticks(list(ys), labels)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=0, labelsize=9)
    ax.xaxis.grid(True, color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    top = max(values) if values else 1
    for y, v in zip(ys, values):
        text = fmt.format(v)
        if total:
            share = 100 * v / total
            text += f"  ({share:.0f}%)" if share >= 1 else "  (<1%)"
        ax.text(v + top * 0.015, y, text, va="center", ha="left", fontsize=9.5, color=theme["ink2"])
    ax.set_xlim(0, top * 1.22)


# ---------------------------------------------------------------- figures


def fig_funnel(data, theme, mode):
    stages = [
        ("Unique DROID instructions", data["unique_instructions"]),
        ("Merged into task clusters", data["clusters"]),
        ("Ranked candidates kept", data["ranked"]),
        ("Processed by the generator", data["processed"]),
        ("Became a task (submitted)", data["submitted"]),
        ("Distinct tasks after dedup", data["tasks"]),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    colors = [theme["neutral"]] * 3 + [theme["blue"]] * 3
    hbar(ax, [s for s, _ in stages], [v for _, v in stages], colors, theme)
    ax.set_title("From DROID text to REALM tasks", loc="left", fontsize=12, fontweight="bold",
                 color=theme["ink"], pad=10)
    save(fig, "funnel", mode)


def fig_outcomes(data, theme, mode):
    rows = [("Became a distinct task", data["tasks"], theme["blue"]),
            ("Duplicate of another task", data["duplicates"], theme["neutral"])]
    for label, count in data["declines"].most_common():
        rows.append((f"Declined: {label[0].lower()}{label[1:]}", count, theme["orange"]))
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    hbar(ax, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], theme,
         total=data["processed"])
    ax.set_title(f"What happened to the {data['processed']} processed instructions", loc="left",
                 fontsize=12, fontweight="bold", color=theme["ink"], pad=10)
    save(fig, "outcomes", mode)


def fig_selection(data, theme, mode):
    rejected = sorted(data["rejected_phrasings"].items(), key=lambda kv: -kv[1])
    fig, ax = plt.subplots(figsize=(7.4, 2.5))
    hbar(ax, [k.capitalize() for k, _ in rejected], [v for _, v in rejected],
         [theme["neutral"]] * len(rejected), theme)
    ax.set_title("Phrasings filtered out before generation (stage 1)", loc="left", fontsize=12,
                 fontweight="bold", color=theme["ink"], pad=10)
    save(fig, "selection-filter", mode)


def fig_family(data, theme, mode):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 2.7), gridspec_kw={"width_ratios": [1, 1.35]})
    types = data["task_types"].most_common()
    hbar(a, [t for t, _ in types], [v for _, v in types], [theme["blue"]] * len(types), theme,
         total=data["tasks"])
    a.set_title("Task types", loc="left", fontsize=11.5, fontweight="bold", color=theme["ink"])
    scenes = data["scenes"].most_common()
    hbar(b, [s.replace("_int", "").replace("/", " · ").replace("_", " ") for s, _ in scenes],
         [v for _, v in scenes], [theme["blue"]] * len(scenes), theme, total=data["tasks"])
    b.set_title("Scene and table", loc="left", fontsize=11.5, fontweight="bold", color=theme["ink"])
    fig.tight_layout(w_pad=3)
    save(fig, "family-composition", mode)


def _quat_yaw(q) -> float:
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _footprint(center, size, yaw):
    cx, cy = center
    hx, hy = size[0] / 2, size[1] / 2
    c, s = math.cos(yaw), math.sin(yaw)
    return [(cx + px * c - py * s, cy + px * s + py * c)
            for px, py in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]


def fig_layouts(data, theme, mode):
    examples = data["layout_examples"]
    cols = 3
    rows = math.ceil(len(examples) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10.2, 3.75 * rows))
    role_style = {
        "main_objects": ("main object", theme["blue"], 1.0, 4),
        "target_objects": ("target", theme["orange"], 1.0, 3),
        "immutables": ("source / fixture", theme["aqua"], 1.0, 2),
        "distractors": ("distractor", theme["neutral"], 0.9, 1),
    }
    for ax, (folder, doc, region) in zip(axes.flat, examples):
        width, depth = region
        ax.add_patch(Rectangle((0, 0), width, depth, facecolor=theme["region"],
                               edgecolor=theme["axis"], linewidth=1.0, zorder=0))
        for role, (_, color, alpha, z) in role_style.items():
            for obj in doc.get(role) or []:
                size = obj.get("bounding_box") or obj.get("size") or [0.05, 0.05, 0.05]
                pos = obj["relative_bbox_position"]
                yaw = _quat_yaw(obj.get("orientation") or [0, 0, 0, 1])
                poly = Polygon(_footprint(pos[:2], size, yaw), closed=True, facecolor=color,
                               alpha=alpha, edgecolor=theme["ink"] if role != "distractors" else "none",
                               linewidth=0.6, zorder=z)
                ax.add_patch(poly)
                label = str(obj.get("category") or obj.get("name")).replace("distractor_", "")
                label = label.replace("_", " ")
                # The main object sits on top of its container in pick tasks, so its label goes
                # above the footprint and everything else's below; a backing box keeps it legible.
                half = max(size[0] * abs(math.sin(yaw)) + size[1] * abs(math.cos(yaw)), 0.02) / 2
                above = role == "main_objects"
                ax.text(pos[0], pos[1] + (half + 0.01 if above else -half - 0.01), label,
                        ha="center", va="bottom" if above else "top", fontsize=7.4,
                        color=theme["ink"] if role != "distractors" else theme["ink2"], zorder=6,
                        bbox={"facecolor": theme["region"], "edgecolor": "none", "pad": 0.6,
                              "alpha": 0.85})
        # 10 cm scale bar under the region.
        ax.plot([0, 0.10], [-0.035, -0.035], color=theme["ink2"], linewidth=1.6,
                solid_capstyle="butt")
        ax.text(0.05, -0.045, "10 cm", ha="center", va="top", fontsize=7.4, color=theme["ink2"])
        ax.set_xlim(-0.03, max(width, 0.4) + 0.03)
        ax.set_ylim(-0.075, max(depth, 0.4) + 0.05)
        ax.set_aspect("equal")
        ax.axis("off")
        title = doc["instruction"]
        ax.set_title(f"“{title}”\n{doc['task_type']} · {next(iter(doc['supported_scenes'])).replace('_int', '')}",
                     fontsize=9.2, color=theme["ink"], loc="left")
    for ax in list(axes.flat)[len(examples):]:
        ax.axis("off")
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, alpha=a, edgecolor="none")
               for _, (lbl, c, a, _) in role_style.items()]
    fig.legend(handles, [s[0] for s in role_style.values()], loc="lower center", ncol=4,
               frameon=False, fontsize=9.5, labelcolor=theme["ink2"], bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Generated layouts seen from above, in each table region's own frame",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=theme["ink"])
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, "layouts", mode)


def fig_render(data, theme, mode):
    verdicts = data["render_verdicts"]
    findings = data["render_findings"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 2.8), gridspec_kw={"width_ratios": [1, 1.4]})
    order = [v for v in ("PASS", "FAIL", "ERROR", "NO_JSON", "BAD_JSON") if verdicts.get(v)]
    colors = {"PASS": theme["blue"], "FAIL": theme["orange"]}
    hbar(a, order, [verdicts[v] for v in order], [colors.get(v, theme["neutral"]) for v in order],
         theme, total=sum(verdicts.values()))
    a.set_title("Physics check verdicts", loc="left", fontsize=11.5, fontweight="bold",
                color=theme["ink"])
    common = findings.most_common(8)
    if common:
        hbar(b, [c for c, _ in common], [v for _, v in common], [theme["neutral"]] * len(common),
             theme)
    b.set_title("Findings raised (tasks may raise several)", loc="left", fontsize=11.5,
                fontweight="bold", color=theme["ink"])
    fig.tight_layout(w_pad=3)
    save(fig, "render-check", mode)


# ---------------------------------------------------------------- data


LAYOUT_EXAMPLES = [
    "put_the_marker_in_the_cup_d31114",
    "remove_the_lid_from_the_pot_8b4a39",
    "put_the_plate_on_top_of_the_bowl_f04584",
    "remove_the_marker_from_the_mug_e57716",
    "take_the_blue_block_out_of_the_box_and_put_it_in_the_bowl_52e039",
    "rotate_the_pot_20a53e",
]


def collect(args) -> dict:
    clusters = json.loads(Path(args.clusters).read_text(encoding="utf-8"))
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    scenes = yaml.safe_load(Path(args.scenes).read_text(encoding="utf-8"))
    family = Path(args.family)
    docs = {Path(p).parent.name: yaml.safe_load(Path(p).read_text(encoding="utf-8"))
            for p in sorted(glob.glob(str(family / "*" / "default.yaml")))}

    results = report["results"]
    declines = collections.Counter(classify_decline(r["ungroundable"]["reason"])
                                   for r in results if r.get("ungroundable"))
    data = {
        "unique_instructions": clusters["unique_instructions"],
        "clusters": clusters["clusters"],
        "ranked": len(clusters["tasks"]),
        "rejected_phrasings": clusters["rejected_phrasings"],
        "processed": report["total"],
        "submitted": report["grounded"] + report["duplicate"],
        "duplicates": report["duplicate"],
        "tasks": len(docs),
        "declines": declines,
        "task_types": collections.Counter(d["task_type"] for d in docs.values()),
        "scenes": collections.Counter(f"{s}/{r[0]}" for d in docs.values()
                                      for s, r in d["supported_scenes"].items()),
    }
    examples = []
    for folder in LAYOUT_EXAMPLES:
        doc = docs.get(folder)
        if not doc:
            continue
        scene, regions = next(iter(doc["supported_scenes"].items()))
        reg = scenes[scene][regions[0]]
        examples.append((folder, doc, (reg["x_max"] - reg["x_min"], reg["y_max"] - reg["y_min"])))
    data["layout_examples"] = examples

    if args.render_dir:
        verdicts, findings = collections.Counter(), collections.Counter()
        for path in sorted(Path(args.render_dir).glob("*.json")):
            if path.name.startswith("_"):
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            verdicts[record.get("verdict", "NO_JSON")] += 1
            for code in {f["code"] for f in record.get("compute_findings") or []}:
                findings[code] += 1
        data["render_verdicts"], data["render_findings"] = verdicts, findings
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clusters", default=REPO / "tmp/droid100/DROID_clusters.json")
    parser.add_argument("--report", default=REPO / "tmp/droid100/interactive_report.json")
    parser.add_argument("--family", default=REPO / "realm/config/tasks/REALM_DROID100")
    parser.add_argument("--scenes", default=REPO / "realm/config/scenes/scenes.yaml")
    parser.add_argument("--render-dir", default=None,
                        help="directory of render_review.py JSON records (adds render-check.svg)")
    args = parser.parse_args()
    data = collect(args)
    for mode, theme in THEMES.items():
        setup(theme)
        fig_funnel(data, theme, mode)
        fig_outcomes(data, theme, mode)
        fig_selection(data, theme, mode)
        fig_family(data, theme, mode)
        fig_layouts(data, theme, mode)
        if data.get("render_verdicts"):
            fig_render(data, theme, mode)
    summary = {k: (dict(v) if isinstance(v, collections.Counter) else v)
               for k, v in data.items() if k != "layout_examples"}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
