"""Plot Figure 1 from manuscript counts and labels."""

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from .common import ROOT, OUTPUTS


def run():
    data = json.loads(
        (ROOT / "figure1_reported_counts.json").read_text(encoding="utf-8")
    )
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(11, 12))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 11), ylim=(0, 14))
    ax.axis("off")
    boxes = {}
    texts = []

    def box(key, x, y, width, height, title, note="", size=14):
        ax.add_patch(
            FancyBboxPatch(
                (x, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0,rounding_size=0.065",
                edgecolor="#8ea6b6",
                facecolor="#F2F6F8",
                lw=1.1,
            )
        )
        boxes[key] = (x, y - height / 2, x + width, y + height / 2)
        t = ax.text(
            x + 0.15,
            y + height / 2 - 0.14,
            title,
            ha="left",
            va="top",
            weight="bold",
            fontsize=size,
            color="#233e50",
            linespacing=1.13,
        )
        fig.canvas.draw()
        bb = t.get_window_extent(fig.canvas.get_renderer())
        coords = ax.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        if note:
            texts.append(
                (
                    key,
                    ax.text(
                        x + 0.15,
                        coords[0, 1] - 0.12,
                        note,
                        ha="left",
                        va="top",
                        fontsize=11.2,
                        color="#4e6e82",
                        linespacing=1.15,
                    ),
                )
            )
        texts.append((key, t))

    ys = [12.9, 11.15, 9.65, 8.15, 6.65, 5.15, 3.65, 1.55]
    for i, z in enumerate(data["main_nodes"]):
        box(
            "L" + str(i),
            0.25,
            ys[i],
            6.0,
            z["height"],
            z["label"] + "\nn = " + format(z["n"], ","),
            z["note"],
        )
    for i, z in enumerate(data["side_nodes"]):
        box(
            "R" + str(i),
            7.15,
            z.get("y", ys[z["row"]]),
            3.6,
            z["height"],
            z["label"] + "\nn = " + format(z["n"], ","),
            z["note"],
            12.5,
        )

    def arrow(a, b):
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="-|>",
                mutation_scale=12,
                lw=1.1,
                color="#8ea6b6",
                shrinkA=0,
                shrinkB=0,
            )
        )

    for i in range(7):
        arrow((3.25, boxes["L" + str(i)][1]), (3.25, boxes["L" + str(i + 1)][3]))
    for i in [0, 1, 2, 4, 5, 6]:
        arrow((6.25, ys[i]), (7.15, ys[i]))
    gap = (boxes["R3"][1] + boxes["R4"][3]) / 2
    ax.plot([8.95, 8.95, 5.8], [boxes["R3"][1], gap, gap], color="#8ea6b6", lw=1.1)
    arrow((5.8, gap), (5.8, boxes["L4"][3]))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for key, t in texts:
        bb = t.get_window_extent(renderer)
        lo, hi = ax.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        x0, y0, x1, y1 = boxes[key]
        assert (
            lo[0] >= x0 and hi[0] <= x1 - 0.06 and lo[1] >= y0 + 0.06 and hi[1] <= y1
        ), (key, t.get_text())
    folder = OUTPUTS / "figures"
    folder.mkdir(exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(folder / ("Figure1." + ext), dpi=350, facecolor="white")
    plt.close(fig)
