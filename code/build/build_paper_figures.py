"""Build the two vector figures used in the main paper from audited data."""

from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "vcpo-paper" / "figures"
AUDIT = ROOT / "vcpo-paper" / "RESULTS_AUDIT.json"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#707070"
LIGHT = "#F3F4F6"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def box(ax, xy, width, height, text, edge, face="white", fontsize=6.5, lw=1.0):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.014,rounding_size=0.022",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.12,
    )
    return patch


def arrow(ax, start, end, color=GRAY, lw=1.0, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        )
    )


def build_overview():
    fig, ax = plt.subplots(figsize=(3.35, 2.18))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.015, 0.965, "(a) All sampled programs fail", weight="bold", va="top", fontsize=7.2)
    for i in range(4):
        x = 0.02 + i * 0.145
        box(ax, (x, 0.71), 0.115, 0.13, f"Rollout {i + 1}\n$R=0$", "#999999", LIGHT, 5.8, 0.8)
        arrow(ax, (x + 0.057, 0.70), (0.31, 0.56), "#999999", 0.75)
    box(ax, (0.18, 0.46), 0.27, 0.11, "center rewards\n" + r"$\widehat A^r=0$", "#999999", "#FAFAFA", 6.1)
    arrow(ax, (0.45, 0.515), (0.60, 0.515), "#999999", 0.9)
    box(ax, (0.61, 0.46), 0.36, 0.11, "group-relative\nreward gradient $=0$", "#777777", "#FAFAFA", 6.1)

    box(ax, (0.02, 0.20), 0.24, 0.14, "Certificate $s$\n(correct, $R=1$)", GREEN, "#EAF7F2", 6.3, 1.1)
    ax.text(0.015, 0.405, "(b) A certificate supplies a gradient", weight="bold", va="top", fontsize=7.2)
    arrow(ax, (0.26, 0.29), (0.45, 0.29), ORANGE, 1.2)
    arrow(ax, (0.26, 0.23), (0.74, 0.125), BLUE, 1.2, -0.11)
    ax.text(0.35, 0.315, "weight $R$", color=ORANGE, ha="center", va="bottom", fontsize=6)
    ax.text(0.48, 0.105, r"weight $(p_\theta/q)R$", color=BLUE, ha="center", va="top", fontsize=6)
    box(ax, (0.46, 0.23), 0.23, 0.13, r"$\nabla F_q$" + "\nlikelihood", ORANGE, "#FCEFEA", 6.4, 1.2)
    box(ax, (0.74, 0.06), 0.23, 0.13, r"$\nabla J$" + "\nexpected reward", BLUE, "#EAF3F8", 6.4, 1.2)
    ax.text(0.98, 0.01, "fixed $q,R$; support coverage for correction", ha="right", va="bottom", fontsize=5.2, color=GRAY)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(OUT / "objective_overview.pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(OUT / "objective_overview.png", dpi=240, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def build_summary():
    data = json.loads(AUDIT.read_text())
    models = [
        "Qwen3.5-0.8B",
        "Qwen3.5-2B",
        "Qwen3.5-4B",
        "Qwen3.5-9B",
        "OLMo-2-1124-7B",
        "glm-4-9b",
    ]
    labels = ["Qwen\n0.8B", "Qwen\n2B", "Qwen\n4B", "Qwen\n9B", "OLMo\n7B", "GLM\n9B"]
    component_keys = ["hsd", "opid", "edge_opd"]
    component_labels = ["Teacher CE", "Log-shift", "Mask-only"]
    markers = ["o", "s", "D"]
    offsets = [-0.22, 0.0, 0.22]

    align = {
        r["model"]: r
        for r in data["alignment"]
        if r["dataset"] == "humaneval" and r["meta"]["extra"]["n_tasks"] == 64
    }
    decay = {(r["model"], r["dataset"]): r for r in data["decay"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.15, 2.75), gridspec_kw={"width_ratios": [1.23, 1]})
    xs = list(range(len(models)))

    for mi, model in enumerate(models):
        if model == "glm-4-9b":
            ax1.add_patch(Rectangle((mi - 0.43, -0.04), 0.86, 0.55, color="#EEEEEE", zorder=0))
        for key, marker, off in zip(component_keys, markers, offsets):
            values = align[model]["alignment"][key]
            fq = values["cos_with_grad_Fq"]
            gc = values["cos_with_grad_VCPO"]
            x = mi + off
            alpha = 0.38 if model == "glm-4-9b" else 1.0
            ax1.plot([x, x], [gc, fq], color="#B5B5B5", lw=0.75, zorder=1, alpha=alpha)
            ax1.scatter(x, fq, marker=marker, s=25, color=ORANGE, edgecolor="white", linewidth=0.35, zorder=3, alpha=alpha)
            ax1.scatter(x, gc, marker=marker, s=25, facecolor="white", edgecolor=BLUE, linewidth=1.05, zorder=3, alpha=alpha)

    ax1.axhline(0, color="#AAAAAA", lw=0.65)
    ax1.set_xticks(xs, labels)
    ax1.set_ylabel("Cosine similarity")
    ax1.set_ylim(-0.035, 0.51)
    ax1.set_title("(a) Diagnostic alignment on HumanEval", loc="left", weight="bold")
    ax1.grid(axis="y", color="#E5E5E5", lw=0.65)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.text(5, 0.485, "excluded", ha="center", va="top", fontsize=5.8, color=GRAY)

    ref_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markeredgecolor="white", markersize=5.5, label=r"with $\nabla F_q$"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=BLUE, markersize=5.5, label=r"with $g_{\mathrm{C}}$"),
    ]
    comp_handles = [
        Line2D([0], [0], marker=m, color=GRAY, markerfacecolor=GRAY, lw=0, markersize=4.5, label=l)
        for m, l in zip(markers, component_labels)
    ]
    ax1.legend(handles=ref_handles + comp_handles, ncol=3, loc="upper left", bbox_to_anchor=(-0.01, -0.23), frameon=False, columnspacing=0.9, handletextpad=0.35)

    he_ratio, mb_ratio, he_abs, mb_abs = [], [], [], []
    for model in models:
        he = decay[(model, "humaneval")]["probability_ess"]
        mb = decay[(model, "mbpp")]["probability_ess"]
        he_abs.append(he)
        mb_abs.append(mb)
        he_ratio.append(he / 164)
        mb_ratio.append(mb / 257)
    for i in xs:
        ax2.plot([i - 0.09, i + 0.09], [he_ratio[i], mb_ratio[i]], color="#C7C7C7", lw=0.8, zorder=1)
    ax2.scatter([x - 0.09 for x in xs], he_ratio, color=GREEN, marker="o", s=27, label="HumanEval / 164", zorder=3)
    ax2.scatter([x + 0.09 for x in xs], mb_ratio, color=PURPLE, marker="s", s=25, label="MBPP / 257", zorder=3)
    for i, (yr, val) in enumerate(zip(he_ratio, he_abs)):
        ax2.annotate(f"{val:.2f}", (i - 0.09, yr), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=5.2, color=GREEN)
    for i, (yr, val) in enumerate(zip(mb_ratio, mb_abs)):
        ax2.annotate(f"{val:.2f}", (i + 0.09, yr), xytext=(0, -8), textcoords="offset points", ha="center", fontsize=5.2, color=PURPLE)
    ax2.set_xticks(xs, labels)
    ax2.set_ylabel("ESS / certificate count")
    ax2.set_ylim(0, 0.062)
    ax2.set_title("(b) Probability-weighted ESS", loc="left", weight="bold")
    ax2.grid(axis="y", color="#E5E5E5", lw=0.65)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.23), frameon=False, handletextpad=0.35, columnspacing=0.9)

    fig.subplots_adjust(left=0.075, right=0.99, top=0.91, bottom=0.28, wspace=0.30)
    fig.savefig(OUT / "diagnostic_summary.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "diagnostic_summary.png", dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build_overview()
    build_summary()
    print(f"Wrote figures to {OUT}")
