"""Figures for the pitch."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GLOSS = {"Hello": "MOI\nhello", "Thank you": "KIITOS\nthank you", "No": "EI\nno", "Okay": "OKEI\nokay"}


def plot_confusion(cm, classes, path, title):
    names = [GLOSS.get(c, c) for c in classes]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                    color="white" if norm[i, j] > 0.6 else "#222")
    ax.set_xticks(range(len(classes)), names, fontsize=8)
    ax.set_yticks(range(len(classes)), names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    acc = np.trace(cm) / cm.sum()
    ax.set_title(f"{title}\naccuracy {acc:.1%} (n={cm.sum()})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_reject(correct, wrong, negs, thr, path):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bins = np.linspace(0, max([max(correct)] + [max(v) for v in negs.values()] + ([max(wrong)] if wrong else [])), 40)
    ax.hist(correct, bins, alpha=0.6, label="real signs, correct")
    if wrong:
        ax.hist(wrong, bins, alpha=0.8, label="real signs, wrong")
    for k, v in negs.items():
        ax.hist(v, bins, alpha=0.4, label=f"non-sign: {k}")
    ax.axvline(thr, color="k", ls="--", label=f"reject > {thr:.2f}")
    ax.set_xlabel("DTW distance to nearest template")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
