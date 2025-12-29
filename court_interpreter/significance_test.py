from itertools import combinations
from pathlib import Path

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------
# Config
# -----------------------------
EVALUATOR = "human/expert"  # "human/expert" or "llm"
DATASET = "handbook"  # "handbook" or "question"
LANGUAGE = "vietnamese"  # "vietnamese" or "chinese" or "english"
CSV_PATH = (
    Path(
        f"../output/evaluation/{EVALUATOR}/remapped_{DATASET}_evaluation_set_{LANGUAGE}.csv"
    )
    if EVALUATOR == "human/expert"
    else Path(
        f"../output/evaluation/{EVALUATOR}/filtered_{DATASET}_evaluation_set_{LANGUAGE}.csv"
    )
)
OUT_DIR = Path(
    f"../output/analysis/significance_tests/{EVALUATOR}/{DATASET}_{LANGUAGE}"
)
SYSTEM_ORDER = (
    ["target", "gpt", "llama", "azure"]
    if DATASET == "handbook"
    else ["gpt", "llama", "azure"]
)
SYSTEM_ORDER_JA = (
    ["既存対訳", "GPT", "Llama", "Azure"]
    if DATASET == "handbook"
    else ["GPT", "Llama", "Azure"]
)
N_BOOT_CI = 20000
N_BOOT_TEST = 50000
CI = 0.95
ALPHA = 0.05
CORRECTION = "none"  # "none" or "holm"
EFFECT_THRESHOLD = 0.0  # ★スターまみれを抑えたいなら 0.05 などに
RNG_SEED = 12345


# -----------------------------
# Bootstrap helpers
# -----------------------------
def bootstrap_mean_ci(x, n_boot=20000, ci=0.95, rng=None):
    x = np.asarray(x, dtype=float)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    alpha = 1 - ci
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap_diff_test(x, y, n_boot=50000, ci=0.95, rng=None):
    """
    Paired bootstrap over items: resample item indices, compute mean(x - y).
    Returns mean_diff, ci_low, ci_high, p_two_sided.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert len(x) == len(y)
    n = len(x)
    # mean diff of given samples
    mean_diff = float((x - y).mean())
    # mean diffs of bootstrap samples
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = (x[idx] - y[idx]).mean(axis=1)
    alpha = 1 - ci
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    p = float(min(1.0, p))
    return mean_diff, float(lo), float(hi), p


def holm_bonferroni(pvals):
    """Holm-Bonferroni adjusted p-values, aligned to original order."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    p_sorted = pvals[order]
    adj_sorted = np.minimum(1.0, (m - np.arange(m)) * p_sorted)
    for i in range(1, m):
        adj_sorted[i] = max(adj_sorted[i], adj_sorted[i - 1])
    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    return adj.tolist()


def p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def compute_pairwise(values_by_system, systems_order, correction="holm"):
    rows, pvals = [], []
    for a, b in combinations(systems_order, 2):
        md, lo, hi, p = paired_bootstrap_diff_test(
            values_by_system[a], values_by_system[b], n_boot=N_BOOT_TEST, ci=CI, rng=rng
        )
        rows.append(
            {"A": a, "B": b, "mean_diff(A-B)": md, "ci_low": lo, "ci_high": hi, "p": p}
        )
        pvals.append(p)
    padj = holm_bonferroni(pvals) if correction == "holm" else pvals
    for r, pa in zip(rows, padj):
        r["p_adj"] = float(pa)
        r["sig"] = p_to_stars(pa)
    return pd.DataFrame(rows)


def compute_mean_ci(values_by_system, systems_order):
    rows = []
    for s in systems_order:
        mean, lo, hi = bootstrap_mean_ci(
            values_by_system[s], n_boot=N_BOOT_CI, ci=CI, rng=rng
        )
        rows.append({"system": s, "mean": mean, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


# -----------------------------
# Load + parse
# -----------------------------


def parse_metric_system_columns(columns):
    out = []
    for c in columns:
        if c == "id" or "_" not in c:
            continue
        metric, system = c.rsplit("_", 1)
        out.append((metric, system, c))
    return out


df = pd.read_csv(CSV_PATH)
ms_cols = parse_metric_system_columns(df.columns)
metrics = sorted({m for m, s, c in ms_cols})
values = {m: {} for m in metrics}
for m, s, c in ms_cols:
    values[m][s] = df[c].to_numpy(dtype=float)

# overall mean per row per system
overall = {}
for s in SYSTEM_ORDER:
    cols = [f"{m}_{s}" for m in metrics if f"{m}_{s}" in df.columns]
    overall[s] = df[cols].mean(axis=1).to_numpy(dtype=float)

# -----------------------------
# Compute pairwise tests and mean CIs
# -----------------------------

rng = np.random.default_rng(RNG_SEED)
pairwise_tables = {
    m: compute_pairwise(values[m], SYSTEM_ORDER, correction=CORRECTION) for m in metrics
}
mean_ci_tables = {m: compute_mean_ci(values[m], SYSTEM_ORDER) for m in metrics}
pairwise_tables["overall_mean"] = compute_pairwise(
    overall, SYSTEM_ORDER, correction=CORRECTION
)
mean_ci_tables["overall_mean"] = compute_mean_ci(overall, SYSTEM_ORDER)


# -----------------------------
# Triangle plot
# -----------------------------
def build_triangle_matrices(pair_df, systems_order):
    n = len(systems_order)
    idx = {s: i for i, s in enumerate(systems_order)}
    diff = np.full((n, n), np.nan, dtype=float)
    p_adj = np.full((n, n), np.nan, dtype=float)
    sig = np.full((n, n), "", dtype=object)

    for _, r in pair_df.iterrows():
        a, b = r["A"], r["B"]
        i, j = idx[a], idx[b]
        diff[i, j] = r["mean_diff(A-B)"]
        p_adj[i, j] = r["p_adj"]
        sig[i, j] = r["sig"]
        # symmetric
        diff[j, i] = -r["mean_diff(A-B)"]
        p_adj[j, i] = r["p_adj"]
        sig[j, i] = r["sig"]
    return diff, p_adj, sig


metrics_name_ja = {
    "omission": "省略",
    "addition": "付加",
    "word_meaning": "単語の意味",
    "question": "疑問文の訳出",
    "fluency": "流暢性",
    "overall_mean": "平均",
}


def plot_matrix_heatmap(
    metric_name,
    pair_df,
    systems_order,
    out_path,
    alpha=0.05,
    effect_threshold=0.0,
    mode="both",  # "lower" or "upper" or "both"
    show_text=True,
):
    """
    Show mean differences matrix (row - col).
    - mode="lower": show only lower triangle (i>j)
    - mode="upper": show only upper triangle (i<j)
    - mode="both" : show full matrix (except diagonal)
    Text in each cell: +0.123*** (stars use adjusted p in pair_df)
    """
    # --- build full symmetric matrices (diff, p_adj, sig) ---
    n = len(systems_order)
    idx = {s: i for i, s in enumerate(systems_order)}

    diff = np.full((n, n), np.nan, dtype=float)
    p_adj = np.full((n, n), np.nan, dtype=float)
    sig = np.full((n, n), "", dtype=object)

    # pair_df has rows for A<B in some order; fill both directions
    for _, r in pair_df.iterrows():
        a, b = r["A"], r["B"]
        i, j = idx[a], idx[b]
        d = float(r["mean_diff(A-B)"])
        pa = float(r["p_adj"])
        st = str(r["sig"])

        diff[i, j] = d
        diff[j, i] = -d
        p_adj[i, j] = pa
        p_adj[j, i] = pa
        sig[i, j] = st
        sig[j, i] = st

    # diagonal is meaningless
    np.fill_diagonal(diff, np.nan)
    np.fill_diagonal(p_adj, np.nan)

    # --- mask depending on mode ---
    if mode == "lower":
        mask = np.triu(np.ones((n, n), dtype=bool))  # diag+upper masked
    elif mode == "upper":
        mask = np.tril(np.ones((n, n), dtype=bool))  # diag+lower masked
    elif mode == "both":
        mask = np.eye(n, dtype=bool)  # only diag masked
    else:
        raise ValueError("mode must be 'lower', 'upper', or 'both'")

    diff_plot = diff.copy()
    diff_plot[mask] = np.nan

    vmax = np.nanmax(np.abs(diff_plot))
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(diff_plot, vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(n), SYSTEM_ORDER_JA)
    ax.set_yticks(np.arange(n), SYSTEM_ORDER_JA)
    ax.set_title(metrics_name_ja[metric_name], fontsize=14)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    # grid
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", linestyle=":", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    if show_text:
        for i in range(n):
            for j in range(n):
                if mask[i, j]:
                    continue
                if np.isnan(diff[i, j]):
                    continue
                d = diff[i, j]
                pa = p_adj[i, j]
                # show stars only if significant AND (optional) effect is large enough
                st = sig[i, j] if (pa < alpha and abs(d) >= effect_threshold) else ""
                ax.text(
                    j,
                    i,
                    f"{d:+.3f}{st}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if d < 0 else "black",
                )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# -----------------------------
# Export
# -----------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)

for m in metrics:
    plot_matrix_heatmap(
        metric_name=f"{m}",
        pair_df=pairwise_tables[m],
        systems_order=SYSTEM_ORDER,
        out_path=OUT_DIR / f"triangle_{m}.pdf",
        alpha=ALPHA,
        effect_threshold=EFFECT_THRESHOLD,
        mode="both",
        show_text=True,
    )

plot_matrix_heatmap(
    metric_name="overall_mean",
    pair_df=pairwise_tables["overall_mean"],
    systems_order=SYSTEM_ORDER,
    out_path=OUT_DIR / "triangle_overall_mean.pdf",
    alpha=ALPHA,
    effect_threshold=EFFECT_THRESHOLD,
    mode="both",
    show_text=True,
)

# also save tables
pairs_all = []
means_all = []
for m in metrics + ["overall_mean"]:
    tmp = pairwise_tables[m].copy()
    tmp.insert(0, "metric", m)
    pairs_all.append(tmp)
    tmp2 = mean_ci_tables[m].copy()
    tmp2.insert(0, "metric", m)
    means_all.append(tmp2)

pd.concat(pairs_all, ignore_index=True).to_csv(
    OUT_DIR / "paired_bootstrap_all_metrics.csv", index=False
)
pd.concat(means_all, ignore_index=True).to_csv(
    OUT_DIR / "bootstrap_mean_ci_all_metrics.csv", index=False
)
print("done")
