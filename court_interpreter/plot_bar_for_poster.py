import japanize_matplotlib  # noqa: F401
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# 凡例名の日本語化
METRICS_LABEL_JA = {
    "omission": "省略",
    "addition": "付加",
    "word_meaning": "単語の意味",
    "question": "疑問文",
    "fluency": "流暢性",
    "average": "平均",
}

COLOR_PALETTE = {
    "question": colors[0],
    "omission": colors[1],
    "addition": colors[2],
    "word_meaning": colors[3],
    "fluency": colors[4],
    "average": colors[5],
}
COLOR_PALETTE2 = {
    "question": "#1F3A5F",
    "omission": "#2E5E8C",
    "addition": "#3F78B2",
    "word_meaning": "#5B94CC",
    "fluency": "#82B1E3",
    "average": "#B7D3F2",
}
COLOR_PALETTE3 = {
    "question": "#1F3A5F",
    "omission": "#2E5EAA",
    "addition": "#4A7BC8",
    "word_meaning": "#7FA6E8",
    "fluency": "#BFD3F5",
    "average": "#4A4A4A",
}

# hue の並び順も固定（常にこの順で色割当・legend順になる）
HUE_ORDER_ALL = [
    "question",
    "omission",
    "addition",
    "word_meaning",
    "fluency",
    "average",
]
HUE_ORDER_NO_QUESTION = ["omission", "addition", "word_meaning", "fluency", "average"]

fig, axes = plt.subplots(4, 2, figsize=(9, 7.5))
plt.subplots_adjust(
    left=0.05, right=0.98, bottom=0.1, top=0.96, wspace=0.05, hspace=0.4
)


def plot_bar_handbook(ax, df, title):
    for system in ["target", "gpt", "llama", "azure"]:
        results = []
        for metrics in ["omission", "addition", "word_meaning", "fluency"]:
            results.extend(list(df[f"{metrics}_{system}"]))
        df[f"average_{system}"] = np.mean(results)

    system_list, metrics_list, average_list, std_list = [], [], [], []
    for metrics in ["omission", "addition", "word_meaning", "fluency", "average"]:
        for system in ["target", "gpt", "llama", "azure"]:
            system_list.append(system)
            metrics_list.append(metrics)
            average_list.append(df[f"{metrics}_{system}"].mean())
            std_list.append(df[f"{metrics}_{system}"].std())

    df_plot = pd.DataFrame(
        {
            "system": system_list,
            "metrics": metrics_list,
            "average": average_list,
            "std": std_list,
        }
    )

    sns.barplot(
        x="system",
        y="average",
        hue="metrics",
        data=df_plot,
        hue_order=HUE_ORDER_NO_QUESTION,  # ←順序固定
        ax=ax,
        palette=COLOR_PALETTE,  # ←色固定
    )
    ax.set_title(title)
    # for bar in ax.patches:
    #     bar.set_hatch("+")


def plot_bar_question(ax, df, title):
    for system in ["gpt", "llama", "azure"]:
        results = []
        for metrics in ["omission", "addition", "word_meaning", "question", "fluency"]:
            results.extend(list(df[f"{metrics}_{system}"]))
        df[f"average_{system}"] = np.mean(results)

    system_list, metrics_list, average_list, std_list = [], [], [], []
    for metrics in [
        "question",
        "omission",
        "addition",
        "word_meaning",
        "fluency",
        "average",
    ]:
        for system in ["gpt", "llama", "azure"]:
            system_list.append(system)
            metrics_list.append(metrics)
            average_list.append(df[f"{metrics}_{system}"].mean())
            std_list.append(df[f"{metrics}_{system}"].std())

    df_plot = pd.DataFrame(
        {
            "system": system_list,
            "metrics": metrics_list,
            "average": average_list,
            "std": std_list,
        }
    )

    sns.barplot(
        x="system",
        y="average",
        hue="metrics",
        data=df_plot,
        hue_order=HUE_ORDER_ALL,  # ←順序固定
        ax=ax,
        palette=COLOR_PALETTE,  # ←色固定
    )
    ax.set_title(title)
    # for bar in ax.patches:
    #     bar.set_hatch("x")


for i, dataset in enumerate(["handbook", "question"]):
    for j, language in enumerate(["vietnamese", "chinese"]):
        dataset_ja = "ハンドブック" if dataset == "handbook" else "疑問文"
        language_ja = "ベトナム語" if language == "vietnamese" else "中国語"
        ax = axes[2 * i, j]
        ax.grid(axis="y", linestyle="-", alpha=0.7)
        human_df = pd.read_csv(
            f"../output/evaluation/human/expert/remapped_{dataset}_evaluation_set_{language}.csv"
        )
        if dataset == "handbook":
            plot_bar_handbook(
                ax,
                human_df,
                title=f"{dataset_ja} - {language_ja} - 人手評価",
            )
        else:
            plot_bar_question(
                ax,
                human_df,
                title=f"{dataset_ja} - {language_ja} - 人手評価",
            )
        ax.legend().remove()
        ax.set_ylim(0, 1)
        ax.set_xlabel("")  # Remove x-axis label
        ax.set_ylabel("")  # Remove y-axis label
        if i == 0:
            ax.set_xticklabels(
                ["既存対訳", "GPT", "Llama", "Azure"]
            )  # Set x-axis tick labels
        else:
            ax.set_xticklabels(["GPT", "Llama", "Azure"])  # Set x-axis tick labels
        if j == 1:
            ax.set_yticklabels([])  # Remove y-axis ticks
            ax.set_ylabel("")  # Remove y-axis label
        ax = axes[2 * i + 1, j]
        ax.grid(axis="y", linestyle="-", alpha=0.7)
        llm_df = pd.read_csv(
            f"../output/evaluation/llm/remapped_{dataset}_evaluation_set_{language}.csv"
        )
        if dataset == "handbook":
            plot_bar_handbook(
                ax,
                llm_df,
                title=f"{dataset_ja} - {language_ja} - LLM",
            )
        else:
            plot_bar_question(
                ax,
                llm_df,
                title=f"{dataset_ja} - {language_ja} - LLM",
            )
        ax.set_ylim(0, 1)
        ax.set_xlabel("")  # Remove x-axis label
        ax.set_ylabel("")  # Remove y-axis label
        if i == 0:
            ax.legend().remove()
            ax.set_xticklabels(
                ["既存対訳", "GPT", "Llama", "Azure"]
            )  # Set x-axis tick labels
        else:
            ax.set_xticklabels(["GPT", "Llama", "Azure"])  # Set x-axis tick labels
            if j == 0:
                leg = ax.legend(loc="upper left", bbox_to_anchor=(0.2, -0.2), ncol=6)
                # legend の表示名を metrics → 日本語へ
                for t in leg.get_texts():
                    key = t.get_text()
                    if key in METRICS_LABEL_JA:
                        t.set_text(METRICS_LABEL_JA[key])
            else:
                ax.legend().remove()
        if j == 1:
            ax.set_yticklabels([])  # Remove y-axis ticks
            ax.set_ylabel("")  # Remove y-axis label

plt.savefig("../output/analysis/bar_plot_for_poster.pdf")
