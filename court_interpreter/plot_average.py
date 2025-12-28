# 人手評価と LLM の平均値をプロット
from collections import defaultdict

import japanize_matplotlib  # noqa: F401
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

# Color-blind friendly (okabe-ito) colors
colors = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

avg_std_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

for dataset in ["handbook", "question"]:
    for language in ["vietnamese", "chinese", "english"]:
        human_df = pd.read_csv(
            f"../output/evaluation/human/expert/remapped_{dataset}_evaluation_set_{language}.csv"
        )
        llm_df = pd.read_csv(
            f"../output/evaluation/llm/remapped_{dataset}_evaluation_set_{language}.csv"
        )
        llm_df = llm_df[llm_df["id"].isin(human_df["id"])]
        if dataset == "handbook":
            for metrics in ["omission", "addition", "word_meaning", "fluency"]:
                human_results = np.array(
                    pd.concat(
                        [
                            human_df[f"{metrics}_target"],
                            human_df[f"{metrics}_gpt"],
                            human_df[f"{metrics}_llama"],
                            human_df[f"{metrics}_azure"],
                        ]
                    )
                ).astype(float)
                llm_results = np.array(
                    pd.concat(
                        [
                            llm_df[f"{metrics}_target"],
                            llm_df[f"{metrics}_gpt"],
                            llm_df[f"{metrics}_llama"],
                            llm_df[f"{metrics}_azure"],
                        ]
                    )
                ).astype(float)
                avg_std_dict[dataset][language][metrics]["human_avg"] = np.mean(
                    human_results
                )
                avg_std_dict[dataset][language][metrics]["human_ci"] = (
                    np.std(human_results, ddof=1) / np.sqrt(len(human_results)) * 1.96
                )
                avg_std_dict[dataset][language][metrics]["llm_avg"] = np.mean(
                    llm_results
                )
                avg_std_dict[dataset][language][metrics]["llm_ci"] = (
                    np.std(llm_results, ddof=1) / np.sqrt(len(llm_results)) * 1.96
                )

        else:
            for metrics in [
                "omission",
                "addition",
                "word_meaning",
                "question",
                "fluency",
            ]:
                human_results = np.array(
                    pd.concat(
                        [
                            human_df[f"{metrics}_gpt"],
                            human_df[f"{metrics}_llama"],
                            human_df[f"{metrics}_azure"],
                        ]
                    )
                ).astype(float)
                llm_results = np.array(
                    pd.concat(
                        [
                            llm_df[f"{metrics}_gpt"],
                            llm_df[f"{metrics}_llama"],
                            llm_df[f"{metrics}_azure"],
                        ]
                    )
                ).astype(float)
                avg_std_dict[dataset][language][metrics]["human_avg"] = np.mean(
                    human_results
                )
                avg_std_dict[dataset][language][metrics]["human_ci"] = (
                    np.std(human_results, ddof=1) / np.sqrt(len(human_results)) * 1.96
                )
                avg_std_dict[dataset][language][metrics]["llm_avg"] = np.mean(
                    llm_results
                )
                avg_std_dict[dataset][language][metrics]["llm_ci"] = (
                    np.std(llm_results, ddof=1) / np.sqrt(len(llm_results)) * 1.96
                )
marker_dict = {
    "omission": "o",
    "addition": "^",
    "word_meaning": "v",
    "fluency": "s",
    "question": "D",
}
marker_dict_jp = {
    "省略": "o",
    "付加": "^",
    "単語の意味": "v",
    "流暢性": "s",
    "疑問文の訳出": "D",
}
color_dict = {
    "vietnamese": colors[0],
    "chinese": colors[1],
    "english": colors[2],
}
color_dict_jp = {
    "ベトナム語": colors[0],
    "中国語": colors[1],
    "英語": colors[2],
}
linestyle_dict = {
    "vietnamese": "-",
    "chinese": "--",
    "english": ":",
}
linestyle_dict_jp = {
    "ベトナム語": "-",
    "中国語": "--",
    "英語": ":",
}
figure, axes = plt.subplots(1, 2, figsize=(6.5, 3))
for dataset, language_dict in avg_std_dict.items():
    ax = axes[0] if dataset == "handbook" else axes[1]
    if dataset == "handbook":
        ax.set_title("ハンドブックデータセット")
    elif dataset == "question":
        ax.set_title("疑問文データセット")
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1], ["人手評価", "LLM-as-a-Judge"])
    for language, metrics_dict in language_dict.items():
        for metrics, stats in metrics_dict.items():
            marker = marker_dict[metrics]
            color = color_dict[language]
            linestyle = linestyle_dict[language]
            ax.plot(
                [0, 1],
                [stats["human_avg"], stats["llm_avg"]],
                marker=marker,
                color=color,
                linestyle=linestyle,
                linewidth=2,
                alpha=0.8,
            )
            # ax.errorbar(
            #     0,
            #     stats["human_avg"],
            #     xerr=None,
            #     yerr=stats["human_ci"],
            #     capsize=5,
            #     color=color,
            #     fmt=marker,
            #     label=f"{dataset}-{language}-{metrics}",
            # )
            # ax.errorbar(
            #     1,
            #     stats["llm_avg"],
            #     xerr=None,
            #     yerr=stats["llm_ci"],
            #     capsize=5,
            #     color=color,
            #     fmt=marker,
            #     label=f"{dataset}-{language}-{metrics}",
            # )
    if dataset == "question":
        # 凡例用の Line2D オブジェクトを作成
        marker_handles = [
            Line2D(
                [0], [0], marker=marker, color="black", linestyle="None", label=label
            )
            for label, marker in marker_dict_jp.items()
        ]
        color_handles = [
            Line2D(
                [0],
                [0],
                color=color_dict_jp[label],
                label=label,
                linestyle=linestyle_dict_jp[label],
            )
            for label in color_dict_jp.keys()
        ]

        # 描画
        ax.legend(
            handles=marker_handles + color_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9,
        )
figure.tight_layout()
figure.savefig("../output/analysis/avg_human_llm.pdf")
