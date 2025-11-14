from collections import defaultdict

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

df_dict = defaultdict(lambda: defaultdict(dict))

for dataset in ["handbook", "question"]:
    for language in ["vietnamese", "chinese", "english"]:
        human_df = pd.read_csv(
            f"../output/evaluation/human/expert/remapped_{dataset}_evaluation_set_{language}.csv"
        )
        llm_df = pd.read_csv(
            f"../output/evaluation/llm/remapped_{dataset}_evaluation_set_{language}.csv"
        )
        llm_df = llm_df[llm_df["id"].isin(human_df["id"])]
        human_df = human_df.astype(str)
        llm_df = llm_df.astype(str)
        for metrics in ["omission", "addition", "word_meaning", "fluency"]:
            if dataset == "handbook":
                for metrics in ["omission", "addition", "word_meaning", "fluency"]:
                    human_results = pd.concat(
                        [
                            human_df[f"{metrics}_target"],
                            human_df[f"{metrics}_gpt"],
                            human_df[f"{metrics}_llama"],
                            human_df[f"{metrics}_azure"],
                        ]
                    )
                    llm_results = pd.concat(
                        [
                            llm_df[f"{metrics}_target"],
                            llm_df[f"{metrics}_gpt"],
                            llm_df[f"{metrics}_llama"],
                            llm_df[f"{metrics}_azure"],
                        ]
                    )
                    conf_mat = confusion_matrix(human_results, llm_results)
                    df_dict[metrics][dataset][language] = conf_mat
            else:
                for metrics in [
                    "omission",
                    "addition",
                    "word_meaning",
                    "question",
                    "fluency",
                ]:
                    human_results = pd.concat(
                        [
                            human_df[f"{metrics}_gpt"],
                            human_df[f"{metrics}_llama"],
                            human_df[f"{metrics}_azure"],
                        ]
                    )
                    llm_results = pd.concat(
                        [
                            llm_df[f"{metrics}_gpt"],
                            llm_df[f"{metrics}_llama"],
                            llm_df[f"{metrics}_azure"],
                        ]
                    )
                    conf_mat = confusion_matrix(human_results, llm_results)
                    df_dict[metrics][dataset][language] = conf_mat

# 図を5枚作る
# Figure 1. 省略 (2x3)
# Figure 2. 追加 (2x3)
# Figure 3. 単語の意味 (2x3) -> サイズを2-3倍にする
# Figure 4. 流暢さ (2x3) -> サイズを2-3倍にする
# Figure 5. 疑問文 (1x3) -> サイズを1.5-2倍にする


for metrics, dataset_dict in df_dict.items():
    if metrics in ["word_meaning", "fluency"]:
        figure = plt.figure(figsize=(8, 6.5))
    elif metrics == "question":
        figure = plt.figure(figsize=(7, 2.5))
    else:
        figure = plt.figure(figsize=(8, 6))
    for i, (dataset, language_dict) in enumerate(dataset_dict.items()):
        for j, (language, conf_mat) in enumerate(language_dict.items()):
            ax = figure.add_subplot(
                len(dataset_dict), len(language_dict), i * len(language_dict) + j + 1
            )
            if dataset == "handbook":
                sns.heatmap(
                    conf_mat,
                    cmap="Blues",
                    annot=True,
                    fmt="d",
                    ax=ax,
                    cbar=False,
                )
            else:
                sns.heatmap(
                    conf_mat,
                    cmap="Greens",
                    annot=True,
                    fmt="d",
                    ax=ax,
                    cbar=False,
                )
            ax.invert_yaxis()
            ax.set_title(f"{dataset.capitalize()} - {language.capitalize()}")
            ax.set_xlabel("LLM-as-a-Judge")
            ax.set_ylabel("人手評価")
            plt.yticks(rotation=0)
    figure.tight_layout(h_pad=3.0)
    figure.savefig(f"../output/analysis/heatmap_human_llm/heatmap_{metrics}.pdf")
