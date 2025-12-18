from collections import defaultdict

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from semopy import polycorr
from sklearn.metrics import confusion_matrix


# クラメールの連関係数を求める関数
def cramers_v(x, y):
    # 分割表を作成
    cont_table = pd.crosstab(x, y)
    # カイ二乗を求める
    chi2, pval = stats.chi2_contingency(cont_table, correction=False)[:2]

    min_d = min(cont_table.shape) - 1
    if min_d == 0:
        return np.nan, np.nan
    n = len(x)

    # クラメールの連関係数
    v = np.sqrt(chi2 / (min_d * n))

    # p値はカイ二乗検定のp値を返す
    return (v, pval)


confmat_dict = defaultdict(lambda: defaultdict(dict))
stats_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

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
                if metrics in ["word_meaning", "fluency"]:
                    human_results = human_results * 4
                    llm_results = llm_results * 4
                confmat_dict[metrics][dataset][language] = confusion_matrix(
                    human_results, llm_results
                )
                stats_dict[metrics][dataset][language]["pearson"] = stats.pearsonr(
                    human_results, llm_results
                )
                stats_dict[metrics][dataset][language]["spearman"] = stats.spearmanr(
                    human_results, llm_results
                )
                stats_dict[metrics][dataset][language]["kendall"] = stats.kendalltau(
                    human_results, llm_results
                )
                stats_dict[metrics][dataset][language]["cramers"] = cramers_v(
                    human_results, llm_results
                )
                stats_dict[metrics][dataset][language]["polycorr"] = (
                    polycorr.polychoric_corr(human_results, llm_results)
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
                if metrics in ["word_meaning", "fluency"]:
                    human_results = human_results * 4
                    llm_results = llm_results * 4
                elif metrics in ["question"]:
                    human_results = human_results * 2
                    llm_results = llm_results * 2
                confmat_dict[metrics][dataset][language] = confusion_matrix(
                    human_results, llm_results
                )
                stats_dict[metrics][dataset][language]["pearson"] = stats.pearsonr(
                    human_results, llm_results
                )
                stats_dict[metrics][dataset][language]["spearman"] = stats.spearmanr(
                    human_results, llm_results
                )
                stats_dict[metrics][dataset][language]["kendall"] = stats.kendalltau(
                    human_results, llm_results
                )
                stats_dict[metrics][dataset][language]["cramers"] = cramers_v(
                    human_results, llm_results
                )
                stats_dict[metrics][dataset][language]["polycorr"] = (
                    polycorr.polychoric_corr(human_results, llm_results)
                )

# 図を5枚作る
# Figure 1. 省略 (2x3)
# Figure 2. 追加 (2x3)
# Figure 3. 単語の意味 (2x3) -> サイズを2-3倍にする
# Figure 4. 流暢さ (2x3) -> サイズを2-3倍にする
# Figure 5. 疑問文 (1x3) -> サイズを1.5-2倍にする

map_language = {
    "vietnamese": "ベトナム語",
    "chinese": "中国語",
    "english": "英語",
}


for metrics, dataset_dict in confmat_dict.items():
    if metrics in ["word_meaning", "fluency"]:
        figure = plt.figure(figsize=(8, 6.5))
    elif metrics == "question":
        figure = plt.figure(figsize=(6.5, 2.5))
    else:
        figure = plt.figure(figsize=(5.5, 5))
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
            if conf_mat.shape[0] != 2:
                ax.set_xticklabels(range(1, conf_mat.shape[0] + 1))
                ax.set_yticklabels(range(1, conf_mat.shape[0] + 1))
            ax.invert_yaxis()
            ax.set_title(map_language[language])
            ax.set_xlabel("LLM-as-a-Judge")
            if language == "vietnamese":
                ax.set_ylabel("人手評価")
            plt.yticks(rotation=0)
    figure.tight_layout(h_pad=3.0)
    figure.savefig(f"../output/analysis/heatmap_human_llm/heatmap_{metrics}.pdf")

with open("../output/analysis/heatmap_human_llm/statistics.csv", mode="w") as f:
    f.write(
        "metrics,dataset,language,pearson,pearson_pval,spearman,spearman_pval,kendall,kendall_pval,cramers,cramers_pval,polycorr\n"
    )
    for metrics, dataset_dict in stats_dict.items():
        for dataset, language_dict in dataset_dict.items():
            for language, value_dict in language_dict.items():
                f.write(f"{metrics},{dataset},{language},")
                for stat_name, value in value_dict.items():
                    if stat_name == "polycorr":
                        f.write(f"{value:.4f}\n")
                    else:
                        f.write(f"{value[0]:.4f},{value[1]:.4f},")
