from collections import defaultdict

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# 区間の境界値（bins）とラベル（labels）
bins = [
    0,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
]
labels = [
    "0",
    "0.05",
    "0.1",
    "0.15",
    "0.2",
    "0.25",
    "0.3",
    "0.35",
    "0.4",
    "0.45",
    "0.5",
    "0.55",
    "0.6",
    "0.65",
    "0.7",
    "0.75",
    "0.8",
    "0.85",
    "0.9",
    "0.95",
]

df_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
stats_dict = defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
)

dataset = "handbook"
for language in ["vietnamese", "chinese", "english"]:
    human_df = pd.read_csv(
        f"../output/evaluation/human/expert/remapped_{dataset}_evaluation_set_{language}.csv"
    )
    bertscore_df = pd.read_csv(
        f"../output/evaluation/bertscore/{dataset}_evaluation_set_{language}.tsv",
        sep="\t",
    )
    bleu_df = pd.read_csv(
        f"../output/evaluation/bleu/{dataset}_evaluation_set_{language}.tsv",
        sep="\t",
    )
    comet_df = pd.read_csv(
        f"../output/evaluation/comet/{dataset}_evaluation_set_{language}.tsv",
        sep="\t",
    )
    comet_ref_free_df = pd.read_csv(
        f"../output/evaluation/comet_ref_free/{dataset}_evaluation_set_{language}.tsv",
        sep="\t",
    )
    # filter out
    bertscore_df = bertscore_df[bertscore_df["ID"].isin(human_df["id"])]
    bleu_df = bleu_df[bleu_df["ID"].isin(human_df["id"])]
    comet_df = comet_df[comet_df["ID"].isin(human_df["id"])]
    comet_ref_free_df = comet_ref_free_df[comet_ref_free_df["ID"].isin(human_df["id"])]
    bertscore_results = pd.concat(
        [
            bertscore_df["gpt_f1"],
            bertscore_df["llama_f1"],
            bertscore_df["azure_f1"],
        ]
    )
    bleu_results = pd.concat(
        [
            bleu_df["gpt"] / 100,
            bleu_df["llama"] / 100,
            bleu_df["azure"] / 100,
        ]
    )
    comet_results = pd.concat(
        [
            comet_df["gpt"],
            comet_df["llama"],
            comet_df["azure"],
        ]
    )
    comet_ref_free_results = pd.concat(
        [
            comet_ref_free_df["gpt"],
            comet_ref_free_df["llama"],
            comet_ref_free_df["azure"],
        ]
    )
    human_df = human_df.astype(str)
    for metrics in ["omission", "addition", "word_meaning", "fluency"]:
        human_results = pd.concat(
            [
                human_df[f"{metrics}_gpt"],
                human_df[f"{metrics}_llama"],
                human_df[f"{metrics}_azure"],
            ]
        )
        for eval, results in [
            ("bertscore", bertscore_results),
            ("bleu", bleu_results),
            ("comet", comet_results),
            ("comet_ref_free", comet_ref_free_results),
        ]:
            # 連続値をそのまま使って相関係数を計算
            stats_dict[eval][metrics][dataset][language]["spearman"] = stats.spearmanr(
                human_results, results
            )
            stats_dict[eval][metrics][dataset][language]["kendall"] = stats.kendalltau(
                human_results, results
            )
            # 連続値を区間に分けてクロス集計表を作成
            results_mapped = pd.cut(results, bins=bins, labels=labels).astype(str)
            crosstab = pd.crosstab(list(human_results), results_mapped)
            # 0-1.0まで0.05刻みの列を補完
            new_columns = np.arange(0, 1.00, 0.05).round(2).astype(str).tolist()
            # 新しい列でDataFrameを再作成
            new_df = pd.DataFrame(index=crosstab.index, columns=new_columns)
            # 元のデータを新しいDataFrameにコピー
            for col in crosstab.columns:
                new_df[col] = crosstab[col]
            df_dict[eval][metrics][dataset][language] = new_df.fillna(0)

dataset = "question"
for language in ["vietnamese", "chinese", "english"]:
    human_df = pd.read_csv(
        f"../output/evaluation/human/expert/remapped_{dataset}_evaluation_set_{language}.csv"
    )
    comet_ref_free_df = pd.read_csv(
        f"../output/evaluation/comet_ref_free/{dataset}_evaluation_set_{language}.tsv",
        sep="\t",
    )
    # filter out
    comet_ref_free_df = comet_ref_free_df[comet_ref_free_df["ID"].isin(human_df["id"])]
    comet_ref_free_results = pd.concat(
        [
            comet_ref_free_df["gpt"],
            comet_ref_free_df["llama"],
            comet_ref_free_df["azure"],
        ]
    )
    human_df = human_df.astype(str)
    for metrics in ["omission", "addition", "word_meaning", "question", "fluency"]:
        human_results = pd.concat(
            [
                human_df[f"{metrics}_gpt"],
                human_df[f"{metrics}_llama"],
                human_df[f"{metrics}_azure"],
            ]
        )
        for eval, results in [
            ("comet_ref_free", comet_ref_free_results),
        ]:
            # 連続値をそのまま使って相関係数を計算
            stats_dict[eval][metrics][dataset][language]["spearman"] = stats.spearmanr(
                human_results, results
            )
            stats_dict[eval][metrics][dataset][language]["kendall"] = stats.kendalltau(
                human_results, results
            )
            # 連続値を区間に分けてクロス集計表を作成
            results_mapped = pd.cut(results, bins=bins, labels=labels).astype(str)
            crosstab = pd.crosstab(list(human_results), results_mapped)
            # 0-1.0まで0.05刻みの列を補完
            new_columns = np.arange(0, 1.00, 0.05).round(2).astype(str).tolist()
            # 新しい列でDataFrameを再作成
            new_df = pd.DataFrame(index=crosstab.index, columns=new_columns)
            # 元のデータを新しいDataFrameにコピー
            for col in crosstab.columns:
                new_df[col] = crosstab[col]
            df_dict[eval][metrics][dataset][language] = new_df.fillna(0)


# bertscore, bleu, comet: 図を4枚作る。handbook なので色は全て Blues とする
# Figure 1. 省略 (1x3)
# Figure 2. 追加 (1x3)
# Figure 3. 単語の意味 (1x3) -> サイズを2-3倍にする
# Figure 4. 流暢さ (1x3) -> サイズを2-3倍にする

# comet_ref_free: 図を5枚作る
# Figure 1. 省略 (2x3)
# Figure 2. 追加 (2x3)
# Figure 3. 単語の意味 (2x3) -> サイズを2-3倍にする
# Figure 4. 流暢さ (2x3) -> サイズを2-3倍にする
# Figure 5. 疑問文 (1x3) -> サイズをxx倍にする

map_language = {
    "vietnamese": "ベトナム語",
    "chinese": "中国語",
    "english": "英語",
}

for eval, other_info in df_dict.items():
    for metrics, dataset_dict in other_info.items():
        if eval != "comet_ref_free":
            if metrics in ["word_meaning", "fluency"]:
                figure = plt.figure(figsize=(10, 2.2))
            else:
                figure = plt.figure(figsize=(10, 1.5))
        else:
            if metrics in ["word_meaning", "fluency"]:
                figure = plt.figure(figsize=(10, 4))
            elif metrics == "question":
                figure = plt.figure(figsize=(10, 1.75))
            else:
                figure = plt.figure(figsize=(10, 3))
        for i, (dataset, language_dict) in enumerate(dataset_dict.items()):
            for j, (language, cross_tab) in enumerate(language_dict.items()):
                ax = figure.add_subplot(
                    len(dataset_dict),
                    len(language_dict),
                    i * len(language_dict) + j + 1,
                )
                if dataset == "handbook":
                    sns.heatmap(
                        cross_tab,
                        cmap="Blues",
                        ax=ax,
                        cbar=False,
                    )
                else:
                    sns.heatmap(
                        cross_tab,
                        cmap="Greens",
                        ax=ax,
                        cbar=False,
                    )
                ax.invert_yaxis()
                ax.set_title(map_language[language])
                ax.set_xticks([v + 0.5 for v in range(len(labels))])
                ax.set_xticklabels(
                    ["" if i % 2 else str(i / 20) for i in range(len(labels))],
                    rotation=45,
                )
                if cross_tab.shape[0] != 2:
                    ax.set_yticklabels(range(1, cross_tab.shape[0] + 1))
                if eval == "comet_ref_free":
                    ax.set_xlabel("COMET-RF")
                if language == "vietnamese":
                    ax.set_ylabel("人手評価")
                else:
                    ax.set_ylabel("")
                plt.yticks(rotation=0)
        figure.tight_layout(h_pad=3.0)
        figure.savefig(f"../output/analysis/heatmap_human_{eval}/heatmap_{metrics}.pdf")

for eval, other_info in stats_dict.items():
    with open(f"../output/analysis/heatmap_human_{eval}/statistics.csv", mode="w") as f:
        f.write(
            "metrics,dataset,language,spearman,spearman_pval,kendall,kendall_pval\n"
        )
        for metrics, dataset_dict in other_info.items():
            for dataset, language_dict in dataset_dict.items():
                for language, value_dict in language_dict.items():
                    f.write(f"{metrics},{dataset},{language},")
                    stat_lines = ""
                    for stat_name, value in value_dict.items():
                        stat_lines += f"{value[0]:.4f},{value[1]:.4f},"
                    # remove last comma and add newline
                    f.write(f"{stat_lines[:-1]}\n")
