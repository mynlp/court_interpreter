import re

import pandas as pd


class Statistics:
    def __init__(self):
        self.sentences_num = 0
        self.terms_num_list = []
        self.sentence_length_list = []  # 文字数

    def update(self, sentence, terms):
        self.sentences_num += 1
        terms_num = sum([1 for term in terms if term in sentence])
        self.terms_num_list.append(terms_num)
        self.sentence_length_list.append(len(sentence))


terms = []
with open("../dataset_raw/term.txt") as f:
    for line in f:
        term = re.sub(r"（.*）", "", line.strip())
        terms.append(term)

statistics = {"handbook": Statistics(), "question": Statistics()}
for dataset in ["handbook", "question"]:
    language = "english"
    human_df = pd.read_csv(
        f"../output/evaluation/human/expert/remapped_{dataset}_evaluation_set_{language}.csv"
    )
    translation_df = pd.read_csv(
        f"../output/translation/{dataset}_evaluation_set_{language}.tsv", sep="\t"
    )
    translation_df = translation_df[translation_df["id"].isin(human_df["id"])]
    source = translation_df["source"].tolist()
    for sentence in source:
        statistics[dataset].update(sentence, terms)

for dataset, stat in statistics.items():
    print(
        f"{dataset}\t{stat.sentences_num}\t{sum(stat.terms_num_list) / len(stat.terms_num_list):.2f}\t{sum(stat.sentence_length_list) / len(stat.sentence_length_list):.2f}"
    )
