# Step 6: Remap the evaluation set from A,B,C,D to target, gpt, llama, azure
# Running this script generates `../output/evaluation/llm/remapped_{evaluation_set}.csv` and `../output/evaluation/human/remapped_{evaluation_set}.csv`

import ast
import csv

import pandas as pd


def remap_columns_handbook(
    input_csv: str, output_csv: str, mappings: list[dict[str, str]]
):
    with open(input_csv, mode="r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
    remapped_rows: list[dict[str, str]] = []
    for row, mapping in zip(rows, mappings):
        new_dict = {"id": row["id"]}
        for X, name in mapping.items():  # A, translation_gpt
            if "_" in name:
                name = name.split("_")[1]
            keys = [
                f"omission_{X}",
                f"addition_{X}",
                f"word_meaning_{X}",
                f"fluency_{X}",
            ]
            for key in keys:
                new_name = f"{key.rsplit('_', 1)[0]}_{name}"
                new_dict[new_name] = row[key]
        remapped_rows.append(new_dict)
    with open(output_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            [
                "id",
                "omission_target",
                "addition_target",
                "word_meaning_target",
                "fluency_target",
                "omission_gpt",
                "addition_gpt",
                "word_meaning_gpt",
                "fluency_gpt",
                "omission_llama",
                "addition_llama",
                "word_meaning_llama",
                "fluency_llama",
                "omission_azure",
                "addition_azure",
                "word_meaning_azure",
                "fluency_azure",
            ],
            delimiter=",",
        )
        writer.writeheader()
        writer.writerows(remapped_rows)


def remap_columns_question(
    input_csv: str, output_csv: str, mappings: list[dict[str, str]]
):
    with open(input_csv, mode="r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
    remapped_rows: list[dict[str, str]] = []
    for row, mapping in zip(rows, mappings):
        new_dict = {"id": row["id"]}
        for X, name in mapping.items():  # A, translation_gpt
            if "_" in name:
                name = name.split("_")[1]
            keys = [
                f"omission_{X}",
                f"addition_{X}",
                f"word_meaning_{X}",
                f"fluency_{X}",
                f"question_{X}",
            ]
            for key in keys:
                new_name = f"{key.rsplit('_', 1)[0]}_{name}"
                new_dict[new_name] = row[key]
        remapped_rows.append(new_dict)
    with open(output_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            [
                "id",
                "omission_gpt",
                "addition_gpt",
                "word_meaning_gpt",
                "question_gpt",
                "fluency_gpt",
                "omission_llama",
                "addition_llama",
                "word_meaning_llama",
                "question_llama",
                "fluency_llama",
                "omission_azure",
                "addition_azure",
                "word_meaning_azure",
                "question_azure",
                "fluency_azure",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(remapped_rows)


for language in ["chinese", "english", "vietnamese"]:
    ### handbook
    df = pd.read_csv(  # type: ignore
        f"../output/translation/handbook_evaluation_set_{language}.tsv", sep="\t"
    )
    mapping_list: list[str] = df["mapping"].to_list()
    mappings: list[dict[str, str]] = [ast.literal_eval(m) for m in mapping_list]
    # llm
    remap_columns_handbook(
        f"../output/evaluation/llm/handbook_evaluation_set_{language}.csv",
        f"../output/evaluation/llm/remapped_handbook_evaluation_set_{language}.csv",
        mappings,
    )
    # human
    remap_columns_handbook(
        f"../output/evaluation/human/handbook_evaluation_set_{language}.csv",
        f"../output/evaluation/human/remapped_handbook_evaluation_set_{language}.csv",
        mappings,
    )
    ### question
    df = pd.read_csv(  # type: ignore
        f"../output/translation/question_evaluation_set_{language}.tsv", sep="\t"
    )
    mapping_list: list[str] = df["mapping"].to_list()
    mappings: list[dict[str, str]] = [ast.literal_eval(m) for m in mapping_list]
    # llm
    remap_columns_question(
        f"../output/evaluation/llm/question_evaluation_set_{language}.csv",
        f"../output/evaluation/llm/remapped_question_evaluation_set_{language}.csv",
        mappings,
    )
    # human
    remap_columns_question(
        f"../output/evaluation/human/question_evaluation_set_{language}.csv",
        f"../output/evaluation/human/remapped_question_evaluation_set_{language}.csv",
        mappings,
    )
