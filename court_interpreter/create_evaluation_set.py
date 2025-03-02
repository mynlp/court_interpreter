# Step 4: Create evaluation set, which is presented to human evaluators and LLM for evaluation
# Running this script generates `../output/translation/handbook_evaluation_set_{language}.tsv` and `../output/translation/question_evaluation_set.tsv`

import csv
import random

# === Handbook evaluation set ===
# Create language-wise translation dictionary
# {language: [{id, source, translation_gpt, translation_llama, translation_azure}]}
dataset_dict = {
    "english": [],
    "chinese": [],
    "vietnamese": [],
}
with (
    open("../output/translation/handbook_filtered-gpt.tsv") as gpt_f,
    open("../output/translation/handbook_filtered-llama.tsv") as llama_f,
    open("../output/translation/handbook_filtered-azure.tsv") as azure_f,
    open("../dataset/handbook_filtered.tsv") as original_f,
):
    gpt_reader = csv.DictReader(gpt_f, delimiter="\t")
    llama_reader = csv.DictReader(llama_f, delimiter="\t")
    azure_reader = csv.DictReader(azure_f, delimiter="\t")
    original_reader = csv.DictReader(original_f, delimiter="\t")
    for row_gpt, row_llama, row_azure, row_original in zip(
        gpt_reader, llama_reader, azure_reader, original_reader
    ):
        for language in ["english", "chinese", "vietnamese"]:
            if language == "english":
                source = row_original["target_chinese"]
            elif language == "chinese":
                source = row_original["source_japanese"]
            elif language == "vietnamese":
                source = row_original["target_english"]
            sentence_info = {
                "id": row_original["id"],
                "source": source,
                "target": row_original[f"target_{language}"],
                "translation_gpt": row_gpt[f"gpt_{language}"],
                "translation_llama": row_llama[f"llama_{language}"],
                "translation_azure": row_azure[f"azure_{language}"],
            }
            dataset_dict[language].append(sentence_info)


# Create random mappings of translations
# mappings = [{language: [{A: azure, B: gpt, C: target, D: llama}, {A: ...}, ...]
mappings = {}
for language in ["english", "chinese", "vietnamese"]:
    mapping_lang = []
    for i in range(len(dataset_dict[language])):
        translations = random.sample(
            [
                "translation_gpt",
                "translation_llama",
                "translation_azure",
                "target",
            ],
            4,
        )
        mapping_lang.append(
            {
                "A": translations[0],
                "B": translations[1],
                "C": translations[2],
                "D": translations[3],
            }
        )
    mappings[language] = mapping_lang


# Construct final evaluation set
for language in ["english", "chinese", "vietnamese"]:
    evaluation_set = []
    for sentence_info, mapping in zip(dataset_dict[language], mappings[language]):
        evaluation_set.append(
            {
                "id": sentence_info["id"],
                "source": sentence_info["source"],
                "translation_A": sentence_info[mapping["A"]],
                "translation_B": sentence_info[mapping["B"]],
                "translation_C": sentence_info[mapping["C"]],
                "translation_D": sentence_info[mapping["D"]],
                "mapping": mapping,
            }
        )
    with open(
        f"../output/translation/handbook_evaluation_set_{language}.tsv", "w"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "source",
                "translation_A",
                "translation_B",
                "translation_C",
                "translation_D",
                "mapping",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(evaluation_set)

# === Question evaluation set ===

# Create translation dictionary
# {id, source, translation_gpt, translation_llama, translation_azure}
dataset_q = []
with (
    open("../output/translation/question-gpt.tsv") as gpt_f,
    open("../output/translation/question-llama.tsv") as llama_f,
    open("../output/translation/question-azure.tsv") as azure_f,
    open("../dataset/question.tsv") as original_f,
):
    gpt_reader = csv.DictReader(gpt_f, delimiter="\t")
    llama_reader = csv.DictReader(llama_f, delimiter="\t")
    azure_reader = csv.DictReader(azure_f, delimiter="\t")
    original_reader = csv.DictReader(original_f, delimiter="\t")
    for row_gpt, row_llama, row_azure, row_original in zip(
        gpt_reader, llama_reader, azure_reader, original_reader
    ):
        sentence_info = {
            "id": row_original["id"],
            "source": row_original["source_japanese"],
            "translation_gpt": row_gpt["gpt_chinese"],
            "translation_llama": row_llama["llama_chinese"],
            "translation_azure": row_azure["azure_chinese"],
        }
        dataset_q.append(sentence_info)

# Create random mappings of translations
# mappings = {A: azure, B: gpt, C: llama}, {A: ...}, ...]
mappings_q = []
for i in range(len(dataset_q)):
    translations = random.sample(
        [
            "translation_gpt",
            "translation_llama",
            "translation_azure",
        ],
        3,
    )
    mappings_q.append(
        {
            "A": translations[0],
            "B": translations[1],
            "C": translations[2],
        }
    )

# Construct final evaluation set
evaluation_set = []
for sentence_info, mapping in zip(dataset_q, mappings_q):
    evaluation_set.append(
        {
            "id": sentence_info["id"],
            "source": sentence_info["source"],
            "translation_A": sentence_info[mapping["A"]],
            "translation_B": sentence_info[mapping["B"]],
            "translation_C": sentence_info[mapping["C"]],
            "mapping": mapping,
        }
    )
with open("../output/translation/question_evaluation_set.tsv", "w") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "id",
            "source",
            "translation_A",
            "translation_B",
            "translation_C",
            "mapping",
        ],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(evaluation_set)
