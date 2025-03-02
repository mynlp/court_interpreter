# Step 2. Arrange the structured dataset to a TSV file
# Running this script generates `../dataset/handbook.tsv`
# Note: `../dataset/handbook_filtered.tsv` is manually created by removing easy or duplicated sentences.
# Note: Examples in `../dataset/question.tsv` is generated with LLM.

import re


def sanitize_line(line: str):
    m = re.match(r"^\t\t\t\t\((.+?)\) (.+)", line)
    assert isinstance(m, re.Match), line
    content = m.group(2)
    # Remove bullets
    if content.startswith("• "):
        content = content.replace("• ", "", 1)
    if content.startswith("- "):
        content = content.replace("- ", "", 1)
    if content.startswith("・ "):
        content = content.replace("・ ", "", 1)
    return content


def extract_sentences(path: str) -> list[str]:
    sentences: list[str] = []
    with open(path) as f:
        for line in f:
            if line.startswith("\t\t\t\t("):
                sentences.append(sanitize_line(line))
    return sentences


sentences_map: dict[str, list[str]] = {}
for language in ["japanese", "chinese", "english", "vietnamese"]:
    aligned_path = f"../dataset_pre/{language}_aligned.txt"
    sentences_map[language] = extract_sentences(aligned_path)

assert (
    len(sentences_map["japanese"])
    == len(sentences_map["english"])
    == len(sentences_map["chinese"])
    == len(sentences_map["vietnamese"])
)

dataset: list[dict[str, str]] = []
for id, ja, en, zh, vi in zip(
    range(1, len(sentences_map["japanese"]) + 1),
    sentences_map["japanese"],
    sentences_map["english"],
    sentences_map["chinese"],
    sentences_map["vietnamese"],
):
    dataset.append(
        {
            "id": str(id).zfill(3),
            "japanese": ja,
            "english": en,
            "chinese": zh.replace("、", "，"),
            "vietnamese": vi,
        }
    )

# Save the dataset
csv_path = "../dataset/handbook.tsv"
with open(csv_path, "w") as f:
    f.write("id\tsource_japanese\ttarget_english\ttarget_chinese\ttarget_vietnamese\n")
    for data in dataset:
        f.write(
            f"handbook-{data['id']}\t{data['japanese']}\t{data['english']}\t{data['chinese']}\t{data['vietnamese']}\n"
        )
