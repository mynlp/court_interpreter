# Step 3: Translate the structured dataset to other languages
# Running this script generates `../output/translation/{dataset_name}-{system}.tsv`

import argparse
import csv
import uuid
from logging import Logger

import requests
from groq import Groq
from openai import OpenAI
from tqdm import tqdm

from court_interpreter.utils import get_logger

language_map = {
    "english": "英語",
    "vietnamese": "ベトナム語",
    "chinese": "中国語（簡体字）",
}

system_prompt = (
    "あなたは法廷通訳を行う通訳士です。法廷におけるやり取りを正確に翻訳してください。"
)


def translate_with_client(client, model: str, language: str, sentence: str) -> str:
    user_prompt = (
        f"以下の日本語の文を{language_map[language]}に翻訳してください。"
        f"意訳は避け、単語の省略や付加をしない逐語訳で訳すようにしてください。"
        f"疑問文はニュアンスが変わらないように注意して訳してください。"
        "回答は必ず一行になるようにして下さい。\n"
        f"{sentence}\n"
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content


def translate_with_gpt(
    organization: str,
    api_key: str,
    source_sentences: list[str],
    language: str,
    logger=None,
) -> list[str]:
    client = OpenAI(organization=organization, api_key=api_key)

    translated_dataset: list[str] = []
    for sentence in tqdm(source_sentences):
        translated_dataset.append(
            translate_with_client(client, "gpt-4o", language, sentence)
        )
    return translated_dataset


def translate_with_llama(
    api_key: str,
    source_sentences: list[str],
    language: str,
    logger=None,
) -> list[str]:
    client = Groq(api_key=api_key)
    translated_dataset: list[str] = []
    for sentence in tqdm(source_sentences):
        translated_dataset.append(
            translate_with_client(client, "llama-3.1-70b-versatile", language, sentence)
        )
    return translated_dataset


def translate_with_azure(
    api_key: str,
    source_sentences: list[str],
    logger=None,
) -> tuple[list[str], list[str], list[str]]:
    location = "japaneast"
    url = "https://api.cognitive.microsofttranslator.com/translate"
    params = {"api-version": "3.0", "from": "ja", "to": ["zh-Hans", "en", "vi"]}
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Ocp-Apim-Subscription-Region": location,
        "Content-type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4()),
    }
    english_dataset: list[str] = []
    chinese_dataset: list[str] = []
    vietnamese_dataset: list[str] = []
    for sentence in tqdm(source_sentences):
        body = [{"text": sentence}]
        request = requests.post(url, params=params, headers=headers, json=body)
        response = request.json()
        for translations in response:
            for translation in translations["translations"]:
                if translation["to"] == "en":
                    english_dataset.append(translation["text"])
                elif translation["to"] == "zh-Hans":
                    chinese_dataset.append(translation["text"])
                elif translation["to"] == "vi":
                    vietnamese_dataset.append(translation["text"])
    return english_dataset, chinese_dataset, vietnamese_dataset


def main(args: argparse.Namespace, logger: Logger) -> None:
    sentences_map: dict[str, list[str]] = {}
    # Load dataset (tsv)
    with open(args.dataset_path) as f:
        dataset = csv.DictReader(f, delimiter="\t")
        ids, source_sentences = [], []
        for row in dataset:
            ids.append(row["id"])
            source_sentences.append(row["source_japanese"])
        sentences_map["japanese"] = source_sentences
    logger.info(f"Dataset size (Japanese)\t: {len(source_sentences)}")
    logger.info("Start translation")
    if args.system == "gpt" or args.system == "llama":
        for language in ["english", "chinese", "vietnamese"]:
            if args.system == "gpt":
                translated_sentences = translate_with_gpt(
                    args.gpt_organization, args.api_key, source_sentences, language
                )
            elif args.system == "llama":
                translated_sentences = translate_with_llama(
                    args.api_key, source_sentences, language
                )
            logger.info(
                f"Translated ({language.capitalize()})\t: {len(translated_sentences)}"
            )
            assert len(source_sentences) == len(translated_sentences)
            sentences_map[language] = translated_sentences
    elif args.system == "azure":
        english, chinese, vietnamese = translate_with_azure(
            args.api_key, source_sentences
        )
        sentences_map["english"] = english
        sentences_map["chinese"] = chinese
        sentences_map["vietnamese"] = vietnamese

    results = []
    for id, ja, en, zh, vi in zip(
        ids,
        sentences_map["japanese"],
        sentences_map["english"],
        sentences_map["chinese"],
        sentences_map["vietnamese"],
    ):
        results.append(
            {
                "id": id,
                "japanese": ja,
                "english": en,
                "chinese": zh,
                "vietnamese": vi,
            }
        )
    with open(args.csv_path, "w") as f:
        f.write(
            f"id\tsource_japanese\t{args.system}_english\t{args.system}_chinese\t{args.system}_vietnamese\n"
        )
        for data in results:
            f.write(
                f"{data['id']}\t{data['japanese']}\t{data['english']}\t{data['chinese']}\t{data['vietnamese']}\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_organization")
    parser.add_argument("--api_key")
    parser.add_argument("--dataset_path", default="../dataset/handbook_filtered.tsv")
    parser.add_argument("--system", default="gpt", choices=["gpt", "llama", "azure"])
    parser.add_argument(
        "--csv_path", default="../output/translation/handbook_filtered-gpt.tsv"
    )
    args = parser.parse_args()
    logger = get_logger()
    main(args, logger)
