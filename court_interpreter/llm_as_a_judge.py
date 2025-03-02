# Step 5-1: Evaluate the translations with LLM
# Running this script generates `../output/evaluation/llm/{evaluation_set_name}.tsv`

import argparse
import os
import time
from logging import Logger
from pathlib import Path

import pandas as pd
from openai import APIConnectionError, OpenAI
from tqdm import tqdm

from court_interpreter.utils import get_logger

source_map = {
    "english": "中国語（簡体字）",
    "vietnamese": "英語",
    "chinese": "日本語",
}
language_map = {
    "english": "英語",
    "vietnamese": "ベトナム語",
    "chinese": "中国語（簡体字）",
}


def create_system_prompt(
    source_language: str, translation_language: str, question: str, score_format: str
):
    if source_language == "日本語":
        note = "但し、原文（日本語）で主語が省略されている場合に翻訳文で主語が追加されている部分については付加とみなしません。（日本語では主語が省略されやすいため）\n"
    else:
        note = ""
    return (
        "あなたは法廷通訳を行う通訳士です。\n"
        f"{source_language}の文と、それを{translation_language}に翻訳した4つの文章A,B,C,Dを与えるので、"
        "それぞれの翻訳性能を評価してください。\n"
        "評価は以下の四つの基準に従って行ってください。\n"
        "1. 省略 (Omission)\n"
        "翻訳された文中において、原文にある情報が欠落しているかどうかを評価します。\n"
        "単語レベルで訳の省略があるか否かについて評価を行って下さい。\n"
        "評価基準: \n"
        "0：単語レベルでの省略がある。\n"
        "1：単語レベルでの省略がない。"
        "2. 付加 (Addition)\n"
        "\n"
        "翻訳された文中において、原文にない情報が付加されているかどうかを評価します。\n"
        "単語レベルでの訳の付加があるか否かについて評価を行って下さい。\n"
        f"{note}"
        "評価基準: \n"
        "0：単語レベルでの付加がある。\n"
        "1：単語レベルでの付加がない。\n"
        "\n"
        "3. 単語の意味 (Word Meaning)\n"
        "省略や付加がされていない各単語やフレーズについて、"
        "それぞれ適切に翻訳されているか、意味が変わっていないかを評価します。"
        "ただし、文全体として誤訳になっている場合も「0」と評価して下さい。\n"
        "評価基準: \n"
        "0：単語の意味の誤りがある。または、文全体として誤訳になっている。\n"
        "1：単語の意味が誤りがない。\n"
        "\n"
        "4. 流暢性 (Fluency)\n"
        "翻訳された文が文法的および自然な言葉で表現され、読みやすいかどうかを評価します。\n"
        "ただし、翻訳内容の正しさは考慮せず、純粋に文自体の流暢性を評価して下さい。\n"
        "他の言語の文字が含まれている場合は、流暢性が低いとみなして減点して下さい。\n"
        "評価基準: \n"
        "1：著しく不自然で理解しにくい。\n"
        "2：不自然であり、全体的にぎこちない。\n"
        "3：理解は可能だが、どこか不自然。\n"
        "4：ほぼ自然である。\n"
        "5：非常に自然で流暢である。\n"
        "\n"
        f"{question}"
        "回答は以下のフォーマットに従って一行で出力してください\n"
        f"{score_format}"
    )


def judge(
    organization: str,
    api_key: str,
    ids: list[str],
    sources: list[str],
    translations: list[list[str]],
    language: str,
    question: bool,
    logger: Logger,
) -> list[str]:
    scores: list[str] = []
    client = OpenAI(organization=organization, api_key=api_key)
    source_language = source_map[language]
    translation_language = language_map[language]
    if question:
        question_prompt = (
            "5. 質問 (Question)\n"
            "翻訳された文が原文に対して適切な質問を引き起こすかどうかを評価します。\n"
            "翻訳された文が原文に対して適切な質問を引き起こすか否かについて評価を行って下さい。\n"
            "評価基準: \n"
            "0：翻訳された文が原文に対して適切な質問を引き起こさない。\n"
            "1：翻訳された文が原文に対して適切な質問を引き起こす。\n"
        )
        score_format = (
            "id,"
            "Aのomissionスコア,Aのadditionスコア,Aのword_meaningスコア,AのQuestionスコア,Aのfluencyスコア"
            "Bのomissionスコア,Bのadditionスコア,Bのword_meaningスコア,BのQuestionスコア,Bのfluencyスコア"
            "Cのomissionスコア,Cのadditionスコア,Cのword_meaningスコア,CのQuestionスコア,Cのfluencyスコア\n"
            "例：handbook001,1,0,1,1,5,1,1,1,0,4,0,0,1,1,5\n"
        )
    else:
        question_prompt = ""
        score_format = (
            "id,"
            "Aのomissionスコア,Aのadditionスコア,Aのword_meaningスコア,Aのfluencyスコア"
            "Bのomissionスコア,Bのadditionスコア,Bのword_meaningスコア,Bのfluencyスコア"
            "Cのomissionスコア,Cのadditionスコア,Cのword_meaningスコア,Cのfluencyスコア"
            "Dのomissionスコア,Dのadditionスコア,Dのword_meaningスコア,Dのfluencyスコア\n"
            "例：handbook001,1,0,1,5,1,1,1,4,0,0,1,5,1,1,1,5\n"
        )
    system_prompt = create_system_prompt(
        source_language, translation_language, question_prompt, score_format
    )
    # logger.info(system_prompt)
    for i in tqdm(range(len(ids))):
        id = ids[i]
        s = sources[i]
        user_prompt = f"id: {id}\n{source_language}の文: {s}\n"
        alphabets = ["A", "B", "C", "D"]
        for j in range(len(translations)):
            user_prompt += (
                f"{translation_language}の文{alphabets[j]}: {translations[j][i]}\n"
            )
        for _ in range(5):  # 最大5回リトライ
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                break
            except APIConnectionError:
                time.sleep(2)  # avoid rate limit error
        scores.append(completion.choices[0].message.content.strip())
    return scores


def write_scores(scores: list[str], evaluation_path: str, question: bool) -> None:
    if question:
        columns = (
            "id,omission_A,addition_A,word_meaning_A,question_A,fluency_A,"
            "omission_B,addition_B,word_meaning_B,question_B,fluency_B,"
            "omission_C,addition_C,word_meaning_C,question_C,fluency_C\n"
        )
        score_num = 16
    else:
        columns = (
            "id,omission_A,addition_A,word_meaning_A,fluency_A,omission_B,addition_B,word_meaning_B,fluency_B,"
            "omission_C,addition_C,word_meaning_C,fluency_C,omission_D,addition_D,word_meaning_D,fluency_D\n"
        )
        score_num = 17
    with open(evaluation_path, "w") as f:
        f.write(columns)
        for score in scores:
            assert len(score.split(",")) == score_num
            f.write(f"{score}\n")


def main(args: argparse.Namespace, logger: Logger) -> None:
    evaluation_path = f"{args.output_dir}/{Path(args.evaluation_set_path).name}"
    df = pd.read_csv(args.evaluation_set_path, sep="\t")
    # id  source  translation_A	translation_B	translation_C	translation_D	mapping
    if args.question:
        scores = judge(
            args.gpt_organization,
            args.api_key,
            list(df["id"]),
            list(df["source"]),
            [
                list(df["translation_A"]),
                list(df["translation_B"]),
                list(df["translation_C"]),
            ],
            args.language,
            args.question,
            logger,
        )
    else:
        scores = judge(
            args.gpt_organization,
            args.api_key,
            list(df["id"]),
            list(df["source"]),
            [
                list(df["translation_A"]),
                list(df["translation_B"]),
                list(df["translation_C"]),
                list(df["translation_D"]),
            ],
            args.language,
            args.question,
            logger,
        )
    assert len(scores) == len(df)
    write_scores(scores, evaluation_path, args.question)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_organization")
    parser.add_argument("--api_key")
    parser.add_argument(
        "--language", default="chinese", choices=["english", "vietnamese", "chinese"]
    )
    parser.add_argument(
        "--question", type=bool, default=False, help="whether to evaluate question"
    )
    parser.add_argument(
        "--evaluation_set_path",
        default="../output/translation/handbook_evaluation_set_chinese.tsv",
    )
    parser.add_argument("--output_dir", default="../output/evaluation/llm")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger()
    main(args, logger)
