import json
import os
import string

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = "あなたは法廷通訳を行う通訳士です。"

USER_PROMPT_TRANSLATE = string.Template(
    "法廷での日本語の発話文（翻訳対象文）をベトナム語に翻訳して下さい。"
    "翻訳対象文が発話されるまでの文脈、翻訳対象文の発話者、翻訳対象文の発話対象を併せて与えるので、それらを踏まえて翻訳して下さい。"
    "意訳は避け、単語の省略や付加をしない逐語訳で訳すようにしてください。"
    "回答は必ず一行になるようにして下さい。\n"
    "翻訳対象文: ${target}\n"
    "文脈: ${context}\n"
    "翻訳対象文の発話者: ${speaker}\n"
    "翻訳対象文の発話対象: ${addressee}\n"
)

USER_PROMPT_JUDGE = string.Template(
    "以下に法廷での日本語の発話文（原文）、文脈を与えずに行なった翻訳（文脈なし翻訳）、文脈を踏まえた翻訳（文脈あり翻訳）を示します。"
    "翻訳文は、意訳は避け、単語の省略や付加をしない逐語訳で訳すことを意識して生成されたものですが、文脈なし翻訳は原文の内容を正確に翻訳できておらず、誤訳が含まれています。"
    "文脈なし翻訳に対して専門の法廷通訳人からいただいたコメント（誤訳のコメント）を与えます。"
    "文脈あり翻訳が誤訳のコメントの内容を解消できているかを判断し、自由にコメントしてください。\n"
    "原文: ${target}\n"
    "文脈なし翻訳: ${translation_no_context}\n"
    "文脈あり翻訳: ${translation_with_context}\n"
    "誤訳のコメント: ${comments}\n"
)


def completion(client, model: str, system_prompt: str, user_prompt: str) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content


def translate(client, model, data) -> str:
    return completion(
        client,
        model,
        SYSTEM_PROMPT,
        USER_PROMPT_TRANSLATE.substitute(
            target=data["target"],
            context=data["context"],
            speaker=data["speaker"],
            addressee=data["addressee"],
        ),
    )


def judge(client, data) -> str:
    return completion(
        client,
        "gpt-4.1-mini",
        SYSTEM_PROMPT,
        USER_PROMPT_JUDGE.substitute(
            target=data["target"],
            translation_no_context=data["translation"],
            translation_with_context=data["translation_with_context"],
            comments=data["comment"],
        ),
    )


def main():
    client_gpt = OpenAI(
        organization=os.getenv("OPENAI_ORGANIZATION"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    client_llama = Groq(api_key=os.getenv("GROQ_API_KEY"))
    with (
        open("../dataset/handbook_context.jsonl") as f,
        open(
            "../output/analysis/evaluation_with_context/evaluation_with_context.jsonl",
            "w",
        ) as f_out,
    ):
        for line in f:
            data = json.loads(line)
            data_updated = data.copy()
            if data["model"] == "gpt":
                translation_with_context = translate(client_gpt, "gpt-4o", data)
            elif data["model"] == "llama":
                translation_with_context = translate(
                    client_llama, "llama-3.3-70b-versatile", data
                )
            else:
                raise ValueError(f"Unknown model: {data['model']}")
            data_updated["translation_with_context"] = translation_with_context
            judge_text = judge(client_gpt, data_updated)
            data_updated["judge"] = judge_text
            f_out.write(json.dumps(data_updated, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
