# Step 1: Preprocess OCR data to create a structured dataset
# Running this script generates `../dataset_pre/{language}.txt`
# We performed alignment between Japanese text and the other languages, then named them as `../dataset_pre/{language}_aligned.txt`

import argparse
import re
from logging import Logger

from court_interpreter.utils import get_logger


def is_index(line: str, regexp: str) -> bool:
    return re.match(regexp, line) is not None


def get_talker(line: str) -> str:
    match = re.match(r"\(.+\)( *)$", line)
    if match is None:
        return ""
    return match.group()


def preprocess_dataset(
    language: str,
    chapter_regexp: str,
    section_regexp: str,
    subsection_regexp: str,
    subsubsection_regexp: str,
    logger: Logger,
):
    with (
        open(f"../dataset_raw/{language}.txt", mode="r", encoding="utf-8") as f_prep,
        open(f"../dataset_pre/{language}.txt", mode="w", encoding="utf-8") as f_data,
    ):
        for line in f_prep:
            try:
                if is_index(line, chapter_regexp):
                    f_data.write(line)
                elif is_index(line, section_regexp):
                    f_data.write("\t" + line)
                elif is_index(line, subsection_regexp):
                    f_data.write("\t\t" + line)
                elif is_index(line, subsubsection_regexp):
                    f_data.write("\t\t\t" + line)
                elif get_talker(line):
                    t = get_talker(line)
                else:
                    if not line.strip():
                        continue
                    if line.startswith("【"):
                        f_data.write("\t\t\t\t" + line)
                    else:
                        f_data.write(f"\t\t\t\t{t.strip()} {line}")
            except:
                logger.error(line)


def main(args: argparse.Namespace, logger: Logger) -> None:
    if args.language == "english":
        chapter_regexp = r"^(I|II|III|IV|V|VI)\. (.+)"
    elif args.language == "vietnamese":
        chapter_regexp = r"^Chương (\d+)(.+)"
    else:
        chapter_regexp = r"第(\d+)章 (.+)"
    if args.language == "english" or args.language == "vietnamese":
        section_regexp = r"(\d+)\. (.+)"
    else:
        section_regexp = r"(\d+) (.+)"
    subsection_regexp = r"(\(\d+\)) (.+)"
    subsubsection_regexp = r"([a-z]\.) (.+)"
    preprocess_dataset(
        args.language,
        chapter_regexp,
        section_regexp,
        subsection_regexp,
        subsubsection_regexp,
        logger,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language",
        default="chinese",
        choices=["japanese", "chinese", "english", "vietnamese"],
    )
    args = parser.parse_args()
    logger = get_logger()
    main(args, logger)
