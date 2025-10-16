# Step 5-2: Evaluate the translations with BLEU, BERTScore, and COMET
# Running this script generates `../output/evaluation/{metrics}/{evaluation_set_name}.tsv`

import argparse
import ast
import os
from pathlib import Path

import pandas as pd
from bert_score import score
from comet import download_model, load_from_checkpoint
from comet.models.base import CometModel
from sacrebleu import BLEU

from court_interpreter.utils import get_logger

logger = get_logger()


def generate_data_for_comet(
    srcs: list[str], syss: list[str], refs: list[str]
) -> list[dict[str, str]]:
    return [
        {"src": src, "mt": sys, "ref": ref} for (src, sys, ref) in zip(srcs, syss, refs)
    ]


def generate_data_for_comet_reference_free(
    srcs: list[str], transs: list[str]
) -> list[dict[str, str]]:
    return [{"src": src, "mt": trans} for (src, trans) in zip(srcs, transs)]


def calculate_bleu(
    language: str,
    output_path: str,
    ids: list[str],
    transs: dict[str, list[str]],
):
    if language == "chinese":
        bleu = BLEU(tokenize="zh", effective_order=True)
    else:
        bleu = BLEU(effective_order=True)
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, mode="w") as f:
        f.write("ID\tgpt\tllama\tazure\ttarget_len\tgpt_len\tllama_len\tazure_len\n")
        for id_s, gpt_s, llama_s, azure_s, target_s in zip(
            ids, transs["gpt"], transs["llama"], transs["azure"], transs["target"]
        ):
            gpt_bleu = bleu.sentence_score(gpt_s, [target_s])
            llama_bleu = bleu.sentence_score(llama_s, [target_s])
            azure_bleu = bleu.sentence_score(azure_s, [target_s])
            f.write(
                f"{id_s.zfill(3)}\t{round(gpt_bleu.score, 2)}\t{round(llama_bleu.score, 2)}\t{round(azure_bleu.score, 2)}\t{gpt_bleu.ref_len}\t{gpt_bleu.sys_len}\t{llama_bleu.sys_len}\t{azure_bleu.sys_len}\n"
            )
        gpt_corpus = bleu.corpus_score(transs["gpt"], [transs["target"]])
        llama_corpus = bleu.corpus_score(transs["llama"], [transs["target"]])
        azure_corpus = bleu.corpus_score(transs["azure"], [transs["target"]])
        f.write(
            f"ALL\t{round(gpt_corpus.score, 2)}\t{round(llama_corpus.score, 2)}\t{round(azure_corpus.score, 2)}\t{gpt_corpus.ref_len}\t{gpt_corpus.sys_len}\t{llama_corpus.sys_len}\t{azure_corpus.sys_len}\n"
        )


def calculate_bertscore(
    language: str,
    output_path: str,
    ids: list[str],
    transs: dict[str, list[str]],
):
    os.makedirs(Path(output_path).parent, exist_ok=True)

    def _get_bertscores(lang: str):
        gpt_p, gpt_r, gpt_f1 = score(
            transs["gpt"], transs["target"], lang=lang, verbose=True
        )
        llama_p, llama_r, llama_f1 = score(
            transs["llama"], transs["target"], lang=lang, verbose=True
        )
        azure_p, azure_r, azure_f1 = score(
            transs["azure"], transs["target"], lang=lang, verbose=True
        )
        return (
            gpt_p,
            gpt_r,
            gpt_f1,
            llama_p,
            llama_r,
            llama_f1,
            azure_p,
            azure_r,
            azure_f1,
        )

    if language == "chinese":
        gpt_p, gpt_r, gpt_f1, llama_p, llama_r, llama_f1, azure_p, azure_r, azure_f1 = (
            _get_bertscores("zh")
        )
    elif language == "english":
        gpt_p, gpt_r, gpt_f1, llama_p, llama_r, llama_f1, azure_p, azure_r, azure_f1 = (
            _get_bertscores("en")
        )
    elif language == "vietnamese":
        # "bert-base-multilingual-cased" is used (see utils)
        gpt_p, gpt_r, gpt_f1, llama_p, llama_r, llama_f1, azure_p, azure_r, azure_f1 = (
            _get_bertscores("vi")
        )
    else:
        raise ValueError(f"Invalid language: {language}")
    with open(output_path, mode="w") as f:
        f.write(
            "ID\tgpt_prec\tgpt_rec\tgpt_f1\tllama_prec\tllama_rec\tllama_f1\tazure_prec\tazure_rec\tazure_f1\n"
        )
        for (
            id_s,
            gpt_p_s,
            gpt_r_s,
            gpt_f1_s,
            llama_p_s,
            llama_r_s,
            llama_f1_s,
            azure_p_s,
            azure_r_s,
            azure_f1_s,
        ) in zip(
            ids,
            gpt_p,
            gpt_r,
            gpt_f1,
            llama_p,
            llama_r,
            llama_f1,
            azure_p,
            azure_r,
            azure_f1,
        ):
            f.write(
                f"{id_s.zfill(3)}\t{round(float(gpt_p_s), 2)}\t{round(float(gpt_r_s), 2)}\t{round(float(gpt_f1_s), 2)}\t"
                f"{round(float(llama_p_s), 2)}\t{round(float(llama_r_s), 2)}\t{round(float(llama_f1_s), 2)}\t"
                f"{round(float(azure_p_s), 2)}\t{round(float(azure_r_s), 2)}\t{round(float(azure_f1_s), 2)}\n"
            )
        f.write(
            f"ALL\t{round(gpt_p.mean().item(), 2)}\t{round(gpt_r.mean().item(), 2)}\t{round(gpt_f1.mean().item(), 2)}\t"
            f"{round(llama_p.mean().item(), 2)}\t{round(llama_r.mean().item(), 2)}\t{round(llama_f1.mean().item(), 2)}\t"
            f"{round(azure_p.mean().item(), 2)}\t{round(azure_r.mean().item(), 2)}\t{round(azure_f1.mean().item(), 2)}"
        )


def calculate_comet_ref_free_question(
    output_path: str,
    ids: list[str],
    srcs: list[str],
    transs: dict[str, list[str]],
):
    os.makedirs(Path(output_path).parent, exist_ok=True)
    comet_model_path: str = download_model("Unbabel/XCOMET-XL")
    comet_model: CometModel = load_from_checkpoint(comet_model_path)
    assert isinstance(comet_model, CometModel)
    comet_gpt = generate_data_for_comet_reference_free(srcs, transs["gpt"])
    comet_llama = generate_data_for_comet_reference_free(srcs, transs["llama"])
    comet_azure = generate_data_for_comet_reference_free(srcs, transs["azure"])
    comet_gpt_output = comet_model.predict(comet_gpt, batch_size=4, gpus=1)
    comet_llama_output = comet_model.predict(comet_llama, batch_size=4, gpus=1)
    comet_azure_output = comet_model.predict(comet_azure, batch_size=4, gpus=1)
    with open(output_path, mode="w") as f:
        f.write("ID\tgpt\tllama\tazure\n")
        for id_s, gpt, llama, azure in zip(
            ids,
            comet_gpt_output.scores,
            comet_llama_output.scores,
            comet_azure_output.scores,
        ):
            f.write(
                f"{id_s.zfill(3)}\t{round(gpt, 2)}\t{round(llama, 2)}\t{round(azure, 2)}\n"
            )
        f.write(
            f"ALL\t{round(comet_gpt_output.system_score, 2)}\t"
            f"{round(comet_llama_output.system_score, 2)}\t"
            f"{round(comet_azure_output.system_score, 2)}"
        )


def calculate_comet_ref_free(
    output_path: str,
    ids: list[str],
    srcs: list[str],
    transs: dict[str, list[str]],
):
    os.makedirs(Path(output_path).parent, exist_ok=True)
    comet_model_path: str = download_model("Unbabel/XCOMET-XL")
    comet_model: CometModel = load_from_checkpoint(comet_model_path)
    assert isinstance(comet_model, CometModel)
    comet_target = generate_data_for_comet_reference_free(srcs, transs["target"])
    comet_gpt = generate_data_for_comet_reference_free(srcs, transs["gpt"])
    comet_llama = generate_data_for_comet_reference_free(srcs, transs["llama"])
    comet_azure = generate_data_for_comet_reference_free(srcs, transs["azure"])
    comet_target_output = comet_model.predict(comet_target, batch_size=4, gpus=1)
    comet_gpt_output = comet_model.predict(comet_gpt, batch_size=4, gpus=1)
    comet_llama_output = comet_model.predict(comet_llama, batch_size=4, gpus=1)
    comet_azure_output = comet_model.predict(comet_azure, batch_size=4, gpus=1)
    with open(output_path, mode="w") as f:
        f.write("ID\ttarget\tgpt\tllama\tazure\n")
        for id_s, target, gpt, llama, azure in zip(
            ids,
            comet_target_output.scores,
            comet_gpt_output.scores,
            comet_llama_output.scores,
            comet_azure_output.scores,
        ):
            f.write(
                f"{id_s.zfill(3)}\t{round(target, 2)}\t{round(gpt, 2)}\t{round(llama, 2)}\t{round(azure, 2)}\n"
            )
        f.write(
            f"ALL\t{round(comet_target_output.system_score, 2)}\t"
            f"{round(comet_gpt_output.system_score, 2)}\t"
            f"{round(comet_llama_output.system_score, 2)}\t"
            f"{round(comet_azure_output.system_score, 2)}"
        )


def calculate_comet(
    output_path: str,
    ids: list[str],
    srcs: list[str],
    transs: dict[str, list[str]],
):
    os.makedirs(Path(output_path).parent, exist_ok=True)
    comet_model_path: str = download_model("Unbabel/XCOMET-XL")
    comet_model: CometModel = load_from_checkpoint(comet_model_path)
    assert isinstance(comet_model, CometModel)
    comet_gpt = generate_data_for_comet(srcs, transs["gpt"], transs["target"])
    comet_llama = generate_data_for_comet(srcs, transs["llama"], transs["target"])
    comet_azure = generate_data_for_comet(srcs, transs["azure"], transs["target"])
    comet_gpt_output = comet_model.predict(comet_gpt, batch_size=4, gpus=1)
    comet_llama_output = comet_model.predict(comet_llama, batch_size=4, gpus=1)
    comet_azure_output = comet_model.predict(comet_azure, batch_size=4, gpus=1)
    with open(output_path, mode="w") as f:
        f.write("ID\tgpt\tllama\tazure\n")
        for id_s, gpt, llama, azure in zip(
            ids,
            comet_gpt_output.scores,
            comet_llama_output.scores,
            comet_azure_output.scores,
        ):
            f.write(
                f"{id_s.zfill(3)}\t{round(gpt, 2)}\t{round(llama, 2)}\t{round(azure, 2)}\n"
            )
        f.write(
            f"ALL\t{round(comet_gpt_output.system_score, 2)}\t"
            f"{round(comet_llama_output.system_score, 2)}\t"
            f"{round(comet_azure_output.system_score, 2)}"
        )


def remap(
    translation_A: list[str],
    translation_B: list[str],
    translation_C: list[str],
    translation_D: list[str],
    mappings: list[dict[str, str]],
) -> dict[str, list[str]]:
    def _append_helper(
        gpt: list[str],
        llama: list[str],
        azure: list[str],
        target: list[str],
        v: str,
        trans: str,
    ):
        if v == "translation_gpt":
            gpt.append(trans)
        elif v == "translation_llama":
            llama.append(trans)
        elif v == "translation_azure":
            azure.append(trans)
        elif v == "target":
            target.append(trans)
        else:
            raise ValueError(f"Invalid value: {v}")

    gpt: list[str] = []
    llama: list[str] = []
    azure: list[str] = []
    target: list[str] = []
    for a, b, c, d, m in zip(
        translation_A, translation_B, translation_C, translation_D, mappings
    ):
        _append_helper(gpt, llama, azure, target, m["A"], a)
        _append_helper(gpt, llama, azure, target, m["B"], b)
        _append_helper(gpt, llama, azure, target, m["C"], c)
        _append_helper(gpt, llama, azure, target, m["D"], d)
        assert len(gpt) == len(llama) == len(azure) == len(target)
    return {"gpt": gpt, "llama": llama, "azure": azure, "target": target}


def remap_sys(
    translation_A: list[str],
    translation_B: list[str],
    translation_C: list[str],
    mappings: list[dict[str, str]],
) -> dict[str, list[str]]:
    def _append_helper(
        gpt: list[str],
        llama: list[str],
        azure: list[str],
        v: str,
        trans: str,
    ):
        if v == "translation_gpt":
            gpt.append(trans)
        elif v == "translation_llama":
            llama.append(trans)
        elif v == "translation_azure":
            azure.append(trans)
        else:
            raise ValueError(f"Invalid value: {v}")

    gpt: list[str] = []
    llama: list[str] = []
    azure: list[str] = []
    for a, b, c, m in zip(translation_A, translation_B, translation_C, mappings):
        _append_helper(gpt, llama, azure, m["A"], a)
        _append_helper(gpt, llama, azure, m["B"], b)
        _append_helper(gpt, llama, azure, m["C"], c)
        assert len(gpt) == len(llama) == len(azure)
    return {"gpt": gpt, "llama": llama, "azure": azure}


def main(args: argparse.Namespace) -> None:
    filename = Path(args.evaluation_set_path).name
    if args.question:
        df = pd.read_csv(args.evaluation_set_path, sep="\t")  # type: ignore
        ids: list[str] = df["id"].to_list()
        srcs: list[str] = df["source"].to_list()
        translation_A: list[str] = df["translation_A"].to_list()
        translation_B: list[str] = df["translation_B"].to_list()
        translation_C: list[str] = df["translation_C"].to_list()
        mapping_list: list[str] = df["mapping"].to_list()
        mappings: list[dict[str, str]] = [ast.literal_eval(m) for m in mapping_list]
        syss = remap_sys(translation_A, translation_B, translation_C, mappings)
        if args.comet_ref_free:
            output_path = f"{args.output_dir}/comet_ref_free/{filename}"
            calculate_comet_ref_free_question(output_path, ids, srcs, syss)
    else:
        df = pd.read_csv(args.evaluation_set_path, sep="\t")  # type: ignore
        ids: list[str] = df["id"].to_list()
        srcs: list[str] = df["source"].to_list()
        translation_A: list[str] = df["translation_A"].to_list()
        translation_B: list[str] = df["translation_B"].to_list()
        translation_C: list[str] = df["translation_C"].to_list()
        translation_D: list[str] = df["translation_D"].to_list()
        mapping_list: list[str] = df["mapping"].to_list()
        mappings: list[dict[str, str]] = [ast.literal_eval(m) for m in mapping_list]
        transs = remap(
            translation_A, translation_B, translation_C, translation_D, mappings
        )
        if args.bleu:
            logger.info("Calculating BLEU...")
            output_path = f"{args.output_dir}/bleu/{filename}"
            calculate_bleu(args.language, output_path, ids, transs)
        if args.bertscore:
            logger.info("Calculating BERTScore...")
            output_path = f"{args.output_dir}/bertscore/{filename}"
            calculate_bertscore(args.language, output_path, ids, transs)
        if args.comet:
            logger.info("Calculating COMET...")
            output_path = f"{args.output_dir}/comet/{filename}"
            calculate_comet(output_path, ids, srcs, transs)
        if args.comet_ref_free:
            logger.info("Calculating COMET (reference-free)...")
            output_path = f"{args.output_dir}/comet_ref_free/{filename}"
            calculate_comet_ref_free(output_path, ids, srcs, transs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language", default="chinese", choices=["chinese", "english", "vietnamese"]
    )
    parser.add_argument("--bleu", action="store_true")
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--comet_ref_free", action="store_true")
    parser.add_argument(
        "--evaluation_set_path",
        default="../output/translation/handbook_evaluation_set_chinese.tsv",
    )
    parser.add_argument(
        "--question", type=bool, default=False, help="whether to evaluate question"
    )
    parser.add_argument("--output_dir", default="../output/evaluation")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
