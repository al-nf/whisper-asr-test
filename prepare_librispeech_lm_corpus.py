"""Build the char- and word-tokenized text corpora used to train the KenLM
n-gram models for the English (LibriSpeech) n-gram fusion experiment - the
English counterpart to `prepare_aishell_lm_corpus.py` / `prepare_mdcc_lm_corpus.py`.

This is the negative-control arm of the hypothesis: space-delimited English
is what WhisperLM-style 5-gram fusion was designed for, so word n-grams
should keep gaining through order 4/5 rather than saturating at 2/3.

LM text comes from the disjoint `train.100` + `validation` splits of
`openslr/librispeech_asr` (`clean`); `test` is never touched, matching
`eval_aishell_ngram_fusion.py --dataset-split test`. Only the `text` column
is kept after load - but the HF parquet still embeds audio, so the first
run downloads several GB. That is a one-time cost.

Usage:
    uv run prepare_librispeech_lm_corpus.py
    uv run prepare_librispeech_lm_corpus.py --output-dir ./lm_corpus_en
"""

import argparse
import os

from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

from ngram_lm import get_tokenizers, normalize_en_text

TOKENIZERS = get_tokenizers("en")

DATASET_REPO = "openslr/librispeech_asr"
DATASET_CONFIG = "clean"
TEXT_COLUMN = "text"
# train.100 (~28k utts) + validation (~2.7k) is the in-domain analogue of
# AISHELL train+dev / MDCC train+validation. train.360 is 3.6x more text
# but ~24GB of parquet - skip it unless you pass --include-train-360.
LM_TRAIN_SPLITS = ["train.100", "validation"]


def load_lm_texts(splits: "list[str]") -> "list[str]":
    parts = []
    for split in splits:
        print(f"  loading {DATASET_REPO} ({DATASET_CONFIG}/{split})...")
        ds = load_dataset(DATASET_REPO, name=DATASET_CONFIG, split=split)
        keep = [TEXT_COLUMN] if TEXT_COLUMN in ds.column_names else ds.column_names[:1]
        drop = [c for c in ds.column_names if c not in keep]
        if drop:
            ds = ds.remove_columns(drop)
        parts.append(ds)
    ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    texts = [normalize_en_text(t) for t in ds[TEXT_COLUMN]]
    return [t for t in texts if t]


def write_corpus(path: str, lines: "list[str]") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build char/word LM training corpora for the English (LibriSpeech) n-gram fusion experiment."
    )
    parser.add_argument(
        "--output-dir",
        default="./lm_corpus_en",
        help="Directory to write {char,word}.txt corpora into.",
    )
    parser.add_argument(
        "--include-train-360",
        action="store_true",
        help="Also include train.360 (~104k utts, large download). Default is train.100+validation only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    splits = list(LM_TRAIN_SPLITS)
    if args.include_train_360:
        splits.insert(1, "train.360")

    print(f"Loading {'+'.join(splits)} from {DATASET_REPO} (config={DATASET_CONFIG})...")
    texts = load_lm_texts(splits)
    print(f"  {len(texts)} utterances retained for the LM corpus.")

    for scheme, tokenizer in TOKENIZERS.items():
        out_path = os.path.join(args.output_dir, f"{scheme}.txt")
        lines = [
            " ".join(tokenizer(text))
            for text in tqdm(texts, desc=f"Tokenizing ({scheme})", unit="utt")
        ]
        lines = [line for line in lines if line]
        write_corpus(out_path, lines)
        print(f"Wrote {len(lines)} lines to {out_path}")

    print(f"\nDone. Next: bash scripts/build_kenlm_models.sh {args.output_dir} ./lm_en")
    print(
        "Then: uv run eval_aishell_ngram_fusion.py --lang en --language english "
        "--dataset-repo openslr/librispeech_asr --dataset-config clean "
        "--dataset-split test --text-column text --id-column id "
        "--asr-model openai/whisper-small --lm-dir ./lm_en --do-sample --max-new-tokens 128"
    )


if __name__ == "__main__":
    main()
