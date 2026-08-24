"""Build the char- and pycantonese-word-tokenized text corpora used to train
the KenLM n-gram models for the Cantonese (MDCC) n-gram fusion experiment -
the Cantonese counterpart to `prepare_aishell_lm_corpus.py`.

Unlike AISHELL-1 (where the audio mirror used for eval only has a `test`
split, so the LM corpus has to come from a separately-downloaded official
transcript file with test utterance ids excluded by hand), the
`ming030890/mdcc` mirror already ships proper, disjoint `train` /
`validation` / `test` splits with a `transcript` text column - matching the
original MDCC paper's split sizes (65120 / 5663 / 12492). So here we just
load `train` (+ `validation`) directly and tokenize their `transcript`
column; `test` is never touched, matching `eval_aishell_ngram_fusion.py`'s
`--dataset-split test`.

Usage:
    uv run prepare_mdcc_lm_corpus.py
    uv run prepare_mdcc_lm_corpus.py --output-dir ./lm_corpus_yue
"""

import argparse
import os

from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

from ngram_lm import get_tokenizers, normalize_zh_text

TOKENIZERS = get_tokenizers("yue")

DATASET_REPO = "ming030890/mdcc"
TEXT_COLUMN = "transcript"
LM_TRAIN_SPLITS = ["train", "validation"]


def load_lm_texts() -> "list[str]":
    """Normalized transcript text for every utterance in the LM-training
    splits (train+validation) - `test` is deliberately excluded since that's
    what `eval_aishell_ngram_fusion.py --dataset-repo ming030890/mdcc`
    evaluates on."""
    parts = [load_dataset(DATASET_REPO, split=split) for split in LM_TRAIN_SPLITS]
    ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    texts = [normalize_zh_text(t) for t in ds[TEXT_COLUMN]]
    return [t for t in texts if t]


def write_corpus(path: str, lines: "list[str]") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build char/word LM training corpora for the Cantonese (MDCC) n-gram fusion experiment."
    )
    parser.add_argument(
        "--output-dir",
        default="./lm_corpus_yue",
        help="Directory to write {char,word}.txt corpora into.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {'+'.join(LM_TRAIN_SPLITS)} splits from {DATASET_REPO}...")
    texts = load_lm_texts()
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

    print(f"\nDone. Next: bash scripts/build_kenlm_models.sh {args.output_dir} ./lm_yue")


if __name__ == "__main__":
    main()
