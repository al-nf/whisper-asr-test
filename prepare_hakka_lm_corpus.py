"""Build the char-tokenized text corpus used to train the KenLM n-gram models
for the Hakka (formosan_asr_benchmark) n-gram fusion experiment - the Hakka
counterpart to `prepare_aishell_lm_corpus.py` / `prepare_mdcc_lm_corpus.py`.

Unlike AISHELL-1 (separate transcript file) or MDCC (proper train/validation/
test splits), `slammax/formosan_asr_benchmark`'s `hakka` subset ships only a
single 12k-row `test` split - there is no independent transcript source to
train the LM on without touching the sentences later used for ASR eval. So
this script self-partitions that one split deterministically by utterance id
(via `ngram_lm.is_lm_holdout`, a stable hash - not `random.shuffle`, so this
script and `eval_aishell_ngram_fusion.py` can agree on the partition without
passing an id list between them): utterances hashed into the holdout fraction
are *excluded* from the LM corpus here, and `eval_aishell_ngram_fusion.py
--lm-holdout-frac` evaluates ASR n-gram fusion on exactly (and only) those
same held-out utterances. Both must be run with the same `--holdout-frac`/
`--seed` (defaults already match).

Only "char" is produced - there is no maintained Hakka word-segmentation
library (no Hakka jieba/pycantonese equivalent); see `ngram_lm.py`'s
docstring. Pass `--schemes char` to `eval_aishell_ngram_fusion.py --lang hak`.

Usage:
    uv run prepare_hakka_lm_corpus.py
    uv run prepare_hakka_lm_corpus.py --output-dir ./lm_corpus_hak
"""

import argparse
import os

from datasets import load_dataset
from tqdm import tqdm

from ngram_lm import get_tokenizers, is_lm_holdout, normalize_zh_text

TOKENIZERS = get_tokenizers("hak")

DATASET_REPO = "slammax/formosan_asr_benchmark"
DATASET_CONFIG = "hakka"
DATASET_SPLIT = "test"
TEXT_COLUMN = "transcript"
ID_COLUMN = "audio_id"


def load_lm_texts(holdout_frac: float, seed: int) -> "list[str]":
    """Normalized transcript text for every utterance NOT in the eval holdout
    partition (see module docstring). Loads only the id/text columns - never
    touches the `audio` column, so no audio decoding happens here."""
    ds = load_dataset(DATASET_REPO, name=DATASET_CONFIG, split=DATASET_SPLIT)
    ds = ds.remove_columns([c for c in ds.column_names if c not in (ID_COLUMN, TEXT_COLUMN)])
    texts = [
        normalize_zh_text(text)
        for utt_id, text in zip(ds[ID_COLUMN], ds[TEXT_COLUMN])
        if not is_lm_holdout(str(utt_id), holdout_frac, seed)
    ]
    return [t for t in texts if t]


def write_corpus(path: str, lines: "list[str]") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the char LM training corpus for the Hakka (formosan_asr_benchmark) n-gram fusion experiment."
    )
    parser.add_argument(
        "--output-dir",
        default="./lm_corpus_hak",
        help="Directory to write char.txt corpus into.",
    )
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.3,
        help="Fraction of utterances excluded from the LM corpus and reserved for ASR eval "
        "(must match --lm-holdout-frac in eval_aishell_ngram_fusion.py).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Must match --seed in eval_aishell_ngram_fusion.py for the partitions to agree.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {DATASET_SPLIT} split from {DATASET_REPO} (config={DATASET_CONFIG})...")
    texts = load_lm_texts(args.holdout_frac, args.seed)
    print(f"  {len(texts)} utterances retained for the LM corpus (holdout_frac={args.holdout_frac} excluded).")

    for scheme, tokenizer in TOKENIZERS.items():
        out_path = os.path.join(args.output_dir, f"{scheme}.txt")
        lines = [
            " ".join(tokenizer(text))
            for text in tqdm(texts, desc=f"Tokenizing ({scheme})", unit="utt")
        ]
        lines = [line for line in lines if line]
        write_corpus(out_path, lines)
        print(f"Wrote {len(lines)} lines to {out_path}")

    print(f"\nDone. Next: bash scripts/build_kenlm_models.sh {args.output_dir} ./lm_hak")
    print(
        "Then: uv run eval_aishell_ngram_fusion.py --lang hak --schemes char "
        f"--lm-holdout-frac {args.holdout_frac} --seed {args.seed} ..."
    )


if __name__ == "__main__":
    main()
