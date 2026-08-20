"""Build the char- and word-tokenized text corpora used to train the KenLM
n-gram models for the AISHELL-1 fusion experiment.

Downloads only the transcript file (no audio) from the official
`AISHELL/AISHELL-1` mirror, excludes every utterance id that appears in the
`Serenalay/AISHELL-1` *test* split (the split `eval_aishell_ngram_fusion.py`
evaluates on) so the LM never sees test transcripts, and writes the remainder
(train+dev) as two whitespace-tokenized corpora - one per character, one
jieba-segmented - ready for `scripts/build_kenlm_models.sh` (`lmplz`).

Usage:
    uv run prepare_aishell_lm_corpus.py
    uv run prepare_aishell_lm_corpus.py --output-dir ./lm_corpus
"""

import argparse
import os

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from ngram_lm import TOKENIZERS, normalize_zh_text

TRANSCRIPT_REPO = "AISHELL/AISHELL-1"
TRANSCRIPT_FILENAME = "data_aishell/transcript/aishell_transcript_v0.8.txt"
TEST_SPLIT_REPO = "Serenalay/AISHELL-1"


def load_test_utt_ids() -> set:
    """The set of utterance ids held out for evaluation - anything in this
    set must never end up in the LM training corpus."""
    test_ds = load_dataset(TEST_SPLIT_REPO, split="test")
    return set(test_ds["name"])


def load_transcripts() -> "dict[str, str]":
    """utt_id -> raw (unsegmented, whitespace-free) transcript text, for
    every utterance in the official AISHELL-1 corpus (train+dev+test)."""
    path = hf_hub_download(
        repo_id=TRANSCRIPT_REPO, repo_type="dataset", filename=TRANSCRIPT_FILENAME
    )
    transcripts = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, _, text = line.partition(" ")
            text = text.strip()
            if not text:
                continue
            transcripts[utt_id] = normalize_zh_text(text)
    return transcripts


def write_corpus(path: str, lines: "list[str]") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build char/word LM training corpora for the AISHELL-1 n-gram fusion experiment."
    )
    parser.add_argument(
        "--output-dir",
        default="./lm_corpus",
        help="Directory to write {char,word}.txt corpora into.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading test utterance ids from {TEST_SPLIT_REPO} (to exclude from the LM corpus)...")
    test_ids = load_test_utt_ids()
    print(f"  {len(test_ids)} test utterances will be excluded.")

    print(f"Downloading transcripts from {TRANSCRIPT_REPO}/{TRANSCRIPT_FILENAME}...")
    transcripts = load_transcripts()
    print(f"  {len(transcripts)} total utterances in the official transcript file.")

    train_dev_texts = [text for utt_id, text in transcripts.items() if utt_id not in test_ids]
    missing = len(test_ids) - sum(1 for utt_id in test_ids if utt_id in transcripts)
    print(f"  {len(train_dev_texts)} utterances retained for the LM corpus (train+dev).")
    if missing:
        print(f"  Note: {missing} test utterance ids were not found in the transcript file (harmless).")

    for scheme, tokenizer in TOKENIZERS.items():
        out_path = os.path.join(args.output_dir, f"{scheme}.txt")
        lines = [
            " ".join(tokenizer(text))
            for text in tqdm(train_dev_texts, desc=f"Tokenizing ({scheme})", unit="utt")
        ]
        lines = [line for line in lines if line]
        write_corpus(out_path, lines)
        print(f"Wrote {len(lines)} lines to {out_path}")

    print(f"\nDone. Next: scripts/build_kenlm_models.sh {args.output_dir}")


if __name__ == "__main__":
    main()
