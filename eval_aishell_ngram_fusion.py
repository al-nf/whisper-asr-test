"""Test whether Mandarin ASR benefits most from low-order (bi/tri-gram) n-gram
LM fusion, vs. the 5-gram order WhisperLM-style approaches typically use for
space-delimited languages, and whether that depends on character- vs.
word-segmented (jieba) tokenization.

Benchmark: AISHELL-1 test set (`Serenalay/AISHELL-1`, 7176 utterances).
Model: a fine-tuned Whisper checkpoint (default: `junsor/whisper-small-aishell`).

Fusion mechanism: N-best rescoring, not shallow fusion during beam search.
Whisper's BPE tokens don't align to Chinese characters or jieba words, so
injecting an n-gram score at every decoding step would require guessing where
"character" or "word" boundaries fall inside a partially-generated BPE token -
fragile and hard to validate. Instead:

  1. Generate K beam candidates per utterance from Whisper alone, keeping each
     candidate's length-normalized acoustic log-prob (`sequences_scores`).
  2. Tokenize every candidate under a given scheme (char or jieba word) and
     score it with the matching KenLM n-gram model (order 2-5; order 1 is
     excluded since KenLM can't load a unigram-only model - see
     scripts/build_kenlm_models.sh).
  3. Re-rank candidates by `acoustic_avg_logprob + alpha * lm_avg_logprob` and
     take the top one. `alpha` is grid-searched on a held-out "tune" subset of
     the test set (there is no separate dev-set audio in this dataset mirror)
     and applied to the disjoint "eval" subset.

`alpha=0` reproduces the no-LM baseline for every order, which is a built-in
sanity check.

RER (relative error-rate reduction) = (baseline_error - condition_error) /
baseline_error, computed against the *same* eval subset's baseline. Alpha is
always tuned to minimize CER (the standard, segmentation-tool-independent
metric for Chinese); jieba-based WER is also reported per condition using
that same tuned alpha, as a secondary diagnostic.

Usage:
    uv run eval_aishell_ngram_fusion.py
    uv run eval_aishell_ngram_fusion.py --max-samples 200 --num-beams 5
    uv run eval_aishell_ngram_fusion.py --nbest-cache ./logs/aishell_ngram_fusion/nbest.json
"""

import argparse
import json
import os
import random
from typing import List, Optional

import torch
from datasets import load_dataset
from jiwer import cer, wer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from ngram_lm import KenLMScorer, TOKENIZERS, normalize_zh_text

DATASET_REPO = "Serenalay/AISHELL-1"
DEFAULT_ASR_MODEL = "junsor/whisper-small-aishell"
# Order 1 is excluded: KenLM's query/loading code hard-requires at least a
# bigram model ("This ngram implementation assumes at least a bigram model")
# even though lmplz can technically produce a unigram ARPA file - see
# scripts/build_kenlm_models.sh. The no-LM beam-search baseline computed
# below already serves as the effective "0th order" comparison point.
DEFAULT_ORDERS = [2, 3, 4, 5]
DEFAULT_SCHEMES = ["char", "word"]
DEFAULT_ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def get_run_dir() -> str:
    base = os.path.join("./logs", "aishell_ngram_fusion")
    os.makedirs("./logs", exist_ok=True)
    if not os.path.exists(base):
        os.makedirs(base)
        return base
    counter = 1
    while os.path.exists(f"{base}_{counter}"):
        counter += 1
    path = f"{base}_{counter}"
    os.makedirs(path)
    return path


def save_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def tune_eval_split(n: int, tune_frac: float, seed: int) -> "tuple[List[int], List[int]]":
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_tune = max(1, int(round(n * tune_frac)))
    return indices[:n_tune], indices[n_tune:]


def generate_nbest(
    dataset,
    model,
    processor,
    device: str,
    num_beams: int,
    max_new_tokens: int,
    batch_size: int,
    language: str,
    task: str,
) -> List[dict]:
    """Runs Whisper beam search once per utterance, keeping the top `num_beams`
    candidates and their length-normalized acoustic log-probs. This is the
    only step that touches the GPU/model - everything downstream (rescoring,
    alpha tuning, metric computation) operates on this cached output."""
    results = []
    for start in tqdm(range(0, len(dataset), batch_size), desc="ASR N-best", unit="batch"):
        batch = dataset[start : start + batch_size]
        audio_arrays = [a["array"] for a in batch["audio"]]
        inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device=device, dtype=model.dtype)

        with torch.no_grad():
            output = model.generate(
                input_features,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                language=language,
                task=task,
            )

        texts = processor.batch_decode(output.sequences, skip_special_tokens=True)
        scores = output.sequences_scores.tolist()

        n_items = len(audio_arrays)
        for i in range(n_items):
            item_texts = texts[i * num_beams : (i + 1) * num_beams]
            item_scores = scores[i * num_beams : (i + 1) * num_beams]
            seen = {}
            for text, score in zip(item_texts, item_scores):
                norm = normalize_zh_text(text)
                if norm not in seen or score > seen[norm]:
                    seen[norm] = score
            candidates = sorted(
                ({"text": t, "acoustic_avg_logprob": s} for t, s in seen.items()),
                key=lambda c: c["acoustic_avg_logprob"],
                reverse=True,
            )
            results.append(
                {
                    "utt_id": batch["name"][i],
                    "ref": normalize_zh_text(batch["text"][i]),
                    "candidates": candidates,
                }
            )
    return results


def load_scorers(lm_dir: str, schemes: List[str], orders: List[int]) -> "dict[tuple, KenLMScorer]":
    scorers = {}
    for scheme in schemes:
        for order in orders:
            path = os.path.join(lm_dir, scheme, f"order{order}.klm")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing KenLM model: {path}\n"
                    "Run prepare_aishell_lm_corpus.py then scripts/build_kenlm_models.sh first."
                )
            scorers[(scheme, order)] = KenLMScorer(path)
    return scorers


def rescore_candidates(nbest_item: dict, tokens_by_candidate: List[List[str]], scorer: KenLMScorer, alpha: float) -> int:
    """Returns the index of the best candidate under `acoustic + alpha * lm`."""
    best_idx, best_score = 0, float("-inf")
    for idx, (cand, tokens) in enumerate(zip(nbest_item["candidates"], tokens_by_candidate)):
        lm_score = scorer.avg_logprob(tokens)
        combined = cand["acoustic_avg_logprob"] + alpha * lm_score
        if combined > best_score:
            best_idx, best_score = idx, combined
    return best_idx


def evaluate_condition(
    nbest: List[dict],
    tune_idx: List[int],
    eval_idx: List[int],
    scheme: str,
    order: int,
    scorer: KenLMScorer,
    alpha_grid: List[float],
    tokens_cache: List[List[List[str]]],
) -> dict:
    best_alpha, best_tune_cer = alpha_grid[0], float("inf")
    for alpha in alpha_grid:
        refs, hyps = [], []
        for i in tune_idx:
            item = nbest[i]
            best_idx = rescore_candidates(item, tokens_cache[i], scorer, alpha)
            refs.append(item["ref"])
            hyps.append(item["candidates"][best_idx]["text"])
        tune_cer = cer(refs, hyps)
        if tune_cer < best_tune_cer:
            best_alpha, best_tune_cer = alpha, tune_cer

    rows = []
    for i in eval_idx:
        item = nbest[i]
        best_idx = rescore_candidates(item, tokens_cache[i], scorer, best_alpha)
        rows.append({"utt_id": item["utt_id"], "ref": item["ref"], "hyp": item["candidates"][best_idx]["text"]})

    refs = [r["ref"] for r in rows]
    hyps = [r["hyp"] for r in rows]
    word_refs = [" ".join(TOKENIZERS["word"](r)) for r in refs]
    word_hyps = [" ".join(TOKENIZERS["word"](h)) for h in hyps]

    return {
        "scheme": scheme,
        "order": order,
        "best_alpha": best_alpha,
        "tune_cer": best_tune_cer,
        "eval_cer": cer(refs, hyps),
        "eval_wer": wer(word_refs, word_hyps),
        "rows": rows,
    }


def compute_baseline(nbest: List[dict], eval_idx: List[int]) -> dict:
    """No-LM baseline: rank-0 (top acoustic score) candidate for every
    utterance in the eval subset."""
    rows = []
    for i in eval_idx:
        item = nbest[i]
        rows.append({"utt_id": item["utt_id"], "ref": item["ref"], "hyp": item["candidates"][0]["text"]})
    refs = [r["ref"] for r in rows]
    hyps = [r["hyp"] for r in rows]
    word_refs = [" ".join(TOKENIZERS["word"](r)) for r in refs]
    word_hyps = [" ".join(TOKENIZERS["word"](h)) for h in hyps]
    return {"cer": cer(refs, hyps), "wer": wer(word_refs, word_hyps), "rows": rows}


def render_table(baseline: dict, conditions: List[dict]) -> Table:
    table = Table(title="AISHELL-1 N-gram Fusion Results (N-best rescoring)")
    table.add_column("Scheme", justify="left")
    table.add_column("Order", justify="right")
    table.add_column("alpha*", justify="right")
    table.add_column("CER", justify="right")
    table.add_column("RER (CER)", justify="right")
    table.add_column("WER", justify="right")
    table.add_column("RER (WER)", justify="right")
    table.add_row("baseline", "-", "-", f"{baseline['cer']:.4f}", "-", f"{baseline['wer']:.4f}", "-")
    for c in conditions:
        rer_cer = (baseline["cer"] - c["eval_cer"]) / baseline["cer"] if baseline["cer"] else float("nan")
        rer_wer = (baseline["wer"] - c["eval_wer"]) / baseline["wer"] if baseline["wer"] else float("nan")
        table.add_row(
            c["scheme"],
            str(c["order"]),
            f"{c['best_alpha']:.2f}",
            f"{c['eval_cer']:.4f}",
            f"{rer_cer:+.1%}",
            f"{c['eval_wer']:.4f}",
            f"{rer_wer:+.1%}",
        )
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure RER per n-gram order/tokenization scheme for Whisper+KenLM fusion on AISHELL-1."
    )
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL, help="HF transformers Whisper checkpoint id/path.")
    parser.add_argument("--lm-dir", default="./lm", help="Directory containing {char,word}/order{n}.klm KenLM binaries.")
    parser.add_argument("--orders", nargs="+", type=int, default=DEFAULT_ORDERS)
    parser.add_argument("--schemes", nargs="+", choices=list(TOKENIZERS), default=DEFAULT_SCHEMES)
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=DEFAULT_ALPHA_GRID)
    parser.add_argument("--num-beams", type=int, default=10, help="Beam width == N-best size.")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None, help="Subsample the test set (for quick iteration).")
    parser.add_argument("--tune-frac", type=float, default=0.2, help="Fraction of the test set used to grid-search alpha.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", default="chinese")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="float32")
    parser.add_argument(
        "--nbest-cache",
        default=None,
        help="Path to a previously saved nbest.json - skips loading Whisper / running ASR entirely.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = get_run_dir()
    print(f"Saving logs to: {run_dir}")

    run_config = vars(args).copy()
    save_json(os.path.join(run_dir, "run_config.json"), run_config)

    if args.nbest_cache:
        print(f"Loading cached N-best from {args.nbest_cache}")
        with open(args.nbest_cache, "r", encoding="utf-8") as f:
            nbest = json.load(f)
    else:
        print(f"Loading dataset: {DATASET_REPO} (test split)")
        dataset = load_dataset(DATASET_REPO, split="test")
        if args.max_samples is not None:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        print(f"Loading ASR model: {args.asr_model} (device={args.device}, dtype={args.dtype})")
        processor = WhisperProcessor.from_pretrained(args.asr_model)
        model = WhisperForConditionalGeneration.from_pretrained(
            args.asr_model, torch_dtype=DTYPE_MAP[args.dtype]
        ).to(args.device)
        model.eval()

        nbest = generate_nbest(
            dataset,
            model,
            processor,
            device=args.device,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            language=args.language,
            task=args.task,
        )
        save_json(os.path.join(run_dir, "nbest.json"), nbest)

    tune_idx, eval_idx = tune_eval_split(len(nbest), args.tune_frac, args.seed)
    print(f"tune={len(tune_idx)} eval={len(eval_idx)} utterances")

    baseline = compute_baseline(nbest, eval_idx)
    print(f"Baseline: CER={baseline['cer']:.4f} WER={baseline['wer']:.4f}")

    scorers = load_scorers(args.lm_dir, args.schemes, args.orders)

    conditions = []
    for scheme in args.schemes:
        tokenize = TOKENIZERS[scheme]
        print(f"Tokenizing N-best candidates for scheme={scheme}...")
        tokens_cache = [
            [tokenize(c["text"]) for c in item["candidates"]]
            for item in tqdm(nbest, desc=f"tokenize ({scheme})", unit="utt")
        ]
        for order in args.orders:
            print(f"Rescoring: scheme={scheme} order={order}")
            result = evaluate_condition(
                nbest, tune_idx, eval_idx, scheme, order, scorers[(scheme, order)], args.alpha_grid, tokens_cache
            )
            conditions.append(result)
            print(
                f"  alpha*={result['best_alpha']:.2f} eval_cer={result['eval_cer']:.4f} "
                f"eval_wer={result['eval_wer']:.4f}"
            )

    console = Console()
    table = render_table(baseline, conditions)
    console.print(table)

    record_console = Console(record=True, highlight=False)
    record_console.print(table)
    with open(os.path.join(run_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write(record_console.export_text())

    save_json(
        os.path.join(run_dir, "results.json"),
        {
            "run_config": run_config,
            "n_tune": len(tune_idx),
            "n_eval": len(eval_idx),
            "baseline": baseline,
            "conditions": conditions,
        },
    )
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
