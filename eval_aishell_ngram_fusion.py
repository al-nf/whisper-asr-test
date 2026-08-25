"""Test whether Mandarin ASR benefits most from low-order (bi/tri-gram) n-gram
LM fusion, vs. the 5-gram order WhisperLM-style approaches typically use for
space-delimited languages, and whether that depends on character- vs.
word-segmented (jieba) tokenization.

Benchmark: AISHELL-1 test set (`Serenalay/AISHELL-1`, 7176 utterances) by
default. Also usable for Cantonese via `--lang yue --dataset-repo
ming030890/mdcc --text-column transcript --id-column id --asr-model
<cantonese checkpoint>` - see README for the full Cantonese walkthrough.
`--lang` only changes the "word" tokenizer (jieba for zh,
pycantonese.segment for yue); "char" tokenization is identical either way.
Note: only whisper-large-v3/-turbo-derived checkpoints have a real
`<|yue|>` language token (added in large-v3); tiny/base/small/medium
Cantonese fine-tunes were trained against `<|zh|>` like any Mandarin
fine-tune, so `--language` should stay `chinese` for those (the script
auto-detects and falls back if you pass `cantonese`/`yue` on such a model).
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
always tuned to minimize CER, the standard metric for Chinese (WER isn't
reported: Mandarin has no native word boundaries, so any "word" only exists
relative to an arbitrary segmentation tool's choices - unlike CER, it
wouldn't be measuring something intrinsic to the text).

Usage:
    uv run eval_aishell_ngram_fusion.py
    uv run eval_aishell_ngram_fusion.py --max-samples 200 --num-beams 5
    uv run eval_aishell_ngram_fusion.py --nbest-cache ./logs/aishell_ngram_fusion/nbest.json

Resuming after a crash (e.g. the Jetson NVML/CUDACachingAllocator assertion -
see README): the ASR stage writes `nbest.json` into the run dir after *every*
batch, not just at the end. Re-run with `--run-dir` pointing at the same
directory and it will skip utterances already present in that file and pick
up where it left off:
    uv run eval_aishell_ngram_fusion.py --run-dir ./logs/aishell_ngram_fusion
"""

import argparse
import gc
import json
import os
import re

# Must be set before CUDA is initialized. On Jetson/Tegra, PyTorch's caching
# allocator can hit contiguous-memory (CMA) exhaustion under repeated
# alloc/free churn from beam search; when that happens it tries to query NVML
# for diagnostics and NVML's partial Tegra support makes it crash with
# `RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at
# CUDACachingAllocator.cpp` instead of a catchable OOM. Expandable segments
# reduce allocator fragmentation and make that failure far less likely.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
from typing import List, Optional

import torch
from datasets import load_dataset
from jiwer import cer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

from ngram_lm import KenLMScorer, TOKENIZERS, get_tokenizers, normalize_zh_text

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


# Keyed by (d_model, encoder_layers, num_mel_bins) - these three uniquely
# identify every official Whisper architecture size, including telling
# large-v3/large-v3-turbo (128 mel bins, 100 language tokens incl. `<|yue|>`)
# apart from large-v1/v2 (80 mel bins, 99 tokens, no dedicated Cantonese token).
_WHISPER_BASE_REPO_BY_ARCH = {
    (384, 4, 80): "openai/whisper-tiny",
    (512, 6, 80): "openai/whisper-base",
    (768, 12, 80): "openai/whisper-small",
    (1024, 24, 80): "openai/whisper-medium",
    (1280, 32, 80): "openai/whisper-large-v2",
    (1280, 32, 128): "openai/whisper-large-v3",
}


def _infer_base_whisper_repo(model) -> Optional[str]:
    cfg = model.config
    key = (cfg.d_model, cfg.encoder_layers, cfg.num_mel_bins)
    return _WHISPER_BASE_REPO_BY_ARCH.get(key)


def ensure_multilingual_generation_config(model, asr_model_name: str) -> None:
    """Some Whisper fine-tunes (typically ones saved in 2023, before
    huggingface/transformers#25298) ship a `generation_config.json` that's
    missing `lang_to_id`/`task_to_id` entirely, because those fields were
    added to the base checkpoints' configs *after* the fine-tune was
    uploaded. Passing `language=`/`task=` to `.generate()` on such a
    checkpoint raises `ValueError: The generation config is outdated...`
    (see https://github.com/huggingface/transformers/issues/25084).

    Patches `model.generation_config` in place (in memory only - never
    pushed to the Hub, and no model weights are touched) by borrowing the
    token-id mappings from the official checkpoint of matching architecture
    size, which is exactly the fix the transformers maintainers recommend in
    that issue."""
    if getattr(model.generation_config, "lang_to_id", None):
        return
    base_repo = _infer_base_whisper_repo(model)
    if base_repo is None:
        print(
            f"[warn] {asr_model_name}'s generation_config.json is missing lang_to_id/task_to_id "
            "(see https://github.com/huggingface/transformers/issues/25084) and its architecture doesn't "
            "match a known Whisper size, so it can't be auto-repaired. --language/--task will likely fail."
        )
        return
    print(
        f"[warn] {asr_model_name}'s generation_config.json is missing lang_to_id/task_to_id (an outdated "
        f"fine-tuned-checkpoint issue, transformers#25084) - borrowing them from {base_repo} in memory so "
        "--language/--task work. No model weights are changed."
    )
    base_config = GenerationConfig.from_pretrained(base_repo)
    for attr in ("lang_to_id", "task_to_id", "is_multilingual"):
        if hasattr(base_config, attr):
            setattr(model.generation_config, attr, getattr(base_config, attr))


def resolve_language_for_model(model, asr_model_name: str, language: str) -> str:
    """Only whisper-large-v3/large-v3-turbo-derived checkpoints have a real
    `<|yue|>` (Cantonese) token - Whisper's original 99-language set (tiny
    through large-v2) has no dedicated Cantonese token at all, so
    tiny/base/small/medium Cantonese fine-tunes (e.g.
    Oblivion208/whisper-small-cantonese) were necessarily trained to map
    Cantonese audio onto `<|zh|>` (Chinese) text, same as a Mandarin
    fine-tune. Passing `--language cantonese` to one of those would fail (no
    `<|yue|>` token to force) or silently do the wrong thing; fall back to
    `chinese` and say so."""
    if language not in ("cantonese", "yue"):
        return language
    lang_to_id = getattr(model.generation_config, "lang_to_id", None) or {}
    if "<|yue|>" in lang_to_id:
        return language
    print(
        f"[warn] {asr_model_name} has no dedicated Cantonese ('<|yue|>') token - only "
        "large-v3/large-v3-turbo-derived Whisper checkpoints do. This model was fine-tuned to map "
        "Cantonese audio onto Chinese ('<|zh|>') text like any small/base/medium Cantonese Whisper "
        "fine-tune. Falling back to --language chinese; use a large-v3-based checkpoint if you need "
        "the real yue token."
    )
    return "chinese"


def get_run_dir(lang: str = "zh") -> str:
    name = "aishell_ngram_fusion" if lang == "zh" else f"{lang}_ngram_fusion"
    base = os.path.join("./logs", name)
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


def _transition_avg_logprobs(model, sequences: "torch.Tensor", scores: tuple) -> List[float]:
    """Per-sequence average natural-log token probability, computed directly
    from `compute_transition_scores` instead of `output.sequences_scores`.

    Needed because `WhisperForConditionalGeneration.generate()`'s internal
    "temperature fallback" logic (`generate_with_fallback` in
    `transformers.models.whisper.generation_whisper`) unconditionally forces
    `num_beams=1` whenever `do_sample=True` - so a `do_sample=True` call never
    actually uses beam search, and the returned `GenerateEncoderDecoderOutput`
    has no `sequences_scores` attribute at all (that field only exists on
    `GenerateBeamEncoderDecoderOutput`). This reconstructs an equivalent
    average log-prob directly from `output.scores` for that case."""
    transition_scores = model.compute_transition_scores(sequences, scores, normalize_logits=True)
    n_steps = transition_scores.shape[1]
    generated = sequences[:, sequences.shape[1] - n_steps :]

    eos_token_id = model.generation_config.eos_token_id
    eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
    is_eos = torch.isin(generated, torch.tensor(eos_ids, device=generated.device))

    avg_logprobs = []
    for row_scores, row_is_eos in zip(transition_scores, is_eos):
        eos_pos = torch.nonzero(row_is_eos, as_tuple=True)[0]
        length = max(int(eos_pos[0].item()) + 1 if eos_pos.numel() > 0 else row_scores.shape[0], 1)
        avg_logprobs.append((row_scores[:length].sum() / length).item())
    return avg_logprobs


_WHISPER_CONTROL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
_warned_leaked_control_tokens = False


def _strip_leaked_control_tokens(text: str) -> str:
    """`processor.batch_decode(..., skip_special_tokens=True)` relies on each
    added token's own `special` flag in the tokenizer's added-tokens table.
    Some Whisper fine-tunes (e.g. `Oblivion208/whisper-small-cantonese`) ship
    a tokenizer where control tokens like `<|startoftranscript|>`/`<|zh|>`/
    `<|transcribe|>`/`<|notimestamps|>` are present but marked
    `special=False`, so they leak into the decoded text verbatim - silently
    prepending ~50 characters of garbage to every hypothesis and inflating
    CER past 100% (`normalize_zh_text` strips the `<|>` punctuation but not
    the alphanumeric token names themselves). `<|...|>` markup never appears
    in legitimate transcript text, so stripping it unconditionally here is a
    safe no-op for checkpoints that don't have this bug."""
    cleaned = _WHISPER_CONTROL_TOKEN_RE.sub("", text)
    global _warned_leaked_control_tokens
    if cleaned != text and not _warned_leaked_control_tokens:
        _warned_leaked_control_tokens = True
        print(
            "[warn] Decoded text contained literal Whisper control-token markup (e.g. "
            "'<|startoftranscript|>') even with skip_special_tokens=True - this checkpoint's "
            "tokenizer marks those tokens as non-special. Stripped automatically; if you see this, "
            "double-check other tooling you point at this checkpoint does the same."
        )
    return cleaned


def _run_batch(
    model,
    processor,
    device: str,
    sub_dataset,
    num_beams: int,
    max_new_tokens: int,
    language: str,
    task: str,
    diversity_opts: Optional[dict] = None,
    id_column: str = "name",
    text_column: str = "text",
) -> List[dict]:
    """Runs beam search (or sampling) on a single (small) chunk of the
    dataset. Raises on failure - callers handle OOM-style retries.

    Plain (deterministic) beam search on a narrowly fine-tuned, highly
    confident model - like an AISHELL-only Whisper checkpoint on short,
    clean, in-domain read speech - can converge every beam to the identical
    top-1 sequence: ASR posteriors are usually far more peaked than
    open-ended text generation, so beam search's top-k expansion just
    re-derives the same argmax path k times. That leaves nothing for any
    n-gram LM to rescore, independent of order or alpha. Two ways to force
    real diversity:

    - `num_beam_groups > 1`: HF's diverse beam search - groups beams and
      penalizes within-step similarity across groups. Deterministic;
      guarantees distinct candidates, but the "diversity" is an artificial
      penalty rather than the model's own uncertainty.
    - `do_sample=True`: independent multinomial-sampled candidates. Note this
      is *not* HF's generic "beam-search multinomial sampling" (do_sample +
      num_beams>1) - `WhisperForConditionalGeneration.generate()`'s
      temperature-fallback logic hardcodes `num_beams=1` whenever
      `do_sample=True`, so this always runs as `num_beams` independent
      ancestral samples (`num_return_sequences=num_beams`), not beam search.
      `acoustic_avg_logprob` is reconstructed via `_transition_avg_logprobs`
      since `sequences_scores` isn't populated for non-beam generation.

    `num_beam_groups > 1` and `do_sample=True` are mutually exclusive here -
    if both are set, diverse beam search wins.

    `diversity_opts` (all optional): {num_beam_groups, diversity_penalty,
    do_sample, temperature, top_k, top_p}."""
    opts = diversity_opts or {}
    audio_arrays = [a["array"] for a in sub_dataset["audio"]]
    inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device=device, dtype=model.dtype)

    generate_kwargs = dict(
        num_beams=num_beams,
        num_return_sequences=num_beams,
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=max_new_tokens,
        language=language,
        task=task,
    )
    num_beam_groups = opts.get("num_beam_groups", 1)
    if num_beam_groups > 1:
        generate_kwargs["num_beam_groups"] = num_beam_groups
        generate_kwargs["diversity_penalty"] = opts.get("diversity_penalty", 0.0)
    elif opts.get("do_sample"):
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = opts.get("temperature", 1.0)
        top_k = opts.get("top_k", 0)
        top_p = opts.get("top_p", 1.0)
        if top_k > 0:
            generate_kwargs["top_k"] = top_k
        if top_p < 1.0:
            generate_kwargs["top_p"] = top_p

    with torch.no_grad():
        output = model.generate(input_features, **generate_kwargs)

    texts = [
        _strip_leaked_control_tokens(t)
        for t in processor.batch_decode(output.sequences, skip_special_tokens=True)
    ]
    sequences_scores = getattr(output, "sequences_scores", None)
    if sequences_scores is not None:
        scores = sequences_scores.tolist()
    else:
        scores = _transition_avg_logprobs(model, output.sequences, output.scores)

    out = []
    for i in range(len(audio_arrays)):
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
        out.append(
            {
                "utt_id": sub_dataset[id_column][i],
                "ref": normalize_zh_text(sub_dataset[text_column][i]),
                "candidates": candidates,
            }
        )
    return out


_RECOVERABLE_ERROR_MARKERS = (
    "CUDA out of memory",
    "out of memory",
    "NVML_SUCCESS",
    "CUDACachingAllocator",
    "NvMap",
)


def _is_recoverable_cuda_error(exc: BaseException) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _RECOVERABLE_ERROR_MARKERS)


def _run_batch_with_retry(
    model,
    processor,
    device: str,
    sub_dataset,
    num_beams: int,
    max_new_tokens: int,
    language: str,
    task: str,
    diversity_opts: Optional[dict] = None,
    id_column: str = "name",
    text_column: str = "text",
) -> List[dict]:
    """Runs `_run_batch`, and on a CUDA OOM / Jetson NVML-allocator error
    (see module docstring), frees memory and retries with the batch split in
    half - down to single utterances. A single utterance that still fails
    is recorded with an empty hypothesis rather than aborting the whole run."""
    try:
        return _run_batch(
            model, processor, device, sub_dataset, num_beams, max_new_tokens, language, task, diversity_opts,
            id_column, text_column,
        )
    except RuntimeError as e:
        if not _is_recoverable_cuda_error(e):
            raise
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        n = len(sub_dataset)
        if n <= 1:
            utt_id = sub_dataset[id_column][0]
            tqdm.write(f"  [warn] utt_id={utt_id} failed even at batch size 1 ({e}); recording empty hypothesis.")
            return [
                {
                    "utt_id": utt_id,
                    "ref": normalize_zh_text(sub_dataset[text_column][0]),
                    "candidates": [{"text": "", "acoustic_avg_logprob": 0.0}],
                }
            ]
        mid = n // 2
        tqdm.write(f"  [warn] batch of {n} failed ({type(e).__name__}: {e}); splitting into {mid}+{n - mid} and retrying.")
        left = sub_dataset.select(range(0, mid))
        right = sub_dataset.select(range(mid, n))
        return _run_batch_with_retry(
            model, processor, device, left, num_beams, max_new_tokens, language, task, diversity_opts,
            id_column, text_column,
        ) + _run_batch_with_retry(
            model, processor, device, right, num_beams, max_new_tokens, language, task, diversity_opts,
            id_column, text_column,
        )


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
    checkpoint_path: Optional[str] = None,
    diversity_opts: Optional[dict] = None,
    id_column: str = "name",
    text_column: str = "text",
) -> List[dict]:
    """Runs Whisper beam search once per utterance, keeping the top `num_beams`
    candidates and their length-normalized acoustic log-probs. This is the
    only step that touches the GPU/model - everything downstream (rescoring,
    alpha tuning, metric computation) operates on this cached output.

    If `checkpoint_path` is given, results are written after every batch
    (keyed by utt_id, so order-independent), and any utterances already
    present there at startup are skipped - i.e. re-running with the same
    `checkpoint_path` after a crash resumes instead of starting over."""
    full_names = [str(x) for x in dataset[id_column]]
    results_by_id: "dict[str, dict]" = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                results_by_id[item["utt_id"]] = item
        print(f"Resuming from checkpoint: {len(results_by_id)}/{len(full_names)} utterances already done.")

    remaining_indices = [i for i, name in enumerate(full_names) if name not in results_by_id]
    if remaining_indices:
        subset = dataset.select(remaining_indices)
        n_batches = (len(subset) + batch_size - 1) // batch_size
        for b in tqdm(range(n_batches), desc="ASR N-best", unit="batch"):
            start = b * batch_size
            end = min(start + batch_size, len(subset))
            sub = subset.select(range(start, end))
            batch_results = _run_batch_with_retry(
                model, processor, device, sub, num_beams, max_new_tokens, language, task, diversity_opts,
                id_column, text_column,
            )
            for item in batch_results:
                item["utt_id"] = str(item["utt_id"])
                results_by_id[item["utt_id"]] = item
            if checkpoint_path:
                ordered_so_far = [results_by_id[name] for name in full_names if name in results_by_id]
                save_json(checkpoint_path, ordered_so_far)

    return [results_by_id[name] for name in full_names]


def load_scorers(lm_dir: str, schemes: List[str], orders: List[int]) -> "dict[tuple, KenLMScorer]":
    scorers = {}
    for scheme in schemes:
        for order in orders:
            path = os.path.join(lm_dir, scheme, f"order{order}.klm")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing KenLM model: {path}\n"
                    "Run prepare_aishell_lm_corpus.py (or prepare_mdcc_lm_corpus.py for Cantonese) then "
                    f"'bash scripts/build_kenlm_models.sh <corpus_dir> {lm_dir}' first."
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

    return {
        "scheme": scheme,
        "order": order,
        "best_alpha": best_alpha,
        "tune_cer": best_tune_cer,
        "eval_cer": cer(refs, hyps),
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
    return {"cer": cer(refs, hyps), "rows": rows}


def render_table(baseline: dict, conditions: List[dict], title: str = "AISHELL-1 N-gram Fusion Results (N-best rescoring)") -> Table:
    table = Table(title=title)
    table.add_column("Scheme", justify="left")
    table.add_column("Order", justify="right")
    table.add_column("alpha*", justify="right")
    table.add_column("CER", justify="right")
    table.add_column("RER (CER)", justify="right")
    table.add_row("baseline", "-", "-", f"{baseline['cer']:.4f}", "-")
    for c in conditions:
        rer_cer = (baseline["cer"] - c["eval_cer"]) / baseline["cer"] if baseline["cer"] else float("nan")
        table.add_row(
            c["scheme"],
            str(c["order"]),
            f"{c['best_alpha']:.2f}",
            f"{c['eval_cer']:.4f}",
            f"{rer_cer:+.1%}",
        )
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure RER per n-gram order/tokenization scheme for Whisper+KenLM fusion on AISHELL-1."
    )
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL, help="HF transformers Whisper checkpoint id/path.")
    parser.add_argument(
        "--dataset-repo",
        default=DATASET_REPO,
        help="HF datasets repo to evaluate on. Default is the AISHELL-1 (Mandarin) mirror; pass "
        "'ming030890/mdcc' (with --dataset-split test --text-column transcript --id-column id "
        "--language cantonese) to run the same test on Cantonese.",
    )
    parser.add_argument("--dataset-split", default="test", help="Split of --dataset-repo to evaluate on.")
    parser.add_argument("--text-column", default="text", help="Dataset column holding the reference transcript.")
    parser.add_argument("--id-column", default="name", help="Dataset column holding a unique utterance id.")
    parser.add_argument(
        "--lang",
        choices=["zh", "yue"],
        default="zh",
        help="Selects the 'word' tokenizer: jieba for Mandarin (zh) or pycantonese.segment for Cantonese (yue). "
        "'char' tokenization is identical either way. Use 'yue' together with --dataset-repo ming030890/mdcc.",
    )
    parser.add_argument("--lm-dir", default="./lm", help="Directory containing {char,word}/order{n}.klm KenLM binaries.")
    parser.add_argument("--orders", nargs="+", type=int, default=DEFAULT_ORDERS)
    parser.add_argument("--schemes", nargs="+", choices=list(TOKENIZERS), default=DEFAULT_SCHEMES)
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=DEFAULT_ALPHA_GRID)
    parser.add_argument("--num-beams", type=int, default=5, help="Beam width == N-best size.")
    parser.add_argument(
        "--num-beam-groups",
        type=int,
        default=1,
        help="Use HF diverse beam search with this many groups (must divide --num-beams evenly, and be >1 to "
        "take effect). Fixes beam collapse on confident, narrow-domain models where plain beam search produces "
        "near-duplicate candidates that dedup down to 1 per utterance, leaving nothing to rescore. Try e.g. 5 "
        "(with --num-beams 5) if diagnose_nbest.py shows most utterances have only 1 unique candidate.",
    )
    parser.add_argument(
        "--diversity-penalty",
        type=float,
        default=0.5,
        help="Diverse beam search penalty (only used if --num-beam-groups > 1).",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use HF's 'beam-search multinomial sampling' (do_sample=True with --num-beams > 1) instead of "
        "plain deterministic beam search: keeps beam-search score bookkeeping but samples each step's "
        "expansion, so beams aren't guaranteed-identical even when the model is extremely confident. The "
        "more faithful alternative to --num-beam-groups for fixing beam collapse (see diagnose_nbest.py). "
        "Ignored if --num-beam-groups > 1 (HF gives diverse beam search priority over sampling).",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (only if --do-sample).")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k filtering for sampling (0 = disabled).")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus filtering for sampling (1.0 = disabled).")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None, help="Subsample the test set (for quick iteration).")
    parser.add_argument("--tune-frac", type=float, default=0.2, help="Fraction of the test set used to grid-search alpha.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", default="chinese")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=list(DTYPE_MAP) + ["auto"],
        default="auto",
        help="'auto' resolves to float16 on cuda (much faster/lighter on Jetson) or float32 on cpu.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Use this directory instead of auto-numbering a new one. Required to resume a crashed run: "
        "the ASR stage checkpoints nbest.json here after every batch and skips utterances already in it.",
    )
    parser.add_argument(
        "--nbest-cache",
        default=None,
        help="Path to a previously saved (complete) nbest.json - skips loading Whisper / running ASR entirely.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lang == "yue" and args.asr_model == DEFAULT_ASR_MODEL:
        print(
            f"[warn] --lang yue but --asr-model is still the Mandarin default ({DEFAULT_ASR_MODEL}). "
            "Pass a Cantonese-finetuned checkpoint, e.g. --asr-model Oblivion208/whisper-small-cantonese "
            "(see README)."
        )
    if args.run_dir:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = get_run_dir(args.lang)
    print(f"Saving logs to: {run_dir}")

    dtype_name = "float16" if args.dtype == "auto" and args.device.startswith("cuda") else args.dtype
    if dtype_name == "auto":
        dtype_name = "float32"

    run_config = vars(args).copy()
    run_config["resolved_dtype"] = dtype_name

    run_config_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(run_config_path) and os.path.exists(os.path.join(run_dir, "nbest.json")):
        with open(run_config_path, "r", encoding="utf-8") as f:
            prev_config = json.load(f)
        generation_keys = [
            "asr_model", "num_beams", "num_beam_groups", "diversity_penalty",
            "do_sample", "temperature", "top_k", "top_p",
            "max_new_tokens", "language", "task", "resolved_dtype",
        ]
        mismatches = [k for k in generation_keys if prev_config.get(k) != run_config.get(k)]
        if mismatches:
            print(
                f"[warn] Resuming into {run_dir}, but these generation-affecting settings differ from the "
                f"previous run: {mismatches}. The resumed nbest.json would mix candidates generated under "
                "different settings - use a fresh --run-dir (or match the previous settings) instead."
            )

    save_json(run_config_path, run_config)

    if args.nbest_cache:
        print(f"Loading cached N-best from {args.nbest_cache}")
        with open(args.nbest_cache, "r", encoding="utf-8") as f:
            nbest = json.load(f)
    else:
        print(f"Loading dataset: {args.dataset_repo} ({args.dataset_split} split)")
        dataset = load_dataset(args.dataset_repo, split=args.dataset_split)
        if args.max_samples is not None:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        print(f"Loading ASR model: {args.asr_model} (device={args.device}, dtype={dtype_name})")
        processor = WhisperProcessor.from_pretrained(args.asr_model)
        try:
            model = WhisperForConditionalGeneration.from_pretrained(
                args.asr_model, torch_dtype=DTYPE_MAP[dtype_name], attn_implementation="sdpa"
            ).to(args.device)
        except (ImportError, ValueError):
            model = WhisperForConditionalGeneration.from_pretrained(
                args.asr_model, torch_dtype=DTYPE_MAP[dtype_name]
            ).to(args.device)
        model.eval()
        ensure_multilingual_generation_config(model, args.asr_model)
        resolved_language = resolve_language_for_model(model, args.asr_model, args.language)
        if resolved_language != args.language:
            args.language = resolved_language
            run_config["language"] = resolved_language
            save_json(run_config_path, run_config)

        checkpoint_path = os.path.join(run_dir, "nbest.json")
        diversity_opts = {
            "num_beam_groups": args.num_beam_groups,
            "diversity_penalty": args.diversity_penalty,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        }
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
            checkpoint_path=checkpoint_path,
            diversity_opts=diversity_opts,
            id_column=args.id_column,
            text_column=args.text_column,
        )
        save_json(checkpoint_path, nbest)

    tune_idx, eval_idx = tune_eval_split(len(nbest), args.tune_frac, args.seed)
    print(f"tune={len(tune_idx)} eval={len(eval_idx)} utterances")

    baseline = compute_baseline(nbest, eval_idx)
    print(f"Baseline: CER={baseline['cer']:.4f}")

    scorers = load_scorers(args.lm_dir, args.schemes, args.orders)

    tokenizers = get_tokenizers(args.lang)
    conditions = []
    for scheme in args.schemes:
        tokenize = tokenizers[scheme]
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
            print(f"  alpha*={result['best_alpha']:.2f} eval_cer={result['eval_cer']:.4f}")

    console = Console()
    table = render_table(baseline, conditions, title=f"{args.dataset_repo} N-gram Fusion Results (N-best rescoring)")
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
