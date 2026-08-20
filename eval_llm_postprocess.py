"""Measure how much LLM-based postprocessing improves ASR error rate.

Pipeline per sample: audio -> Whisper transcript (raw hyp) -> local LLM
correction pass (corrected hyp). WER/CER are computed for both raw and
corrected hypotheses against the reference so the improvement from the
LLM pass can be measured directly.

The default correction model, Qwen2.5-7B-Instruct, comfortably fits on a
single RTX 3090 (24GB) in bf16 (~15GB of weights) with no quantization
required. Swap models with --llm-model if you want to try something else.

Usage:
    uv run eval_llm_postprocess.py
    uv run eval_llm_postprocess.py --locales en_us es_419 --max-samples 50
    uv run eval_llm_postprocess.py --llm-model Qwen/Qwen2.5-7B-Instruct --load-in-4bit
"""

import argparse
import json
import math
import os
import re
import tempfile
import time
from typing import List, Optional

import soundfile as sf
import torch
import whisper
from datasets import load_dataset
from jiwer import cer, wer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


CORRECTION_PROMPT_TEMPLATE = """You will be given a raw Automatic Speech Recognition (ASR) transcript that may contain errors from word insertion, deletion, or substitution.
<asr_transcript>
{ASR_TEXT}
</asr_transcript>

Carefully review the transcript above and correct errors using phonetic similarity and sentence context to identify the intended words.
Rules:
1. Preserve the original sentence structure, word order, and clause boundaries. Do not rephrase, summarize, condense, or reorganize any part of the sentence.
2. Correct only individual words or short phrases that are clearly erroneous. Do not delete a word or phrase just because it is hard to interpret; instead, infer the most likely intended word(s) from phonetic similarity and context.
3. Never remove named entities, proper nouns, titles, numbers, or technical terms, even if garbled; reconstruct them rather than deleting them. When reconstructing a garbled entity, prefer the option that is phonetically closest to the ASR output over one that is merely topically or historically plausible.
4. Do not substitute a word with a different but similar-meaning word unless the original is phonetically implausible given the context.
5. If a word or phrase is too garbled to confidently reconstruct even with context, leave that portion of the ASR output unchanged rather than deleting or inventing content - do not guess at a plausible-sounding but unsupported replacement.
6. Do not add any word, phrase, or entity that has no plausible phonetic or contextual relationship to a specific span in the ASR output, even if it is historically accurate or fits the topic. Every word in your output must trace back to either (a) an unedited ASR token, or (b) a phonetic correction of a specific ASR token.
Output only the corrected text. No explanations, comments, or formatting."""

# Some models (especially mid-sized ones reasoning through phonetic
# ambiguity) leak their chain-of-thought - e.g. "Correction note: ...
# Final Output: <text>" - despite the prompt's own "output only the
# corrected text" rule. This system message exists purely to reinforce that
# instruction; it doesn't add or change any of the correction rules above.
STRICT_FORMAT_SYSTEM_PROMPT = (
    "You strictly follow output-format instructions. When asked to output only "
    "specific content, respond with exactly that and nothing else: no reasoning, "
    "no notes, no preambles or sign-offs, and no labels like 'Correction note:', "
    "'Final Output:', or 'Corrected transcript:'. Do the reasoning silently and "
    "return only the final requested content."
)

# Prepended to the model's turn so it starts already "mid-answer" instead of
# free to reason first - a response-prefill trick. This is the primary
# defense against reasoning leakage: markers below are a best-effort net for
# whatever slips through, not a substitute for suppressing it up front.
RESPONSE_PREFIX = "Corrected transcript: "

# Fallback markers used to salvage a usable answer if a model leaks reasoning
# anyway. If any of these appear, we take the text after the LAST match,
# since models tend to state their true final answer last after "thinking
# out loud" through the correction. This list can never be exhaustive -
# models invent novel transition phrases (e.g. "the corrected version would
# be") - so it's backed up by a length-ratio heuristic below rather than
# relied on alone.
FORMAT_LEAK_MARKERS = [
    "final output:",
    "final corrected transcript:",
    "final corrected text:",
    "final answer:",
    "final result:",
    "corrected transcript:",
    "corrected text:",
    "corrected version:",
    "correct version:",
    "corrected version would be",
    "correction would be",
    "corrected sentence would be",
    "corrected text would be",
]

# If the (possibly marker-extracted) candidate is still this many times
# longer than the source ASR text, treat it as a failed extraction rather
# than trust it - a single unrecovered reasoning dump can otherwise dominate
# a whole locale's aggregate WER. We fall back to the uncorrected raw ASR
# text in that case (i.e. "correction failed, count it as a no-op") instead
# of scoring the reasoning blob as the hypothesis.
LEAK_LENGTH_RATIO_THRESHOLD = 2.0


LOCALES = {
    "en_us": {"locale": "en_us", "lang": "en"},
    "cmn_hans_cn": {"locale": "cmn_hans_cn", "lang": "zh"},
    "yue_hant_hk": {"locale": "yue_hant_hk", "lang": "zh"},
    "es_419": {"locale": "es_419", "lang": "es"},
    "gl_es": {"locale": "gl_es", "lang": "gl"},
    "fr_fr": {"locale": "fr_fr", "lang": "fr"},
    "oc_fr": {"locale": "oc_fr", "lang": "oc"},
    "hi_in": {"locale": "hi_in", "lang": "hi"},
    "pa_in": {"locale": "pa_in", "lang": "pa"},
}

DEFAULT_LOCALES = ["en_us", "es_419", "fr_fr", "cmn_hans_cn"]

DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Convenience presets for --llm-model so capacity can be A/B tested without
# memorizing repo IDs / VRAM settings. All fit on a single 24GB 3090:
#   small  ~15GB bf16   - fast, but weak at obeying the strict correction rules
#   medium ~9GB  int4   - big jump in instruction-following over small
#   large  ~19GB int4   - closest local proxy to a frontier model; tight on
#                         VRAM, so drop --llm-batch-size (e.g. to 2-4) if you
#                         see OOMs, especially if Whisper is also loaded.
MODEL_TIERS = {
    "small": {"model": "Qwen/Qwen2.5-7B-Instruct", "load_in_4bit": False},
    "medium": {"model": "Qwen/Qwen2.5-14B-Instruct", "load_in_4bit": True},
    "large": {"model": "Qwen/Qwen2.5-32B-Instruct", "load_in_4bit": True},
}


def normalize_text(text: str, lang: str) -> str:
    if lang == "zh":
        return re.sub(r"[^\w]", "", text, flags=re.UNICODE)
    text = text.lower()
    return re.sub(r"[^\w\s]", "", text, flags=re.UNICODE).strip()


def get_run_dir() -> str:
    base = os.path.join("./logs", "llm_postprocess")
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


class LLMCorrector:
    """Wraps a local HF causal LM for greedy, batched ASR-transcript correction."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        load_in_4bit: bool = False,
        max_new_tokens: int = 384,
    ):
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding is required so batched generation aligns on the last token.
        self.tokenizer.padding_side = "left"

        model_kwargs = {"torch_dtype": torch.bfloat16}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device, **model_kwargs
        )
        self.model.eval()

    def _build_chat_text(self, asr_text: str) -> str:
        prompt = CORRECTION_PROMPT_TEMPLATE.format(ASR_TEXT=asr_text)
        messages = [
            {"role": "system", "content": STRICT_FORMAT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Response-prefill: the model's turn already "starts" mid-answer, so
        # there's no room before the answer for it to reason out loud first.
        return chat_text + RESPONSE_PREFIX

    @staticmethod
    def _extract_final_answer(text: str, source_text: str) -> "tuple[str, bool]":
        """Strip leaked reasoning/notes if the model violated the output-only
        rule. Returns (cleaned_text, was_noncompliant). Falls back to the
        uncorrected source text if the result still looks like a reasoning
        dump (implausibly long relative to the input) rather than risk
        scoring a leaked chain-of-thought as the hypothesis."""
        stripped = text.strip()
        lowered = stripped.lower()
        best_idx, best_len = -1, 0
        for marker in FORMAT_LEAK_MARKERS:
            idx = lowered.rfind(marker)
            if idx > best_idx:
                best_idx, best_len = idx, len(marker)

        noncompliant = False
        candidate = stripped
        if best_idx != -1:
            extracted = stripped[best_idx + best_len :].strip(' \n"`')
            if extracted:
                candidate, noncompliant = extracted, True

        source_len = max(len(source_text.split()), 1)
        if len(candidate.split()) > LEAK_LENGTH_RATIO_THRESHOLD * source_len:
            return source_text.strip(), True

        return candidate, noncompliant

    @torch.no_grad()
    def correct_batch(self, texts: List[str]) -> "tuple[List[str], List[bool]]":
        chat_texts = [self._build_chat_text(t) for t in texts]
        inputs = self.tokenizer(
            chat_texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        cleaned, flags = [], []
        for source_text, d in zip(texts, decoded):
            # The model's turn was prefilled with RESPONSE_PREFIX, so `d` is
            # already just the continuation after it - no need to strip it back off.
            text, noncompliant = self._extract_final_answer(d, source_text)
            cleaned.append(text)
            flags.append(noncompliant)
        return cleaned, flags

    def correct_all(self, texts: List[str], batch_size: int, desc: str) -> "tuple[List[str], List[bool]]":
        corrected, flags = [], []
        for i in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
            batch = texts[i : i + batch_size]
            batch_texts, batch_flags = self.correct_batch(batch)
            corrected.extend(batch_texts)
            flags.extend(batch_flags)
        return corrected, flags


def metric_or_nan(metric, refs: List[str], hyps: List[str]) -> float:
    if not refs:
        return math.nan
    return metric(refs, hyps)


def relative_improvement(baseline: float, corrected: float) -> float:
    if baseline == 0 or math.isnan(baseline) or math.isnan(corrected):
        return math.nan
    return (baseline - corrected) / baseline


def transcribe_dataset(dataset, whisper_model, lang: str, language_mode: str, label: str):
    refs, hyps = [], []
    kwargs = {"task": "transcribe"}
    if language_mode == "lang":
        kwargs["language"] = lang
    for row in tqdm(dataset, desc=f"ASR: {label}", unit="sample"):
        audio = row["audio"]
        refs.append(row["transcription"])
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, audio["array"], audio["sampling_rate"])
            result = whisper_model.transcribe(f.name, **kwargs)
        hyps.append(result["text"])
    return refs, hyps


def evaluate_locale(
    dataset,
    locale_cfg: dict,
    whisper_model,
    corrector: LLMCorrector,
    language_mode: str,
    llm_batch_size: int,
) -> dict:
    locale = locale_cfg["locale"]
    lang = locale_cfg["lang"]

    start = time.time()
    refs, raw_hyps = transcribe_dataset(dataset, whisper_model, lang, language_mode, locale)
    asr_seconds = time.time() - start

    start = time.time()
    llm_hyps, format_flags = corrector.correct_all(raw_hyps, llm_batch_size, desc=f"LLM: {locale}")
    llm_seconds = time.time() - start
    format_noncompliance_rate = sum(format_flags) / len(format_flags) if format_flags else math.nan

    refs_clean = [normalize_text(r, lang) for r in refs]
    raw_clean = [normalize_text(h, lang) for h in raw_hyps]
    llm_clean = [normalize_text(h, lang) for h in llm_hyps]

    baseline_wer = metric_or_nan(wer, refs_clean, raw_clean)
    baseline_cer = metric_or_nan(cer, refs_clean, raw_clean)
    llm_wer = metric_or_nan(wer, refs_clean, llm_clean)
    llm_cer = metric_or_nan(cer, refs_clean, llm_clean)

    rows = [
        {
            "index": i,
            "ref": refs[i],
            "raw_hyp": raw_hyps[i],
            "llm_hyp": llm_hyps[i],
            "format_noncompliant": format_flags[i],
            "sample_wer_raw": metric_or_nan(wer, [refs_clean[i]], [raw_clean[i]]),
            "sample_wer_llm": metric_or_nan(wer, [refs_clean[i]], [llm_clean[i]]),
            # WER degenerates for zh: normalize_text produces a whitespace-free
            # string, so jiwer.wer treats the whole sentence as one token
            # (near-binary "exact match or not"). CER is the metric that
            # actually carries signal there.
            "sample_cer_raw": metric_or_nan(cer, [refs_clean[i]], [raw_clean[i]]),
            "sample_cer_llm": metric_or_nan(cer, [refs_clean[i]], [llm_clean[i]]),
        }
        for i in range(len(refs))
    ]

    return {
        "locale": locale,
        "lang": lang,
        "samples": len(refs),
        "baseline_wer": baseline_wer,
        "baseline_cer": baseline_cer,
        "format_noncompliance_rate": format_noncompliance_rate,
        "llm_wer": llm_wer,
        "llm_cer": llm_cer,
        "wer_relative_improvement": relative_improvement(baseline_wer, llm_wer),
        "cer_relative_improvement": relative_improvement(baseline_cer, llm_cer),
        "asr_seconds": asr_seconds,
        "llm_seconds": llm_seconds,
        "rows": rows,
    }


def render_table(results: List[dict]) -> Table:
    table = Table(title="LLM Postprocessing Results")
    table.add_column("Locale", justify="left")
    table.add_column("Samples", justify="right")
    table.add_column("WER (raw)", justify="right")
    table.add_column("WER (llm)", justify="right")
    table.add_column("WER Δ", justify="right")
    table.add_column("CER (raw)", justify="right")
    table.add_column("CER (llm)", justify="right")
    table.add_column("CER Δ", justify="right")
    table.add_column("Fmt Noncompliance", justify="right")
    for r in results:
        table.add_row(
            r["locale"],
            str(r["samples"]),
            f"{r['baseline_wer']:.4f}",
            f"{r['llm_wer']:.4f}",
            f"{r['wer_relative_improvement']:+.1%}",
            f"{r['baseline_cer']:.4f}",
            f"{r['llm_cer']:.4f}",
            f"{r['cer_relative_improvement']:+.1%}",
            f"{r['format_noncompliance_rate']:.1%}",
        )
    return table


def print_results_table(results: List[dict]) -> None:
    Console().print(render_table(results))


def save_results_table(run_dir: str, results: List[dict]) -> None:
    console = Console(record=True, highlight=False)
    console.print(render_table(results))
    with open(os.path.join(run_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write(console.export_text())


def save_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate how much LLM postprocessing reduces Whisper WER/CER."
    )
    parser.add_argument(
        "--locales",
        nargs="+",
        default=DEFAULT_LOCALES,
        choices=list(LOCALES.keys()),
        help="FLEURS locales to evaluate.",
    )
    parser.add_argument("--asr-model", default="small", help="Whisper model size.")
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help=(
            "HF model id for the local correction LLM (must fit on your GPU), "
            f"or a capacity preset: {', '.join(MODEL_TIERS)}."
        ),
    )
    parser.add_argument(
        "--language-mode",
        choices=["auto", "lang"],
        default="lang",
        help="Whether to pass the known dataset language to Whisper or let it auto-detect.",
    )
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--llm-batch-size", type=int, default=8)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
        help=(
            "Generation budget per sample. Kept generous since a model that "
            "leaks reasoning despite the response-prefill trick still needs "
            "room to reach its actual answer before truncating mid-sentence."
        ),
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the correction LLM in 4-bit (requires bitsandbytes); useful on smaller GPUs.",
    )
    parser.add_argument(
        "--device", default="cuda", help="Device for the correction LLM (e.g. cuda, cuda:0, cpu)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = get_run_dir()
    print(f"Saving logs to: {run_dir}")

    llm_model = args.llm_model
    load_in_4bit = args.load_in_4bit
    if llm_model in MODEL_TIERS:
        preset = MODEL_TIERS[llm_model]
        llm_model = preset["model"]
        # An explicit --load-in-4bit always wins; otherwise use the preset's default.
        load_in_4bit = args.load_in_4bit or preset["load_in_4bit"]

    print(f"Loading Whisper model: {args.asr_model}")
    whisper_model = whisper.load_model(args.asr_model)

    print(f"Loading correction LLM: {llm_model} (4bit={load_in_4bit})")
    corrector = LLMCorrector(
        llm_model,
        device=args.device,
        load_in_4bit=load_in_4bit,
        max_new_tokens=args.max_new_tokens,
    )

    run_config = {
        "asr_model": args.asr_model,
        "llm_model": llm_model,
        "load_in_4bit": load_in_4bit,
        "language_mode": args.language_mode,
        "max_samples": args.max_samples,
    }
    save_json(os.path.join(run_dir, "run_config.json"), run_config)

    all_results = []
    for locale_key in args.locales:
        locale_cfg = LOCALES[locale_key]
        print(f"\n{'=' * 60}")
        print(f"Loading dataset: {locale_cfg['locale']}")
        print(f"{'=' * 60}")
        dataset = load_dataset("google/fleurs", locale_cfg["locale"], split="test")
        if args.max_samples is not None:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        try:
            result = evaluate_locale(
                dataset=dataset,
                locale_cfg=locale_cfg,
                whisper_model=whisper_model,
                corrector=corrector,
                language_mode=args.language_mode,
                llm_batch_size=args.llm_batch_size,
            )
            result["asr_model"] = args.asr_model
            result["llm_model"] = llm_model
            all_results.append(result)
            save_json(
                os.path.join(run_dir, f"{locale_cfg['locale']}.json"),
                result,
            )
            print_results_table(all_results)
        except Exception as exc:
            print(f"FAILED: locale={locale_cfg['locale']}")
            print(exc)
            raise

    save_results_table(run_dir, all_results)
    save_json(
        os.path.join(run_dir, "summary.json"),
        [{k: v for k, v in r.items() if k != "rows"} for r in all_results],
    )
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
