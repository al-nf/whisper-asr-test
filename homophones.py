import argparse
import json
import math
import os
import re
import tempfile
import time
from collections import Counter
from typing import Iterable

import soundfile as sf
import whisper
from datasets import load_dataset
from jiwer import cer, wer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm


LOCALES = [
    {"locale": "en_us", "lang": "en", "label": "english"},
    {"locale": "cmn_hans_cn", "lang": "cmn", "label": "mandarin"},
    {"locale": "yue_hant_hk", "lang": "yue", "label": "cantonese"},
]

MODEL_SIZES = ["tiny", "base", "small"]


def normalize_text(text: str, lang: str) -> str:
    if lang in {"cmn", "yue"}:
        return re.sub(r"[^\w]", "", text, flags=re.UNICODE)
    text = text.lower()
    return re.sub(r"[^\w\s]", "", text, flags=re.UNICODE).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def cjk_chars(text: str) -> list[str]:
    return re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text)


def soundex(word: str) -> str:
    """Small dependency-free fallback for English phonetic bucketing."""
    if not word:
        return ""
    codes = {
        **dict.fromkeys("bfpv", "1"),
        **dict.fromkeys("cgjkqsxz", "2"),
        **dict.fromkeys("dt", "3"),
        "l": "4",
        **dict.fromkeys("mn", "5"),
        "r": "6",
    }
    first = word[0].upper()
    encoded = [codes.get(ch, "0") for ch in word.lower()]
    deduped = [encoded[0]]
    for code in encoded[1:]:
        if code != deduped[-1]:
            deduped.append(code)
    digits = [code for code in deduped[1:] if code != "0"]
    return (first + "".join(digits) + "000")[:4]


def english_pronunciations(text: str) -> list[str]:
    try:
        import pronouncing
    except ImportError:
        return [soundex(word) for word in words(text)]

    units = []
    for word in words(text):
        phones = pronouncing.phones_for_word(word)
        if phones:
            units.append(re.sub(r"\d", "", phones[0]))
        else:
            units.append(soundex(word))
    return units


def mandarin_pronunciations(text: str) -> list[str]:
    chars = cjk_chars(text)
    try:
        from pypinyin import Style, pinyin
    except ImportError:
        return chars

    return [item[0] for item in pinyin(chars, style=Style.TONE3, neutral_tone_with_five=True)]


def cantonese_pronunciations(text: str) -> list[str]:
    chars = cjk_chars(text)
    try:
        import pycantonese
    except ImportError:
        return chars

    raw = pycantonese.characters_to_jyutping("".join(chars))
    units = []
    for idx, item in enumerate(raw):
        if isinstance(item, tuple):
            pron = item[-1]
        else:
            pron = item
        if pron:
            units.extend(pron.split())
        elif idx < len(chars):
            units.append(chars[idx])
    return units


def pronunciation_units(text: str, lang: str) -> list[str]:
    if lang == "en":
        return english_pronunciations(text)
    if lang == "cmn":
        return mandarin_pronunciations(text)
    if lang == "yue":
        return cantonese_pronunciations(text)
    return normalize_text(text, lang).split()


def homophonic_density(text: str, lang: str) -> dict:
    units = [unit for unit in pronunciation_units(text, lang) if unit]
    total = len(units)
    if total == 0:
        return {
            "homophonic_density": 0.0,
            "homophone_units": 0,
            "pronunciation_units": 0,
            "unique_pronunciations": 0,
        }

    counts = Counter(units)
    duplicate_units = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "homophonic_density": duplicate_units / total,
        "homophone_units": duplicate_units,
        "pronunciation_units": total,
        "unique_pronunciations": len(counts),
    }


def metric_or_nan(metric, ref: str, hyp: str) -> float:
    if not ref and not hyp:
        return 0.0
    if not ref:
        return math.nan
    return metric(ref, hyp)


def get_run_dir() -> str:
    base = os.path.join("./logs", "homophonic_density")
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


def evaluate_model(dataset: Iterable[dict], model_size: str, locale_cfg: dict) -> dict:
    locale = locale_cfg["locale"]
    lang = locale_cfg["lang"]
    label = locale_cfg["label"]
    model = whisper.load_model(model_size)
    rows = []
    refs = []
    hyps = []
    start = time.time()

    progress = f"{label} | {model_size} | language=auto"
    for idx, row in enumerate(tqdm(dataset, desc=progress, unit="sample")):
        audio = row["audio"]
        ref = row["transcription"]
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, audio["array"], audio["sampling_rate"])
            result = model.transcribe(f.name, task="transcribe")

        hyp = result["text"]
        ref_clean = normalize_text(ref, lang)
        hyp_clean = normalize_text(hyp, lang)
        density = homophonic_density(ref, lang)

        refs.append(ref_clean)
        hyps.append(hyp_clean)
        rows.append(
            {
                "index": idx,
                "ref": ref,
                "hyp": hyp,
                "ref_clean": ref_clean,
                "hyp_clean": hyp_clean,
                "sentence_wer": metric_or_nan(wer, ref_clean, hyp_clean),
                "sentence_cer": metric_or_nan(cer, ref_clean, hyp_clean),
                **density,
            }
        )

    elapsed = time.time() - start
    return {
        "locale": locale,
        "language_label": label,
        "model": model_size,
        "language": "auto",
        "samples": len(rows),
        "wer": wer(refs, hyps),
        "cer": cer(refs, hyps),
        "avg_homophonic_density": sum(r["homophonic_density"] for r in rows) / len(rows),
        "seconds": elapsed,
        "seconds_per_sample": elapsed / len(rows),
        "rows": rows,
    }


def print_results_table(results: list[dict]) -> None:
    console = Console()
    table = Table(title="Homophonic Density ASR Results")
    table.add_column("Locale", justify="left")
    table.add_column("Label", justify="left")
    table.add_column("Model", justify="left")
    table.add_column("Samples", justify="right")
    table.add_column("WER", justify="right")
    table.add_column("CER", justify="right")
    table.add_column("Avg Homo Density", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Sec/Sample", justify="right")
    for result in results:
        table.add_row(
            result["locale"],
            result["language_label"],
            result["model"],
            str(result["samples"]),
            f"{result['wer']:.4f}",
            f"{result['cer']:.4f}",
            f"{result['avg_homophonic_density']:.4f}",
            f"{result['seconds']:.1f}s",
            f"{result['seconds_per_sample']:.2f}",
        )
    console.print(table)


def save_results_table(run_dir: str, results: list[dict]) -> None:
    console = Console(record=True, highlight=False)
    table = Table(title="Homophonic Density ASR Results")
    table.add_column("Locale", justify="left")
    table.add_column("Label", justify="left")
    table.add_column("Model", justify="left")
    table.add_column("Samples", justify="right")
    table.add_column("WER", justify="right")
    table.add_column("CER", justify="right")
    table.add_column("Avg Homo Density", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Sec/Sample", justify="right")
    for result in results:
        table.add_row(
            result["locale"],
            result["language_label"],
            result["model"],
            str(result["samples"]),
            f"{result['wer']:.4f}",
            f"{result['cer']:.4f}",
            f"{result['avg_homophonic_density']:.4f}",
            f"{result['seconds']:.1f}s",
            f"{result['seconds_per_sample']:.2f}",
        )
    console.print(table)
    with open(os.path.join(run_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write(console.export_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper and record per-sentence homophonic density."
    )
    parser.add_argument("--models", nargs="+", default=MODEL_SIZES)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional sample limit for quick smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = get_run_dir()
    all_results = []

    for locale_cfg in LOCALES:
        print(f"\n{'=' * 60}")
        print(f"Loading dataset: {locale_cfg['locale']} ({locale_cfg['label']})")
        print(f"{'=' * 60}")
        dataset = load_dataset("google/fleurs", locale_cfg["locale"], split="test")
        if args.max_samples is not None:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        for model_size in args.models:
            try:
                result = evaluate_model(dataset, model_size, locale_cfg)
                all_results.append({k: v for k, v in result.items() if k != "rows"})
                output_name = f"{locale_cfg['label']}_{model_size}_auto.json"
                save_json(os.path.join(run_dir, output_name), result)
                print_results_table(all_results)
            except Exception as exc:
                failure = {
                    "locale": locale_cfg["locale"],
                    "language_label": locale_cfg["label"],
                    "model": model_size,
                    "language": "auto",
                    "samples": len(dataset),
                    "wer": math.nan,
                    "cer": math.nan,
                    "avg_homophonic_density": math.nan,
                    "seconds": 0.0,
                    "seconds_per_sample": 0.0,
                    "error": str(exc),
                }
                all_results.append(failure)
                print(f"FAILED: locale={locale_cfg['locale']}, model={model_size}, language=auto")
                print(exc)

    save_json(os.path.join(run_dir, "summary.json"), all_results)
    save_results_table(run_dir, all_results)
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
