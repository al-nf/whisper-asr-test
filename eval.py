import re
import tempfile
import time
import os
import json
import soundfile as sf
import whisper
from datasets import load_dataset
from jiwer import wer, cer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm


LOCALES = [
    {
        "locale": "cmn_hans_cn",
        "lang": "zh"
    },
    {
        "locale": "yue_hant_hk",
        "lang": "zh"
    },
    {
        "locale": "es_419",
        "lang": "es"
    },
    {
        "locale": "gl_es",
        "lang": "gl"
    },
    {
        "locale": "fr_fr",
        "lang": "fr"
    },
    {
        "locale": "oc_fr",
        "lang": "oc"
    },
    {
        "locale": "hi_in",
        "lang": "hi"
    },
    {
        "locale": "pa_in",
        "lang": "pa"
    },
    {
        "locale": "ru_ru",
        "lang": "ru"
    },
    {
        "locale": "be_by",
        "lang": "be"
    },
]

model_sizes = [
    "tiny",
    "base",
    "small",
    # "medium",
    # "large",
]

language_settings = [
    None,
    "lang",
]

def normalize_text(text, lang):
    if lang == "zh":
        return re.sub(r"[^\w]", "", text, flags=re.UNICODE)
    text = text.lower()
    return re.sub(r"[^\w\s]", "", text, flags=re.UNICODE).strip()


def get_run_dir(locale: str) -> str:
    base = os.path.join("./logs", locale)
    if not os.path.exists(base):
        os.makedirs(base)
        return base
    counter = 1
    while os.path.exists(f"{base}_{counter}"):
        counter += 1
    path = f"{base}_{counter}"
    os.makedirs(path)
    return path


def save_pairs(run_dir: str, model_size: str, language: str, result: dict):
    model_dir = os.path.join(run_dir, model_size)
    os.makedirs(model_dir, exist_ok=True)
    pairs_path = os.path.join(model_dir, f"{language}.json")
    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump(result.get("pairs", []), f, ensure_ascii=False, indent=2)


def save_results_table(run_dir: str, all_results: list):
    console = Console(record=True, highlight=False)
    table = Table(title="Results")
    table.add_column("Locale", justify="left")
    table.add_column("Model", justify="left")
    table.add_column("Language", justify="left")
    table.add_column("Samples", justify="right")
    table.add_column("WER", justify="right")
    table.add_column("CER", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Sec/Sample", justify="right")
    for r in all_results:
        table.add_row(
            r["locale"],
            r["model"],
            r["language"],
            str(r["samples"]),
            f"{r['wer']:.4f}",
            f"{r['cer']:.4f}",
            f"{r['seconds']:.1f}s",
            f"{r['seconds_per_sample']:.2f}",
        )
    console.print(table)
    with open(os.path.join(run_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write(console.export_text())


def evaluate_model(dataset, model_size, locale, lang, language=None):
    model = whisper.load_model(model_size)
    refs = []
    hyps = []
    start = time.time()
    label = f"{locale} | {model_size} | language={language or 'auto'}"
    for row in tqdm(dataset, desc=label, unit="sample"):
        audio = row["audio"]
        ref = row["transcription"]
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, audio["array"], audio["sampling_rate"])
            kwargs = {"task": "transcribe"}
            if language is not None:
                kwargs["language"] = language
            result = model.transcribe(f.name, **kwargs)
        hyp = result["text"]
        refs.append(ref)
        hyps.append(hyp)
    elapsed = time.time() - start
    refs_clean = [normalize_text(r, lang) for r in refs]
    hyps_clean = [normalize_text(h, lang) for h in hyps]
    return {
        "locale": locale,
        "model": model_size,
        "language": language or "auto",
        "samples": len(dataset),
        "wer": wer(refs_clean, hyps_clean),
        "cer": cer(refs_clean, hyps_clean),
        "seconds": elapsed,
        "seconds_per_sample": elapsed / len(dataset),
        "pairs": [{"ref": r, "hyp": h} for r, h in zip(refs, hyps)],
    }


def print_results_table(results):
    console = Console()
    table = Table(title="Results")
    table.add_column("Locale", justify="left")
    table.add_column("Model", justify="left")
    table.add_column("Language", justify="left")
    table.add_column("Samples", justify="right")
    table.add_column("WER", justify="right")
    table.add_column("CER", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Sec/Sample", justify="right")
    for r in results:
        table.add_row(
            r["locale"],
            r["model"],
            r["language"],
            str(r["samples"]),
            f"{r['wer']:.4f}",
            f"{r['cer']:.4f}",
            f"{r['seconds']:.1f}s",
            f"{r['seconds_per_sample']:.2f}",
        )
    console.print(table)


all_results = []

for locale_cfg in LOCALES:
    locale = locale_cfg["locale"]
    lang = locale_cfg["lang"]

    print(f"\n{'='*60}")
    print(f"Loading dataset: {locale}")
    print(f"{'='*60}")
    dataset = load_dataset("google/fleurs", locale, split="test")

    run_dir = get_run_dir(locale)
    print(f"Saving logs to: {run_dir}")

    for model_size in model_sizes:
        for language in language_settings:
            resolved_language = lang if language == "lang" else language
            try:
                result = evaluate_model(
                    dataset=dataset,
                    model_size=model_size,
                    locale=locale,
                    lang=lang,
                    language=resolved_language,
                )
                all_results.append(result)
                print_results_table(all_results)
                save_pairs(run_dir, model_size, resolved_language or "auto", result)
            except Exception as e:
                all_results.append({
                    "locale": locale,
                    "model": model_size,
                    "language": resolved_language or "auto",
                    "samples": len(dataset),
                    "wer": float("nan"),
                    "cer": float("nan"),
                    "seconds": 0.0,
                    "seconds_per_sample": 0.0,
                })
                print(f"FAILED: locale={locale}, model={model_size}, language={resolved_language or 'auto'}")
                print(e)

print_results_table(all_results)

dir = "./logs"
os.makedirs(dir, exist_ok=True)
save_results_table(dir, all_results)
print(f"Results saved to: {os.path.join(dir, 'results.txt')}")
