"""Triage tool for eval_llm_postprocess.py output.

Aggregate WER/CER (and even sample-level WER) can't tell "the LLM overwrote
a word ASR already had right" (damning) apart from "ASR already had a real
error here and the LLM's fix just doesn't match the reference's exact
wording" (much more ambiguous - a metric-brittleness issue, not necessarily
an approach-brittleness issue) - especially once other, unrelated noise
elsewhere in the same sentence pushes the whole-sample WER above zero even
when the specific word the LLM touched was fine.

This script aligns ref<->raw_hyp and raw_hyp<->llm_hyp word-by-word (via
jiwer's edit-op alignment) so every individual LLM edit is classified by
whether the *specific raw token it touched* was already correct, not by
whether the whole sentence happened to be a perfect match.

Usage:
    python analyze_llm_postprocess.py --run-dir ./logs/llm_postprocess
    python analyze_llm_postprocess.py --run-dir ./logs/llm_postprocess --show-examples 3
"""

import argparse
import glob
import json
import os
import re
import statistics
from typing import List

import jiwer
from rich.console import Console
from rich.table import Table

SKIP_FILES = {"summary.json", "run_config.json"}


def normalize_text(text: str, lang: str) -> str:
    """Mirrors eval_llm_postprocess.normalize_text (duplicated to avoid pulling
    in torch/transformers/whisper just for this lightweight triage tool)."""
    if lang == "zh":
        return re.sub(r"[^\w]", "", text, flags=re.UNICODE)
    text = text.lower()
    return re.sub(r"[^\w\s]", "", text, flags=re.UNICODE).strip()


def load_locale_results(run_dir: str) -> List[dict]:
    results = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        if os.path.basename(path) in SKIP_FILES:
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if "rows" in payload:
            results.append(payload)
    return results


def tokenize(text: str, lang: str) -> List[str]:
    return list(text) if lang == "zh" else text.split()


def word_level_audit(ref: str, raw_hyp: str, llm_hyp: str, lang: str) -> dict:
    """Classify every raw_hyp->llm_hyp edit by whether it touched a raw token
    that was already correct (per the ref<->raw_hyp alignment)."""
    ref_tokens = tokenize(normalize_text(ref, lang), lang)
    raw_tokens = tokenize(normalize_text(raw_hyp, lang), lang)
    llm_tokens = tokenize(normalize_text(llm_hyp, lang), lang)

    ref_raw = jiwer.process_words([" ".join(ref_tokens)], [" ".join(raw_tokens)])
    raw_llm = jiwer.process_words([" ".join(raw_tokens)], [" ".join(llm_tokens)])

    raw_correct = [False] * len(raw_tokens)
    for chunk in ref_raw.alignments[0]:
        if chunk.type == "equal":
            for idx in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                raw_correct[idx] = True

    broke_correct, attempted_fix, ungrounded_inserts = [], [], []
    for chunk in raw_llm.alignments[0]:
        if chunk.type == "equal":
            continue
        if chunk.type == "insert":
            ungrounded_inserts.append(chunk)
            continue
        rs, re = chunk.ref_start_idx, chunk.ref_end_idx  # indices into raw_tokens
        edit = {
            "chunk": chunk,
            "raw_span": raw_tokens[rs:re],
            "llm_span": llm_tokens[chunk.hyp_start_idx : chunk.hyp_end_idx],
        }
        if any(raw_correct[rs:re]):
            broke_correct.append(edit)
        else:
            attempted_fix.append(edit)

    return {
        "broke_correct": broke_correct,
        "attempted_fix": attempted_fix,
        "ungrounded_inserts": [llm_tokens[c.hyp_start_idx : c.hyp_end_idx] for c in ungrounded_inserts],
        "raw_tokens": raw_tokens,
        "llm_tokens": llm_tokens,
    }


def summarize_locale(result: dict) -> dict:
    lang = result["lang"]
    totals = {"broke_correct": 0, "attempted_fix": 0, "ungrounded_inserts": 0}
    samples_with_broke_correct = 0
    deltas = []
    audits = []
    for row in result["rows"]:
        audit = word_level_audit(row["ref"], row["raw_hyp"], row["llm_hyp"], lang)
        audits.append(audit)
        totals["broke_correct"] += len(audit["broke_correct"])
        totals["attempted_fix"] += len(audit["attempted_fix"])
        totals["ungrounded_inserts"] += len(audit["ungrounded_inserts"])
        if audit["broke_correct"]:
            samples_with_broke_correct += 1
        deltas.append(row["sample_wer_raw"] - row["sample_wer_llm"])

    n = len(result["rows"])
    return {
        "locale": result["locale"],
        "n": n,
        **totals,
        "samples_with_broke_correct": samples_with_broke_correct,
        "mean_delta": statistics.mean(deltas) if deltas else float("nan"),
        "median_delta": statistics.median(deltas) if deltas else float("nan"),
        "baseline_wer": result.get("baseline_wer"),
        "llm_wer": result.get("llm_wer"),
        "audits": audits,
    }


def render_summary_table(summaries: List[dict]) -> Table:
    table = Table(title="Word-Level Edit Audit (counts across all edits made by the LLM)")
    table.add_column("Locale", justify="left")
    table.add_column("N", justify="right")
    table.add_column("Broke Correct Word", justify="right")
    table.add_column("Attempted Fix", justify="right")
    table.add_column("Ungrounded Insert", justify="right")
    table.add_column("Samples w/ Broke", justify="right")
    table.add_column("Mean WER Δ", justify="right")
    for s in summaries:
        table.add_row(
            s["locale"],
            str(s["n"]),
            str(s["broke_correct"]),
            str(s["attempted_fix"]),
            str(s["ungrounded_inserts"]),
            f"{s['samples_with_broke_correct']} ({s['samples_with_broke_correct'] / s['n']:.0%})",
            f"{s['mean_delta']:+.4f}",
        )
    return table


def print_examples(result: dict, audits: List[dict], k: int) -> None:
    console = Console()
    rows_with_audits = list(zip(result["rows"], audits))

    broke = [(r, a) for r, a in rows_with_audits if a["broke_correct"]]
    broke.sort(key=lambda ra: len(ra[1]["broke_correct"]), reverse=True)
    if broke:
        console.rule(f"{result['locale']}: LLM overwrote an already-correct word (showing {min(k, len(broke))}/{len(broke)})")
        for row, audit in broke[:k]:
            console.print(f"[bold]ref[/bold]:     {row['ref']}")
            console.print(f"[bold]raw_hyp[/bold]: {row['raw_hyp'].strip()}")
            console.print(f"[bold]llm_hyp[/bold]: {row['llm_hyp'].strip()}")
            for edit in audit["broke_correct"]:
                console.print(f"  [red]{' '.join(edit['raw_span'])!r} -> {' '.join(edit['llm_span'])!r}[/red] (was correct)")
            console.print()

    inserts = [(r, a) for r, a in rows_with_audits if a["ungrounded_inserts"]]
    if inserts:
        console.rule(f"{result['locale']}: LLM inserted words with no raw-token counterpart (showing {min(k, len(inserts))}/{len(inserts)})")
        for row, audit in inserts[:k]:
            console.print(f"[bold]ref[/bold]:     {row['ref']}")
            console.print(f"[bold]raw_hyp[/bold]: {row['raw_hyp'].strip()}")
            console.print(f"[bold]llm_hyp[/bold]: {row['llm_hyp'].strip()}")
            for span in audit["ungrounded_inserts"]:
                console.print(f"  [yellow]inserted: {' '.join(span)!r}[/yellow]")
            console.print()

    fixes = [(r, a) for r, a in rows_with_audits if a["attempted_fix"] and not a["broke_correct"] and row_improved(r)]
    fixes.sort(key=lambda ra: ra[0]["sample_wer_raw"] - ra[0]["sample_wer_llm"], reverse=True)
    if fixes:
        console.rule(f"{result['locale']}: LLM attempted a fix of a genuine ASR error (showing {min(k, len(fixes))}/{len(fixes)})")
        for row, audit in fixes[:k]:
            console.print(f"[bold]ref[/bold]:     {row['ref']}")
            console.print(f"[bold]raw_hyp[/bold]: {row['raw_hyp'].strip()}")
            console.print(f"[bold]llm_hyp[/bold]: {row['llm_hyp'].strip()}")
            for edit in audit["attempted_fix"]:
                console.print(f"  [green]{' '.join(edit['raw_span'])!r} -> {' '.join(edit['llm_span'])!r}[/green] (raw was already wrong)")
            console.print()


def row_improved(row: dict) -> bool:
    return row["sample_wer_llm"] < row["sample_wer_raw"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Word-level triage of eval_llm_postprocess.py results, beyond aggregate WER/CER."
    )
    parser.add_argument("--run-dir", required=True, help="Run directory under ./logs/llm_postprocess*")
    parser.add_argument(
        "--show-examples",
        type=int,
        default=0,
        help="Print up to N examples per category per locale for manual review.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_locale_results(args.run_dir)
    if not results:
        print(f"No per-locale result files found in {args.run_dir}")
        return

    summaries = [summarize_locale(r) for r in results]
    Console().print(render_summary_table(summaries))

    if args.show_examples > 0:
        for result, summary in zip(results, summaries):
            print_examples(result, summary["audits"], args.show_examples)


if __name__ == "__main__":
    main()
