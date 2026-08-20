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


def _expected_ref_span(ref_raw_chunks, ref_tokens, rs: int, re: int) -> List[str]:
    """What the reference actually says over raw-token range [rs, re) - i.e. the
    correct replacement for that span, per the ref<->raw_hyp alignment. Includes
    ref-only words dropped entirely from raw_hyp (zero-width delete chunks) that
    sit at the boundaries of the span."""
    spans = []
    for chunk in ref_raw_chunks:
        overlaps = chunk.hyp_start_idx < re and chunk.hyp_end_idx > rs
        boundary_delete = chunk.type == "delete" and rs <= chunk.hyp_start_idx <= re
        if overlaps or boundary_delete:
            spans.append((chunk.ref_start_idx, chunk.ref_end_idx))
    spans.sort()
    tokens = []
    for rstart, rend in spans:
        tokens.extend(ref_tokens[rstart:rend])
    return tokens


def _merge_edit_segments(chunks) -> List[dict]:
    """Merge maximal runs of consecutive non-'equal' chunks into single edit
    units. jiwer's alignment can split one conceptual edit (e.g. two raw
    tokens 'a parte' collapsing into one llm token 'aparte') into separate
    substitute/delete/insert chunks; treating each atomically would wrongly
    judge a legitimate merge-fix as "still wrong" on its own pieces."""
    segments = []
    current = []
    for chunk in chunks:
        if chunk.type == "equal":
            if current:
                segments.append(current)
                current = []
        else:
            current.append(chunk)
    if current:
        segments.append(current)

    merged = []
    for seg in segments:
        rs = min(c.ref_start_idx for c in seg)
        re = max(c.ref_end_idx for c in seg)
        ls = min(c.hyp_start_idx for c in seg)
        le = max(c.hyp_end_idx for c in seg)
        merged.append({"raw_start": rs, "raw_end": re, "llm_start": ls, "llm_end": le})
    return merged


def word_level_audit(ref: str, raw_hyp: str, llm_hyp: str, lang: str) -> dict:
    """Classify every raw_hyp->llm_hyp edit by whether it touched a raw token
    that was already correct (per the ref<->raw_hyp alignment), and - for edits
    to already-wrong tokens - whether the LLM's replacement actually landed on
    what the reference says, vs. just swapping one wrong guess for another."""
    ref_tokens = tokenize(normalize_text(ref, lang), lang)
    raw_tokens = tokenize(normalize_text(raw_hyp, lang), lang)
    llm_tokens = tokenize(normalize_text(llm_hyp, lang), lang)

    ref_raw = jiwer.process_words([" ".join(ref_tokens)], [" ".join(raw_tokens)])
    raw_llm = jiwer.process_words([" ".join(raw_tokens)], [" ".join(llm_tokens)])
    ref_raw_chunks = ref_raw.alignments[0]

    raw_correct = [False] * len(raw_tokens)
    for chunk in ref_raw_chunks:
        if chunk.type == "equal":
            for idx in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                raw_correct[idx] = True

    broke_correct, fix_matched, fix_unmatched, ungrounded_inserts = [], [], [], []
    for seg in _merge_edit_segments(raw_llm.alignments[0]):
        rs, re = seg["raw_start"], seg["raw_end"]
        llm_span = llm_tokens[seg["llm_start"] : seg["llm_end"]]
        if rs == re:
            # No raw tokens at all in this span - a pure insertion with nothing to ground it.
            ungrounded_inserts.append(llm_span)
            continue
        edit = {"raw_span": raw_tokens[rs:re], "llm_span": llm_span}
        if any(raw_correct[rs:re]):
            broke_correct.append(edit)
            continue
        expected = _expected_ref_span(ref_raw_chunks, ref_tokens, rs, re)
        edit["expected_ref_span"] = expected
        if llm_span == expected:
            fix_matched.append(edit)
        else:
            fix_unmatched.append(edit)

    return {
        "broke_correct": broke_correct,
        "fix_matched": fix_matched,
        "fix_unmatched": fix_unmatched,
        "ungrounded_inserts": ungrounded_inserts,
        "raw_tokens": raw_tokens,
        "llm_tokens": llm_tokens,
    }


def select_metric(result: dict) -> str:
    """WER degenerates for zh/yue: normalize_text produces a whitespace-free
    string, so jiwer.wer treats the whole sentence as one token - a
    near-binary "exact match or not" per sample. CER is the metric that
    actually carries signal for character-based languages. Falls back to
    WER if a run predates per-sample CER tracking."""
    rows = result.get("rows") or []
    has_cer = bool(rows) and "sample_cer_raw" in rows[0]
    return "cer" if result.get("lang") == "zh" and has_cer else "wer"


def summarize_locale(result: dict) -> dict:
    lang = result["lang"]
    totals = {"broke_correct": 0, "fix_matched": 0, "fix_unmatched": 0, "ungrounded_inserts": 0}
    samples_with_broke_correct = 0
    format_noncompliant = 0
    metric = select_metric(result)

    deltas = []
    audits = []
    for row in result["rows"]:
        audit = word_level_audit(row["ref"], row["raw_hyp"], row["llm_hyp"], lang)
        audits.append(audit)
        totals["broke_correct"] += len(audit["broke_correct"])
        totals["fix_matched"] += len(audit["fix_matched"])
        totals["fix_unmatched"] += len(audit["fix_unmatched"])
        totals["ungrounded_inserts"] += len(audit["ungrounded_inserts"])
        if audit["broke_correct"]:
            samples_with_broke_correct += 1
        if row.get("format_noncompliant"):
            format_noncompliant += 1
        deltas.append(row[f"sample_{metric}_raw"] - row[f"sample_{metric}_llm"])

    n = len(result["rows"])
    return {
        "locale": result["locale"],
        "n": n,
        **totals,
        "samples_with_broke_correct": samples_with_broke_correct,
        "format_noncompliant": format_noncompliant,
        "metric": metric,
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
    table.add_column("Broke Correct", justify="right")
    table.add_column("Fix -> Matches Ref", justify="right")
    table.add_column("Fix -> Still Wrong", justify="right")
    table.add_column("Ungrounded Insert", justify="right")
    table.add_column("Samples w/ Broke", justify="right")
    table.add_column("Fmt Noncompliant", justify="right")
    table.add_column("Metric", justify="left")
    table.add_column("Mean Δ", justify="right")
    for s in summaries:
        table.add_row(
            s["locale"],
            str(s["n"]),
            str(s["broke_correct"]),
            str(s["fix_matched"]),
            str(s["fix_unmatched"]),
            str(s["ungrounded_inserts"]),
            f"{s['samples_with_broke_correct']} ({s['samples_with_broke_correct'] / s['n']:.0%})",
            f"{s['format_noncompliant']} ({s['format_noncompliant'] / s['n']:.0%})",
            s["metric"].upper(),
            f"{s['mean_delta']:+.4f}",
        )
    return table


def _print_row_header(console: Console, row: dict) -> None:
    console.print(f"[bold]ref[/bold]:     {row['ref']}")
    console.print(f"[bold]raw_hyp[/bold]: {row['raw_hyp'].strip()}")
    console.print(f"[bold]llm_hyp[/bold]: {row['llm_hyp'].strip()}")
    if row.get("format_noncompliant"):
        console.print(
            "  [magenta]note: model leaked reasoning/notes here; llm_hyp is after fallback "
            "extraction and may itself be unreliable[/magenta]"
        )


def print_examples(result: dict, audits: List[dict], k: int) -> None:
    console = Console()
    rows_with_audits = list(zip(result["rows"], audits))

    broke = [(r, a) for r, a in rows_with_audits if a["broke_correct"]]
    broke.sort(key=lambda ra: len(ra[1]["broke_correct"]), reverse=True)
    if broke:
        console.rule(f"{result['locale']}: LLM overwrote an already-correct word (showing {min(k, len(broke))}/{len(broke)})")
        for row, audit in broke[:k]:
            _print_row_header(console, row)
            for edit in audit["broke_correct"]:
                console.print(f"  [red]{' '.join(edit['raw_span'])!r} -> {' '.join(edit['llm_span'])!r}[/red] (was correct)")
            console.print()

    inserts = [(r, a) for r, a in rows_with_audits if a["ungrounded_inserts"]]
    if inserts:
        console.rule(f"{result['locale']}: LLM inserted words with no raw-token counterpart (showing {min(k, len(inserts))}/{len(inserts)})")
        for row, audit in inserts[:k]:
            _print_row_header(console, row)
            for span in audit["ungrounded_inserts"]:
                console.print(f"  [yellow]inserted: {' '.join(span)!r}[/yellow]")
            console.print()

    matched = [(r, a) for r, a in rows_with_audits if a["fix_matched"]]
    matched.sort(key=lambda ra: len(ra[1]["fix_matched"]), reverse=True)
    if matched:
        console.rule(f"{result['locale']}: LLM fix landed exactly on the reference (showing {min(k, len(matched))}/{len(matched)})")
        for row, audit in matched[:k]:
            _print_row_header(console, row)
            for edit in audit["fix_matched"]:
                console.print(f"  [green]{' '.join(edit['raw_span'])!r} -> {' '.join(edit['llm_span'])!r}[/green] (raw was wrong; llm now matches ref)")
            console.print()

    unmatched = [(r, a) for r, a in rows_with_audits if a["fix_unmatched"]]
    unmatched.sort(key=lambda ra: len(ra[1]["fix_unmatched"]), reverse=True)
    if unmatched:
        console.rule(f"{result['locale']}: LLM changed an already-wrong word but didn't land on the reference (showing {min(k, len(unmatched))}/{len(unmatched)})")
        for row, audit in unmatched[:k]:
            _print_row_header(console, row)
            for edit in audit["fix_unmatched"]:
                expected = " ".join(edit["expected_ref_span"]) or "(nothing - raw had an extra word)"
                console.print(
                    f"  [yellow]{' '.join(edit['raw_span'])!r} -> {' '.join(edit['llm_span'])!r}[/yellow]"
                    f" (still wrong; ref wants {expected!r})"
                )
            console.print()


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
