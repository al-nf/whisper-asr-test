"""Verdict tool for eval_aishell_ngram_fusion.py: turns the raw per-condition
CER/WER numbers into an explicit statement about the hypothesis under test -

    "Chinese ASR benefits most from a low-order (bi/tri-gram) n-gram LM,
    unlike the 5-gram order typically used for space-delimited languages."

For each tokenization scheme, this:
  1. Prints a RER-vs-order table (the raw evidence).
  2. Finds the empirical best order (lowest eval CER).
  3. Runs a paired bootstrap over eval-set utterances (resampling with
     replacement, recomputing corpus-level CER on each resample) to get a
     confidence interval on RER for every order, and on the CER gap between
     the best order and every other order - so "order 2 is best" can be
     distinguished from "order 2 is only nominally best by noise".
  4. Prints an explicit verdict: which order(s) are statistically
     indistinguishable from the best, and whether that set includes order 2
     or 3 (confirming the hypothesis), only includes order 4/5 (refuting it),
     or the LM doesn't help at all (RER not significantly above zero).

Performance note: per-utterance edit-distance/length is computed with jiwer
exactly once per (row list, metric) via `prepare_rows` - the bootstrap loop
itself is pure integer arithmetic over precomputed arrays, not repeated
string alignment, so thousands of resamples stay fast even on the full
~5-7k-utterance eval split.

Usage:
    uv run analyze_aishell_ngram_fusion.py --run-dir ./logs/aishell_ngram_fusion
    uv run analyze_aishell_ngram_fusion.py --run-dir ./logs/aishell_ngram_fusion --bootstrap 2000
"""

import argparse
import json
import random
import statistics
from dataclasses import dataclass
from typing import List

import jiwer
from rich.console import Console
from rich.table import Table

from ngram_lm import TOKENIZERS

CI_LOW, CI_HIGH = 0.025, 0.975


def load_results(run_dir: str) -> dict:
    path = f"{run_dir}/results.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Prepared:
    """Per-utterance edit-distance numerator/denominator for CER and
    jieba-based WER, precomputed once per row list so bootstrap resampling is
    pure arithmetic."""

    cer_edits: List[int]
    cer_lens: List[int]
    wer_edits: List[int]
    wer_lens: List[int]


def prepare_rows(rows: List[dict]) -> Prepared:
    tokenize = TOKENIZERS["word"]
    cer_edits, cer_lens, wer_edits, wer_lens = [], [], [], []
    for row in rows:
        ref, hyp = row["ref"], row["hyp"]
        c = jiwer.process_characters([ref], [hyp])
        cer_edits.append(c.substitutions + c.deletions + c.insertions)
        cer_lens.append(c.hits + c.substitutions + c.deletions)

        w = jiwer.process_words([" ".join(tokenize(ref))], [" ".join(tokenize(hyp))])
        wer_edits.append(w.substitutions + w.deletions + w.insertions)
        wer_lens.append(w.hits + w.substitutions + w.deletions)
    return Prepared(cer_edits, cer_lens, wer_edits, wer_lens)


def rate_from_prepared(prepared: Prepared, sample_idx: List[int], metric: str) -> float:
    edits = prepared.cer_edits if metric == "cer" else prepared.wer_edits
    lens = prepared.cer_lens if metric == "cer" else prepared.wer_lens
    total_len = sum(lens[i] for i in sample_idx)
    if total_len == 0:
        return 0.0
    return sum(edits[i] for i in sample_idx) / total_len


def percentile(values: List[float], p: float) -> float:
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(p * (len(values) - 1)))))
    return values[idx]


def bootstrap_rer(
    baseline_prepared: Prepared, condition_prepared: Prepared, n_boot: int, seed: int, metric: str
) -> "tuple[float, float, float, float]":
    """Paired bootstrap over utterance indices (shared between baseline and
    condition rows, which are aligned 1:1 by construction in
    eval_aishell_ngram_fusion.py). Returns (rer_mean, ci_low, ci_high,
    p_no_improvement) where p_no_improvement is the fraction of resamples
    with RER <= 0."""
    n = len(baseline_prepared.cer_edits)
    rng = random.Random(seed)
    rers = []
    for _ in range(n_boot):
        sample_idx = [rng.randrange(n) for _ in range(n)]
        base_rate = rate_from_prepared(baseline_prepared, sample_idx, metric)
        cond_rate = rate_from_prepared(condition_prepared, sample_idx, metric)
        if base_rate == 0:
            continue
        rers.append((base_rate - cond_rate) / base_rate)
    if not rers:
        return float("nan"), float("nan"), float("nan"), float("nan")
    p_no_improvement = sum(1 for r in rers if r <= 0) / len(rers)
    return statistics.mean(rers), percentile(rers, CI_LOW), percentile(rers, CI_HIGH), p_no_improvement


def bootstrap_rate_gap(
    prepared_a: Prepared, prepared_b: Prepared, n_boot: int, seed: int, metric: str
) -> "tuple[float, float, float]":
    """CI on rate(a) - rate(b) over paired bootstrap resamples. Used to check
    whether the empirically-best order is *significantly* better than a
    given other order (CI excluding 0), vs. just nominally lower."""
    n = len(prepared_a.cer_edits)
    rng = random.Random(seed)
    gaps = []
    for _ in range(n_boot):
        sample_idx = [rng.randrange(n) for _ in range(n)]
        gaps.append(
            rate_from_prepared(prepared_a, sample_idx, metric) - rate_from_prepared(prepared_b, sample_idx, metric)
        )
    return statistics.mean(gaps), percentile(gaps, CI_LOW), percentile(gaps, CI_HIGH)


def render_rer_table(
    scheme: str,
    baseline: dict,
    conditions: List[dict],
    cer_stats: "dict[int, tuple]",
    wer_stats: "dict[int, tuple]",
) -> Table:
    table = Table(title=f"RER vs. n-gram order - scheme={scheme}")
    table.add_column("Order", justify="right")
    table.add_column("CER", justify="right")
    table.add_column("RER (CER)", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("P(no improvement)", justify="right")
    table.add_column("WER", justify="right")
    table.add_column("RER (WER)", justify="right")
    table.add_row("baseline", f"{baseline['cer']:.4f}", "-", "-", "-", f"{baseline['wer']:.4f}", "-")
    for c in conditions:
        mean_rer, lo, hi, p_no = cer_stats[c["order"]]
        wer_mean_rer, _, _, _ = wer_stats[c["order"]]
        table.add_row(
            str(c["order"]),
            f"{c['eval_cer']:.4f}",
            f"{mean_rer:+.1%}",
            f"[{lo:+.1%}, {hi:+.1%}]",
            f"{p_no:.1%}",
            f"{c['eval_wer']:.4f}",
            f"{wer_mean_rer:+.1%}",
        )
    return table


def verdict_for_scheme(scheme: str, baseline: dict, conditions: List[dict], n_boot: int, seed: int, console: Console) -> None:
    baseline_prepared = prepare_rows(baseline["rows"])
    condition_prepared = {c["order"]: prepare_rows(c["rows"]) for c in conditions}

    cer_stats = {
        order: bootstrap_rer(baseline_prepared, prepared, n_boot, seed + order, metric="cer")
        for order, prepared in condition_prepared.items()
    }
    wer_stats = {
        order: bootstrap_rer(baseline_prepared, prepared, n_boot, seed + 500 + order, metric="wer")
        for order, prepared in condition_prepared.items()
    }
    console.print(render_rer_table(scheme, baseline, conditions, cer_stats, wer_stats))

    best = min(conditions, key=lambda c: c["eval_cer"])
    best_mean_rer, best_lo, best_hi, best_p_no = cer_stats[best["order"]]

    if best_p_no > 0.05:
        console.print(
            f"[yellow]{scheme}: no order shows a statistically significant improvement over baseline "
            f"(best is order={best['order']}, RER={best_mean_rer:+.1%}, P(no improvement)={best_p_no:.1%}). "
            f"n-gram fusion does not help for this scheme.[/yellow]\n"
        )
        return

    # Orders statistically indistinguishable from the best (CI on the CER gap includes 0).
    tied_orders = [best["order"]]
    for c in conditions:
        if c["order"] == best["order"]:
            continue
        _, lo, hi = bootstrap_rate_gap(
            condition_prepared[c["order"]], condition_prepared[best["order"]], n_boot, seed + 1000 + c["order"], metric="cer"
        )
        if lo <= 0 <= hi:
            tied_orders.append(c["order"])
    tied_orders.sort()

    low_order_supported = any(o in (2, 3) for o in tied_orders)
    verdict_color = "green" if low_order_supported else "red"
    verdict_word = "SUPPORTED" if low_order_supported else "NOT SUPPORTED"

    console.print(
        f"[bold]{scheme}[/bold]: best order={best['order']} "
        f"(CER={best['eval_cer']:.4f}, RER={best_mean_rer:+.1%} [{best_lo:+.1%}, {best_hi:+.1%}]). "
        f"Statistically tied with orders: {tied_orders}."
    )
    console.print(
        f"[{verdict_color}]Hypothesis (\"lowest relative error at bigram or trigram\") {verdict_word} "
        f"for scheme={scheme}: {'2 or 3 is in' if low_order_supported else 'the tied best-order set excludes 2 and 3, it is'} "
        f"{tied_orders}.[/{verdict_color}]\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute RER-vs-order significance and a verdict for the AISHELL-1 n-gram fusion hypothesis."
    )
    parser.add_argument("--run-dir", required=True, help="Run directory produced by eval_aishell_ngram_fusion.py")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Number of bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(args.run_dir)
    baseline = results["baseline"]
    conditions_by_scheme: "dict[str, list]" = {}
    for c in results["conditions"]:
        conditions_by_scheme.setdefault(c["scheme"], []).append(c)
    for conditions in conditions_by_scheme.values():
        conditions.sort(key=lambda c: c["order"])

    console = Console()
    console.print(
        f"[bold]AISHELL-1 n-gram fusion analysis[/bold] "
        f"(n_eval={results['n_eval']}, bootstrap={args.bootstrap} resamples)\n"
    )
    for scheme, conditions in conditions_by_scheme.items():
        verdict_for_scheme(scheme, baseline, conditions, args.bootstrap, args.seed, console)


if __name__ == "__main__":
    main()
