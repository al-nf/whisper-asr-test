"""Verdict tool for eval_aishell_ngram_fusion.py: turns the raw per-condition
error numbers into an explicit statement about the hypothesis under test.

For Chinese-family runs (zh/yue/hak), the hypothesis is:

    "Chinese ASR benefits most from a low-order (bi/tri-gram) n-gram LM,
    unlike the 5-gram order typically used for space-delimited languages."

For English (`--lang en`, the negative control), the hypothesis is inverted:

    "English word n-grams keep improving through order 4/5 rather than
    saturating at 2/3" — if this is *not* supported, the Chinese result may
    be an artifact of N-best rescoring rather than of character-dense
    tokenization.

The metric is CER for Chinese-family languages (WER is not used there:
Mandarin has no native word boundaries) and WER for English. `eval_aishell_ngram_fusion.py`
stores the chosen metric's value under the historical `cer`/`eval_cer` keys
so existing result JSON still loads; this script reads `run_config.metric`
to decide which jiwer alignment to bootstrap.

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
exactly once per row list via `prepare_rows` - the bootstrap loop itself is
pure integer arithmetic over precomputed arrays, not repeated string
alignment, so thousands of resamples stay fast even on the full
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

CI_LOW, CI_HIGH = 0.025, 0.975


def load_results(run_dir: str) -> dict:
    path = f"{run_dir}/results.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Prepared:
    """Per-utterance CER edit-distance numerator/denominator, precomputed
    once per row list so bootstrap resampling is pure arithmetic."""

    cer_edits: List[int]
    cer_lens: List[int]


def prepare_rows(rows: List[dict], metric: str = "cer") -> Prepared:
    cer_edits, cer_lens = [], []
    for row in rows:
        if metric == "wer":
            c = jiwer.process_words([row["ref"]], [row["hyp"]])
        else:
            c = jiwer.process_characters([row["ref"]], [row["hyp"]])
        cer_edits.append(c.substitutions + c.deletions + c.insertions)
        cer_lens.append(c.hits + c.substitutions + c.deletions)
    return Prepared(cer_edits, cer_lens)


def rate_from_prepared(prepared: Prepared, sample_idx: List[int]) -> float:
    total_len = sum(prepared.cer_lens[i] for i in sample_idx)
    if total_len == 0:
        return 0.0
    return sum(prepared.cer_edits[i] for i in sample_idx) / total_len


def percentile(values: List[float], p: float) -> float:
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(p * (len(values) - 1)))))
    return values[idx]


def bootstrap_rer(
    baseline_prepared: Prepared, condition_prepared: Prepared, n_boot: int, seed: int
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
        base_rate = rate_from_prepared(baseline_prepared, sample_idx)
        cond_rate = rate_from_prepared(condition_prepared, sample_idx)
        if base_rate == 0:
            continue
        rers.append((base_rate - cond_rate) / base_rate)
    if not rers:
        return float("nan"), float("nan"), float("nan"), float("nan")
    p_no_improvement = sum(1 for r in rers if r <= 0) / len(rers)
    return statistics.mean(rers), percentile(rers, CI_LOW), percentile(rers, CI_HIGH), p_no_improvement


def bootstrap_rate_gap(
    prepared_a: Prepared, prepared_b: Prepared, n_boot: int, seed: int
) -> "tuple[float, float, float]":
    """CI on rate(a) - rate(b) over paired bootstrap resamples. Used to check
    whether the empirically-best order is *significantly* better than a
    given other order (CI excluding 0), vs. just nominally lower."""
    n = len(prepared_a.cer_edits)
    rng = random.Random(seed)
    gaps = []
    for _ in range(n_boot):
        sample_idx = [rng.randrange(n) for _ in range(n)]
        gaps.append(rate_from_prepared(prepared_a, sample_idx) - rate_from_prepared(prepared_b, sample_idx))
    return statistics.mean(gaps), percentile(gaps, CI_LOW), percentile(gaps, CI_HIGH)


def render_rer_table(
    scheme: str,
    baseline: dict,
    conditions: List[dict],
    cer_stats: "dict[int, tuple]",
    metric: str = "cer",
) -> Table:
    metric_label = metric.upper()
    table = Table(title=f"RER vs. n-gram order - scheme={scheme}")
    table.add_column("Order", justify="right")
    table.add_column(metric_label, justify="right")
    table.add_column(f"RER ({metric_label})", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("P(no improvement)", justify="right")
    table.add_row("baseline", f"{baseline['cer']:.4f}", "-", "-", "-")
    for c in conditions:
        mean_rer, lo, hi, p_no = cer_stats[c["order"]]
        table.add_row(
            str(c["order"]),
            f"{c['eval_cer']:.4f}",
            f"{mean_rer:+.1%}",
            f"[{lo:+.1%}, {hi:+.1%}]",
            f"{p_no:.1%}",
        )
    return table


def verdict_for_scheme(
    scheme: str,
    baseline: dict,
    conditions: List[dict],
    n_boot: int,
    seed: int,
    console: Console,
    metric: str = "cer",
    lang: str = "zh",
) -> None:
    baseline_prepared = prepare_rows(baseline["rows"], metric=metric)
    condition_prepared = {c["order"]: prepare_rows(c["rows"], metric=metric) for c in conditions}

    cer_stats = {
        order: bootstrap_rer(baseline_prepared, prepared, n_boot, seed + order)
        for order, prepared in condition_prepared.items()
    }
    console.print(render_rer_table(scheme, baseline, conditions, cer_stats, metric=metric))

    best = min(conditions, key=lambda c: c["eval_cer"])
    best_mean_rer, best_lo, best_hi, best_p_no = cer_stats[best["order"]]
    metric_label = metric.upper()

    if best_p_no > 0.05:
        console.print(
            f"[yellow]{scheme}: no order shows a statistically significant improvement over baseline "
            f"(best is order={best['order']}, RER={best_mean_rer:+.1%}, P(no improvement)={best_p_no:.1%}). "
            f"n-gram fusion does not help for this scheme.[/yellow]\n"
        )
        return

    # Orders statistically indistinguishable from the best (CI on the error gap includes 0).
    tied_orders = [best["order"]]
    for c in conditions:
        if c["order"] == best["order"]:
            continue
        _, lo, hi = bootstrap_rate_gap(
            condition_prepared[c["order"]], condition_prepared[best["order"]], n_boot, seed + 1000 + c["order"]
        )
        if lo <= 0 <= hi:
            tied_orders.append(c["order"])
    tied_orders.sort()

    console.print(
        f"[bold]{scheme}[/bold]: best order={best['order']} "
        f"({metric_label}={best['eval_cer']:.4f}, RER={best_mean_rer:+.1%} [{best_lo:+.1%}, {best_hi:+.1%}]). "
        f"Statistically tied with orders: {tied_orders}."
    )

    if lang == "en":
        # Negative control: English should still need 4/5-gram. Supported when
        # the tied-best set does not include 2 or 3.
        high_order_supported = not any(o in (2, 3) for o in tied_orders)
        verdict_color = "green" if high_order_supported else "red"
        verdict_word = "SUPPORTED" if high_order_supported else "NOT SUPPORTED"
        console.print(
            f"[{verdict_color}]Negative control (\"English keeps improving through 4/5-gram, unlike Chinese\") "
            f"{verdict_word} for scheme={scheme}: tied best-order set is {tied_orders}."
            f"[/{verdict_color}]\n"
        )
        return

    low_order_supported = any(o in (2, 3) for o in tied_orders)
    verdict_color = "green" if low_order_supported else "red"
    verdict_word = "SUPPORTED" if low_order_supported else "NOT SUPPORTED"
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

    dataset_repo = results.get("run_config", {}).get("dataset_repo", "AISHELL-1")
    lang = results.get("run_config", {}).get("lang", "zh")
    metric = results.get("run_config", {}).get("metric") or ("wer" if lang == "en" else "cer")
    console = Console()
    console.print(
        f"[bold]{dataset_repo} n-gram fusion analysis[/bold] "
        f"(n_eval={results['n_eval']}, bootstrap={args.bootstrap} resamples, metric={metric})\n"
    )
    for scheme, conditions in conditions_by_scheme.items():
        verdict_for_scheme(
            scheme, baseline, conditions, args.bootstrap, args.seed, console, metric=metric, lang=lang
        )


if __name__ == "__main__":
    main()
