"""Diagnose why n-gram rescoring shows zero effect (alpha*=0.00, CER/WER
identical to baseline for every scheme/order) in eval_aishell_ngram_fusion.py.

That exact pattern - a dead tie for *every* alpha in the grid, for *all*
8 (scheme, order) conditions simultaneously - is very unlikely to mean "n-gram
fusion genuinely never helps here" and much more likely means the N-best
candidate lists have collapsed to a single unique hypothesis per utterance
(beam search on a narrowly fine-tuned, high-confidence model can produce beams
that are near-duplicates and get deduped down to one candidate - leaving
nothing for any LM to rescore, regardless of order or alpha).

This script only needs `nbest.json` (no KenLM/jieba required) and reports:

  1. Candidate-count distribution - the direct test of the collapse theory.
     If almost every utterance has exactly 1 unique candidate, rescoring is
     structurally impossible no matter how alpha is tuned.
  2. Acoustic score gap between rank-0 and rank-1 candidates - how "peaked"
     beam search is even when it does produce >1 candidate.
  3. Oracle CER: the best-case CER if you picked, for every utterance, the
     candidate closest to the reference (impossible to know in practice, but
     an upper bound on what *any* N-best rescoring method could achieve). If
     oracle CER ~= baseline CER, the fix for most errors isn't in the N-best
     list at all - no amount of LM tuning can help without generating a
     wider/more diverse N-best first (bigger --num-beams, sampling, or
     diverse beam search).

Usage:
    uv run diagnose_nbest.py ./logs/aishell_ngram_fusion/nbest.json
"""

import argparse
import json
import sys
from typing import List

import jiwer
from rich.console import Console
from rich.table import Table


def cer_edits_len(ref: str, hyp: str) -> "tuple[int, int]":
    if not ref and not hyp:
        return 0, 0
    c = jiwer.process_characters([ref], [hyp])
    return c.substitutions + c.deletions + c.insertions, c.hits + c.substitutions + c.deletions


def corpus_cer(pairs: List["tuple[str, str]"]) -> float:
    total_edits = total_len = 0
    for ref, hyp in pairs:
        e, l = cer_edits_len(ref, hyp)
        total_edits += e
        total_len += l
    return total_edits / total_len if total_len else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nbest_path", help="Path to nbest.json produced by eval_aishell_ngram_fusion.py")
    args = parser.parse_args()

    with open(args.nbest_path, "r", encoding="utf-8") as f:
        nbest = json.load(f)

    console = Console()
    n = len(nbest)
    console.print(f"[bold]{n}[/bold] utterances in {args.nbest_path}\n")

    # 1. Candidate-count distribution.
    counts = [len(item["candidates"]) for item in nbest]
    hist = {}
    for c in counts:
        hist[c] = hist.get(c, 0) + 1
    table = Table(title="Unique-candidate count per utterance (after N-best dedup)")
    table.add_column("# unique candidates", justify="right")
    table.add_column("# utterances", justify="right")
    table.add_column("%", justify="right")
    for c in sorted(hist):
        table.add_row(str(c), str(hist[c]), f"{hist[c] / n:.1%}")
    console.print(table)
    n_singleton = hist.get(1, 0)
    console.print(
        f"[bold]{n_singleton}/{n} ({n_singleton / n:.1%})[/bold] utterances have exactly 1 unique candidate "
        "- rescoring is a structural no-op for these regardless of alpha or LM order.\n"
    )

    # 2. Acoustic score gap between rank-0 and rank-1 (for utterances with >=2 candidates).
    gaps = [
        item["candidates"][0]["acoustic_avg_logprob"] - item["candidates"][1]["acoustic_avg_logprob"]
        for item in nbest
        if len(item["candidates"]) >= 2
    ]
    if gaps:
        gaps_sorted = sorted(gaps)
        median_gap = gaps_sorted[len(gaps_sorted) // 2]
        console.print(
            f"Among utterances with >=2 candidates: median (rank0 - rank1) acoustic avg-logprob gap = "
            f"{median_gap:.4f} nats/token. Large gaps mean alpha would need to be huge for the LM term "
            f"to ever flip the decision.\n"
        )

    # 3. Oracle CER (best of N-best vs. reference, by edit distance) vs. baseline (rank-0) CER.
    baseline_pairs = [(item["ref"], item["candidates"][0]["text"]) for item in nbest]
    oracle_pairs = []
    n_oracle_helps = 0
    for item in nbest:
        ref = item["ref"]
        best_text, best_edits = item["candidates"][0]["text"], None
        for cand in item["candidates"]:
            e, _ = cer_edits_len(ref, cand["text"])
            if best_edits is None or e < best_edits:
                best_edits, best_text = e, cand["text"]
        oracle_pairs.append((ref, best_text))
        if best_text != item["candidates"][0]["text"]:
            n_oracle_helps += 1

    baseline_cer = corpus_cer(baseline_pairs)
    oracle_cer = corpus_cer(oracle_pairs)
    oracle_rer = (baseline_cer - oracle_cer) / baseline_cer if baseline_cer else float("nan")

    console.print(f"Baseline (rank-0) CER: [bold]{baseline_cer:.4f}[/bold]")
    console.print(
        f"Oracle CER (best possible pick from the existing N-best, by edit distance to ref): "
        f"[bold]{oracle_cer:.4f}[/bold]  (RER={oracle_rer:+.1%})"
    )
    console.print(
        f"[bold]{n_oracle_helps}/{n} ({n_oracle_helps / n:.1%})[/bold] utterances have a *non-rank-0* candidate "
        "that is strictly closer to the reference than the top acoustic pick.\n"
    )

    if oracle_rer < 0.02:
        console.print(
            "[yellow]Verdict: oracle RER is near zero - the correct fix essentially never exists anywhere in "
            "the N-best list. No LM, order, or alpha can help here; the bottleneck is N-best diversity, not "
            "the rescoring LM. Try increasing --num-beams substantially, or switching to sampling / diverse "
            "beam search (num_beam_groups + diversity_penalty) when generating candidates.[/yellow]"
        )
    elif n_singleton / n > 0.5:
        console.print(
            "[yellow]Verdict: more than half of utterances collapsed to a single unique candidate after "
            "dedup - increase --num-beams, or generate N-best via sampling instead of plain beam search, "
            "to get real diversity.[/yellow]"
        )
    else:
        console.print(
            "[green]Verdict: oracle RER is meaningfully positive and most utterances have real candidate "
            "diversity - the N-best list does contain fixes. If eval_aishell_ngram_fusion.py still reports "
            "alpha*=0.00 for every condition, that points to a bug in the LM scoring/combination itself "
            "(e.g. a tokenization/normalization mismatch between the LM training corpus and eval-time "
            "candidates causing near-uniform heavy OOV penalties) rather than a lack of headroom.[/green]"
        )


if __name__ == "__main__":
    main()
