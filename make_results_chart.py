"""Generate the RER-vs-n-gram-order chart for RESULTS.md from the validated
AISHELL-1 (Mandarin) and MDCC (Cantonese) fusion-eval runs.

Numbers are transcribed directly from the final (post-bugfix) Jetson run
output for each dataset - see RESULTS.md for the full tables and run
provenance. Not derived from local log files (those live on the Jetson).

Usage:
    uv run make_results_chart.py
"""

import matplotlib.pyplot as plt

orders = [2, 3, 4, 5]

# RER (%) relative to the no-LM baseline, by (dataset, scheme).
data = {
    ("AISHELL-1 (Mandarin)", "char"): [6.7, 8.0, 8.0, 8.2],
    ("AISHELL-1 (Mandarin)", "word"): [3.8, 3.7, 3.8, 3.8],
    ("MDCC (Cantonese)", "char"): [1.8, 3.0, 3.0, 2.9],
    ("MDCC (Cantonese)", "word"): [-0.2, 0.0, 0.0, 0.0],
}

colors = {"AISHELL-1 (Mandarin)": "tab:red", "MDCC (Cantonese)": "tab:blue"}
markers = {"char": "o", "word": "s"}
linestyles = {"char": "-", "word": "--"}

fig, ax = plt.subplots(figsize=(7, 4.5))

for (dataset, scheme), rer in data.items():
    ax.plot(
        orders,
        rer,
        color=colors[dataset],
        marker=markers[scheme],
        linestyle=linestyles[scheme],
        linewidth=2,
        markersize=7,
        label=f"{dataset} - {scheme}",
    )

ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(5, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
ax.text(4.93, -1.6, "WhisperLM convention\n(5-gram)", fontsize=8, color="gray", ha="right")

ax.set_xticks(orders)
ax.set_xlabel("N-gram order")
ax.set_ylabel("RER vs. no-LM baseline (%)")
ax.set_title("N-gram fusion RER by order: char peaks by trigram/4-gram,\nword scheme is flat")
ax.legend(loc="center right", fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(-2.5, 9.5)

fig.tight_layout()
fig.savefig("results_rer_by_order.png", dpi=150)
print("Wrote results_rer_by_order.png")
