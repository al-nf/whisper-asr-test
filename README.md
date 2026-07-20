# Dependencies
### [uv](https://docs.astral.sh/uv/)

# Usage
`uv run eval.py`

## LLM postprocessing test

`eval_llm_postprocess.py` measures how much a local LLM correction pass reduces
Whisper's WER/CER. For each sample it runs Whisper to get a raw transcript,
then feeds that transcript through a locally-hosted instruct LLM (default:
`Qwen/Qwen2.5-7B-Instruct`, bf16, ~15GB VRAM — fits comfortably on a single
RTX 3090) with a correction prompt, and reports WER/CER before vs. after.

```
uv run eval_llm_postprocess.py
uv run eval_llm_postprocess.py --locales en_us es_419 --max-samples 50
uv run eval_llm_postprocess.py --llm-model Qwen/Qwen2.5-7B-Instruct --load-in-4bit
```

Results (raw/llm WER & CER, per-sample transcripts, and the resolved
`run_config.json`) are saved under `./logs/llm_postprocess/`.

### Capacity A/B testing

If corrections look worse than baseline, it may be a model-capability issue
(the 7B default can violate its own conservative-correction rules — e.g.
overwriting an already-correct word with a "more plausible" hallucination).
`--llm-model` accepts capacity presets to A/B test this on the same locales,
all sized to fit safely on a single 24GB 3090:

| Preset   | Model                  | VRAM (approx) |
|----------|-------------------------|----------------|
| `small`  | Qwen2.5-7B-Instruct (bf16)  | ~15GB |
| `medium` | Qwen2.5-14B-Instruct (int4) | ~9GB  |
| `large`  | Qwen2.5-32B-Instruct (int4) | ~19GB |

```
uv run eval_llm_postprocess.py --llm-model medium
uv run eval_llm_postprocess.py --llm-model large --llm-batch-size 2
```

`large` is tight on VRAM alongside Whisper + batched KV cache — drop
`--llm-batch-size` if you hit OOMs.

### Triage beyond aggregate WER/CER

Aggregate WER/CER (and even sample-level WER) can't distinguish "the LLM
overwrote a word ASR already had right" from "ASR already had a real error
and the LLM's fix just doesn't match the reference's exact wording." A
handful of the latter (inherently ambiguous references, defensible
rewordings) can dominate a small sample and make the approach look worse
than it is. `analyze_llm_postprocess.py` word-aligns `ref`<->`raw_hyp` and
`raw_hyp`<->`llm_hyp` (via jiwer edit-ops) so every individual LLM edit is
classified by whether the *specific token it touched* was already correct:

- **Broke correct word** — ASR had it right; the LLM overwrote it with
  something wrong (the damning failure mode).
- **Fix → matches ref** — ASR was already wrong there, and the LLM's edit
  landed exactly on what the reference says (a genuine, unambiguous win).
- **Fix → still wrong** — ASR was already wrong there, and the LLM changed
  it to a *different* wrong answer (didn't help, but didn't break anything
  that was working either).
- **Ungrounded insert** — the LLM added a word with no counterpart in the
  raw ASR output at all (a rule-6 violation, even if the addition reads as
  linguistically reasonable).

Adjacent edits are merged into single units before matching (e.g. a raw
two-word span collapsing into one corrected word, like `a parte` → `aparte`,
is judged as one edit against the reference — not two separate "still wrong"
half-edits).

```
uv run analyze_llm_postprocess.py --run-dir ./logs/llm_postprocess
uv run analyze_llm_postprocess.py --run-dir ./logs/llm_postprocess --show-examples 3
```
