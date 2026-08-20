# Dependencies
### [uv](https://docs.astral.sh/uv/)
### [FFmpeg](https://ffmpeg.org/) (system package)
`datasets`' audio decoding (used by every script that loads an HF audio
dataset - `eval.py`, `eval_llm_postprocess.py`, `eval_aishell_ngram_fusion.py`)
goes through `torchcodec`, which dynamically loads FFmpeg's shared libraries
(`libavutil`/`libavcodec`/`libavformat`, versions 4-8 supported) at import
time and raises `RuntimeError: Could not load libtorchcodec` if none are
found. Install it via your system package manager, e.g.:
```
sudo apt-get install -y ffmpeg
```

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

## AISHELL-1 n-gram order/tokenization fusion test

**Hypothesis:** WhisperLM-style fusion typically uses a 5-gram LM (tuned for
space-delimited languages). Mandarin's character/word statistics may instead
favor a much lower order (bigram/trigram) — this experiment measures the
relative error-rate reduction (RER) from fusing orders 2-5 with a fine-tuned
Whisper on AISHELL-1, separately for character-segmented and jieba
word-segmented LMs, to confirm or refute that. (Order 1 is excluded: KenLM's
query/loading code hard-requires at least a bigram model, even though
`lmplz` can technically produce a unigram ARPA file. The no-LM beam-search
baseline already serves as the effective "0th order" comparison point.)

**Fusion mechanism: N-best rescoring, not shallow fusion during beam search.**
Whisper's BPE tokens don't align to Chinese characters or jieba words, so
injecting an n-gram score at every decoding step would require guessing
character/word boundaries inside a partially generated BPE token. Instead:
Whisper generates K beam candidates per utterance (with their own
length-normalized acoustic log-probs); each candidate is tokenized under a
scheme (char or jieba word) and scored by the matching KenLM n-gram model;
candidates are re-ranked by `acoustic + alpha * lm` and the top one is kept.
`alpha` is grid-searched on a held-out 20% "tune" slice of the AISHELL-1 test
set (there's no separate dev-set audio in the dataset mirror used here) and
applied to the disjoint 80% "eval" slice that RER is computed on.

### 1. Set up KenLM (Jetson AGX Orin)

KenLM isn't declared in `pyproject.toml` — both the PyPI sdist and the
checked-in `python/kenlm.cpp` on GitHub were pre-generated by an old Cython
and use private CPython C-API symbols that Python 3.13 removed/changed
(`_PyGC_FINALIZED`, `_PyDict_SetItem_KnownHash`, the pre-3.13
`_PyLong_AsByteArray` signature — see
[kpu/kenlm#471](https://github.com/kpu/kenlm/issues/471)), and the CLI tools
(`lmplz`/`build_binary`) used to *train* n-gram models aren't part of the
Python package at all. `scripts/setup_kenlm_jetson.sh` installs the apt
build deps (Boost, Eigen, zlib/bz2/lzma), builds `lmplz`/`build_binary` from
source via cmake, regenerates `kenlm.cpp` from `kenlm.pyx` with a current
Cython (the fix the kenlm maintainers give in that issue), and installs the
resulting `kenlm` Python bindings into this repo's `.venv` with `MAX_ORDER=6`:

```
bash scripts/setup_kenlm_jetson.sh
export KENLM_BIN_DIR=third_party/kenlm/build/bin
```

### 2. Build the LM training corpora

Downloads only the AISHELL-1 transcript file (no audio) from `AISHELL/AISHELL-1`,
excludes every utterance id present in the `Serenalay/AISHELL-1` test split
(the split step 4 evaluates on, so the LM never sees test transcripts), and
writes char- and jieba-word-tokenized corpora from the remaining train+dev
transcripts:

```
uv run prepare_aishell_lm_corpus.py
```

### 3. Train the KenLM models

Trains orders 2-5 for both schemes (8 models total) with `lmplz` +
`build_binary`:

```
bash scripts/build_kenlm_models.sh
```

### 4. Run the fusion eval

Runs Whisper N-best generation once (cached to `nbest.json` in the run dir),
then rescoring + alpha tuning for every (order, scheme) pair:

```
uv run eval_aishell_ngram_fusion.py
uv run eval_aishell_ngram_fusion.py --max-samples 200 --num-beams 5   # quick iteration
uv run eval_aishell_ngram_fusion.py --nbest-cache ./logs/aishell_ngram_fusion/nbest.json  # re-tune without re-running ASR
```

Default ASR model is `junsor/whisper-small-aishell`; pass `--asr-model` to use
your own fine-tune. `--dtype` defaults to `auto`, which resolves to `float16`
on CUDA (roughly half the memory/time of `float32` for a beam-search-heavy
workload like this — important on Jetson) and `float32` on CPU. Results
(per-condition CER/WER/RER, `nbest.json`, `run_config.json`) are saved under
`./logs/aishell_ngram_fusion/`.

#### Jetson: crashes / resuming

The ASR (N-best generation) stage is the only part that touches the GPU, and
on Jetson it can hit:

```
RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at
".../c10/cuda/CUDACachingAllocator.cpp":1319, please report a bug to PyTorch.
```

This is a [known PyTorch-on-Tegra issue](https://github.com/pytorch/pytorch/issues/185240):
Jetson's unified memory needs a physically-contiguous DMA buffer (via NvMap)
for each CUDA allocation, and under fragmentation/CMA pressure that
allocation can fail; PyTorch then tries to query NVML for diagnostics, and
NVML's partial Tegra support turns that into an uncatchable-looking internal
assert instead of a normal `OutOfMemoryError`. The script mitigates this three
ways:

- Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` automatically
  (reduces allocator fragmentation, so the underlying failure is less likely
  in the first place).
- Catches the error around each ASR batch, frees the cache, and retries with
  the batch recursively split in half (down to one utterance); a single
  utterance that still fails is recorded with an empty hypothesis instead of
  aborting the run.
- **Checkpoints `nbest.json` after every batch**, keyed by utterance id. If
  the process is still killed outright (OOM-killer, power event, etc.),
  re-run with `--run-dir` pointing at the same directory and it resumes from
  the last completed batch instead of restarting:
  ```
  uv run eval_aishell_ngram_fusion.py --run-dir ./logs/aishell_ngram_fusion
  ```

If crashes persist, lower `--batch-size` and/or `--num-beams` further.

### 5. Analyze: confirm or refute the hypothesis

Computes RER per order/scheme, a paired bootstrap CI and P(no improvement)
per condition, and an explicit verdict on whether the tied-best order set
includes 2 or 3 (hypothesis supported) or not (refuted):

```
uv run analyze_aishell_ngram_fusion.py --run-dir ./logs/aishell_ngram_fusion
```

### Methodology notes

- **Metric.** Alpha is always tuned to minimize CER (segmentation-tool
  independent, the standard metric for Chinese) for *both* schemes, so
  RER(CER) is directly comparable across char vs. word conditions. Jieba-based
  WER (re-segmenting both ref and hyp with the same tokenizer — not the
  dataset's own pre-baked word boundaries) is reported per condition as a
  secondary diagnostic, using that same CER-tuned alpha.
- **RER.** `(baseline_error - condition_error) / baseline_error`, computed on
  the eval slice; `alpha=0` reproduces the no-LM baseline for every order as a
  built-in sanity check.
- **Significance.** `analyze_aishell_ngram_fusion.py` runs a paired bootstrap
  (utterance-level resampling with replacement, recomputing corpus-level CER
  per resample) to distinguish "order 2 is nominally best" from "order 2 is
  significantly best" — orders whose CER-gap CI against the best order
  includes 0 are reported as statistically tied.
