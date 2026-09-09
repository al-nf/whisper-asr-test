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
space-delimited languages). Chinese languages' character/word statistics may
instead favor a much lower order (bigram/trigram) — this experiment measures
the relative error-rate reduction (RER) from fusing orders 2-5 with a
fine-tuned Whisper, separately for character-segmented and word-segmented
LMs, to confirm or refute that. (Order 1 is excluded: KenLM's query/loading
code hard-requires at least a bigram model, even though `lmplz` can
technically produce a unigram ARPA file. The no-LM beam-search baseline
already serves as the effective "0th order" comparison point.)

The steps below walk through the Mandarin/AISHELL-1 setup; the whole
pipeline (`eval_aishell_ngram_fusion.py`, `analyze_aishell_ngram_fusion.py`,
`diagnose_nbest.py`) is dataset/language-agnostic and reused as-is for
Cantonese/MDCC - see [Running the same test on Cantonese
(MDCC)](#running-the-same-test-on-cantonese-mdcc) - and as a negative
control on English/LibriSpeech - see [English negative control
(LibriSpeech)](#english-negative-control-librispeech). Hakka is documented
separately and currently blocked on gated data.

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
(per-condition CER/RER, `nbest.json`, `run_config.json`) are saved under
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

#### Every condition shows `alpha*=0.00` and CER identical to baseline

This means **no alpha in the grid ever changed which candidate was picked,
for any of the 8 (scheme, order) conditions** - i.e. a dead tie, not "the LM
made things worse." That specific pattern is almost never a genuine "n-gram
fusion doesn't help" result; it's what you get when the N-best candidate
lists have collapsed to one unique hypothesis per utterance. Plain beam
search on a narrowly fine-tuned, highly confident model (AISHELL-1 is short,
clean, in-domain read speech) can produce beams that are near-duplicates and
dedup down to a single candidate - leaving nothing for any LM to rescore,
independent of order or alpha.

Check this directly against the cached `nbest.json` (no KenLM/jieba needed):

```
uv run diagnose_nbest.py ./logs/aishell_ngram_fusion/nbest.json
```

It reports the unique-candidate-count distribution, the acoustic score gap
between rank-0/rank-1 candidates, and an **oracle CER** (best-case CER if you
always picked the N-best candidate closest to the reference) vs. the
baseline CER - the gap between them is the ceiling on what any rescoring
method could achieve with that N-best list.

On `junsor/whisper-small-aishell` over the full AISHELL-1 test set, this
diagnostic came back as total collapse: **100% of the 7176 utterances had
exactly 1 unique candidate**, and oracle CER was bit-for-bit equal to
baseline CER (RER=+0.0%). This isn't a WER-vs-CER artifact - `diagnose_nbest.py`
never computes WER at all, and `eval_aishell_ngram_fusion.py`'s alpha
selection (`evaluate_condition`) already picks `best_alpha` by minimizing
`cer(refs, hyps)`, never `wer`. It also isn't an LM-scoring bug - `avg_logprob`'s
unit conversion (log10 to natural log, per-token normalization) is correct.
It means deterministic beam search's top-k expansion is re-deriving the exact
same argmax path every time: ASR posteriors on clean, in-domain, single-domain
fine-tuned audio are typically far more peaked than open-ended text
generation, so there's nothing left in the N-best list for *any* LM, order,
or alpha to rescore between.

Two diversity mechanisms exist (mutually exclusive - HF gives diverse beam
search priority if both are set), but **only one of them is currently known
to actually work**:

```
# RECOMMENDED: independent multinomial sampling - --num-beams candidates
# sampled independently instead of one deterministic beam search. Not beam
# search at all under the hood - see below - but empirically verified (against
# a real checkpoint + real audio) to produce genuine per-utterance diversity.
uv run eval_aishell_ngram_fusion.py --num-beams 5 --do-sample --temperature 1.0
```

```
# NOT RECOMMENDED (currently broken): diverse beam search groups beams and
# penalizes cross-group similarity at every step. On transformers >= ~4.62/5.x
# this was extracted out of the core library into a Hub-hosted `custom_generate`
# repo (https://hf.co/transformers-community/group-beam-search) - the script
# passes `trust_remote_code=True` for you when `--num-beam-groups > 1`, which
# needs network access on first use to download and cache that code. However,
# testing this directly against a real checkpoint + real audio at
# diversity_penalty = 0.5, 2.0, 5.0, 10.0, and 20.0 (4 orders of magnitude)
# produced byte-identical output every time - the mechanism appears to have no
# effect at all in this transformers version, not just "too weak a penalty."
# Left here for reference/future debugging, but use --do-sample instead.
uv run eval_aishell_ngram_fusion.py --num-beams 5 --num-beam-groups 5 --diversity-penalty 2.0
```

`--do-sample` is *not* HF's generic "beam-search multinomial sampling" (which
would keep beam-search bookkeeping while sampling each expansion step).
`WhisperForConditionalGeneration.generate()`'s temperature-fallback logic
(`generate_with_fallback` in `transformers.models.whisper.generation_whisper`,
its mechanism for retrying low-confidence long-form segments at higher
temperatures) unconditionally forces `num_beams=1` whenever `do_sample=True`,
regardless of what's passed in - Whisper models cannot do beam-search +
sampling together. So `--do-sample` actually runs as `--num-beams`
*independent ancestral samples*, not beam search; `acoustic_avg_logprob` is
reconstructed from `compute_transition_scores` since sampled output has no
`sequences_scores` field. This is still a legitimate, different diversity
mechanism (fully stochastic vs. diverse beam search's explicit penalty) - just
don't expect beam-search-quality candidates from it.

Re-run `diagnose_nbest.py` against the resulting `nbest.json` after either
one - if the unique-candidate-count distribution is still collapsed to 1,
push `--diversity-penalty` / `--temperature` higher before concluding
anything about the n-gram order hypothesis itself.

### 5. Analyze: confirm or refute the hypothesis

Computes RER per order/scheme, a paired bootstrap CI and P(no improvement)
per condition, and an explicit verdict on whether the tied-best order set
includes 2 or 3 (hypothesis supported) or not (refuted):

```
uv run analyze_aishell_ngram_fusion.py --run-dir ./logs/aishell_ngram_fusion
```

### Running the same test on Cantonese (MDCC)

The exact same experiment (N-best generation, KenLM rescoring, alpha tuning,
RER + bootstrap verdict) runs on Cantonese by pointing the pipeline at
[`ming030890/mdcc`](https://huggingface.co/datasets/ming030890/mdcc) - a
mirror of the [Multi-Domain Cantonese Corpus](https://arxiv.org/abs/2201.02419)
(MDCC) with clean `train`/`validation`/`test` splits (65120/5663/12492
utterances) that are directly comparable in scale to AISHELL-1. No dataset
code changes are needed; the differences are all CLI flags plus which
"word" segmenter is used:

- `--text-column transcript --id-column id` (MDCC's schema differs from the
  `Serenalay/AISHELL-1` mirror's `text`/`name` columns).
- **Not** `--language cantonese`, unless your checkpoint is derived from
  `whisper-large-v3`/`-turbo` specifically. Whisper only added a real
  `<|yue|>` (Cantonese) token in large-v3; every smaller size (tiny through
  large-v2) has no dedicated Cantonese token at all, so tiny/base/small/medium
  Cantonese fine-tunes (like the one recommended below) were necessarily
  trained to map Cantonese audio onto `<|zh|>` (Chinese) text, same as any
  Mandarin fine-tune - so `--language chinese` is correct for those. The
  script auto-detects this (from the checkpoint's architecture) and falls
  back with a warning if you pass `cantonese`/`yue` on a non-large-v3 model,
  but it's clearer to just pass `chinese` directly for these checkpoints.
- `--lang yue`, which swaps the "word" tokenizer from `jieba` (Mandarin) to
  [`pycantonese.segment`](https://docs.pycantonese.org/stable/word_segmentation.html)
  - a DAG+HMM segmenter trained on real Cantonese corpora (HKCanCor,
    rime-cantonese, Common Voice Cantonese). Plain jieba's dictionary is
    Mandarin-specific and mis-segments (or degenerates toward char-level on)
    Cantonese-only vocabulary and particles (`嘅`/`喺`/`佢`/`唔`/`啦` etc.), so
    reusing it for the Cantonese "word" scheme would bias that condition.
    `--lang` doesn't affect "char" tokenization, which is identical either way.
- A separate `--lm-dir` (e.g. `./lm_yue`) so the Cantonese KenLM binaries
  don't collide with the Mandarin ones under `./lm`.

**1. Build the Cantonese LM corpus.** Unlike AISHELL-1 (whose eval mirror
only ships a `test` split, so the LM corpus has to come from a separately
downloaded official transcript dump with test ids excluded by hand), MDCC's
mirror already has disjoint splits - so this just loads `train`+`validation`
directly and skips `test` (the split evaluated on below):

```
uv run prepare_mdcc_lm_corpus.py --output-dir ./lm_corpus_yue
```

**2. Train the KenLM models** (same script as Mandarin, pointed at the
Cantonese corpus/output dirs):

```
bash scripts/build_kenlm_models.sh ./lm_corpus_yue ./lm_yue
```

**3. Run the fusion eval**, with a Cantonese-finetuned Whisper checkpoint
(e.g. [`Oblivion208/whisper-small-cantonese`](https://huggingface.co/Oblivion208/whisper-small-cantonese),
a full fine-tune - not a LoRA adapter, so it loads with the same
`from_pretrained` path as `junsor/whisper-small-aishell`, no PEFT merging
needed):

```
uv run eval_aishell_ngram_fusion.py \
    --dataset-repo ming030890/mdcc --text-column transcript --id-column id \
    --lang yue --language chinese --lm-dir ./lm_yue \
    --asr-model Oblivion208/whisper-small-cantonese
```

If you see `ValueError: The generation config is outdated...` on first run,
that's unrelated to the language choice above - it means this specific
checkpoint's `generation_config.json` predates
[huggingface/transformers#25298](https://github.com/huggingface/transformers/issues/25084)
and is missing the `lang_to_id`/`task_to_id` token maps that `--language`/
`--task` rely on. The script detects and auto-repairs this in memory (by
borrowing those maps from the official same-size Whisper checkpoint) with a
`[warn]` message; if you still hit the error, your `transformers` version
may be too old to have `model.config.d_model`/`encoder_layers`/`num_mel_bins`
match one of the known architectures - upgrade `transformers` and retry.

This downloads the full `test` split (12492 utterances, audio included) on
first run; use `--max-samples 200` to iterate quickly first. Everything else
- OOM-safe checkpointed generation, `--num-beam-groups`/`--do-sample` for
beam collapse, `diagnose_nbest.py`, `--nbest-cache` - works identically to
the Mandarin walkthrough above.

**4. Analyze** exactly as before, just pointed at the Cantonese run dir
(auto-named `./logs/yue_ngram_fusion` since `--run-dir` wasn't given above):

```
uv run analyze_aishell_ngram_fusion.py --run-dir ./logs/yue_ngram_fusion
```

### English negative control (LibriSpeech)

If Chinese saturates at bigram/trigram *because* character tokens are dense,
English word n-grams (the setting WhisperLM's 5-gram convention comes from)
should keep improving through order 4/5. Same pipeline, fully open data.

**Dataset:** [`openslr/librispeech_asr`](https://huggingface.co/datasets/openslr/librispeech_asr)
`clean` config. LM corpus = `train.100` + `validation` (~31k transcripts);
eval = `test` (2620 utterances). **Metric is WER**, not CER — English has
real word boundaries. The `word` scheme (whitespace split) is the actual
negative control; `char` is included as a diagnostic (letter 5-grams only
span a few characters).

**Model:** vanilla `openai/whisper-small` (same size as the Mandarin/Cantonese
runs; not a LibriSpeech fine-tune). Pass `--do-sample` — Whisper-small is
confident on read speech and will beam-collapse without it, same as AISHELL.

**1. LM corpus** (first run downloads several GB of parquet; audio is
discarded, text is kept):

```
uv run prepare_librispeech_lm_corpus.py --output-dir ./lm_corpus_en
bash scripts/build_kenlm_models.sh ./lm_corpus_en ./lm_en
```

**2. Fusion eval** (smoke with `--max-samples 100` first, then drop it):

```
uv run eval_aishell_ngram_fusion.py \
    --lang en --language english --metric wer \
    --dataset-repo openslr/librispeech_asr --dataset-config clean \
    --dataset-split test --text-column text --id-column id \
    --asr-model openai/whisper-small --lm-dir ./lm_en \
    --do-sample --temperature 0.5 --max-new-tokens 128
```

**3. Analyze** — the verdict is inverted vs. Chinese: the negative control
is supported only if the tied-best order set excludes 2 and 3.

```
uv run analyze_aishell_ngram_fusion.py --run-dir ./logs/en_ngram_fusion
```

### Running the same test on Hakka

Hakka is a much rougher path than Cantonese - the open-source tooling is far
less mature - but the same pipeline runs on it with three structural
differences from AISHELL-1/MDCC, plus one model-availability caveat you need
to resolve up front.

**Dataset:**
[`slammax/formosan_asr_benchmark`](https://huggingface.co/datasets/slammax/formosan_asr_benchmark)'s
`hakka` config (12016 utterances, `audio_id`/`transcript`/`audio` columns).
Unlike AISHELL-1/MDCC, this ships **only one `test` split** - there's no
separate transcript corpus to train the LM on without the LM having
literally seen the sentences it'll later be asked to disambiguate. So
`prepare_hakka_lm_corpus.py` self-partitions that one split deterministically
by utterance id (a stable hash, not a random shuffle - see
`ngram_lm.is_lm_holdout`): ~70% goes into the LM corpus, and
`eval_aishell_ngram_fusion.py --lm-holdout-frac 0.3` restricts ASR evaluation
to exactly the other ~30%, so the LM is never trained on sentences it's later
scored on. Both scripts must be run with matching `--holdout-frac`/`--seed`
(defaults already agree: `0.3`/`42`).

**No word segmenter.** There is no Hakka equivalent of jieba/pycantonese - no
maintained Hakka word-segmentation library exists. `--lang hak` therefore
only exposes a `"char"` tokenizer; pass `--schemes char` explicitly (the
script warns loudly if you don't and try to use `"word"` anyway). This means
Hakka can only test the char-tokenization side of the hypothesis, not the
word-segmented side.

**Model availability - read this before spending Jetson time.** The one
*open, non-gated* Hakka Whisper checkpoint found while building this,
[`NUTN-KWS/Whisper-Taiwanese-Hakka-model-v0.2.6`](https://huggingface.co/NUTN-KWS/Whisper-Taiwanese-Hakka-model-v0.2.6),
was empirically tested against real `formosan_asr_benchmark` audio during
development and produces largely garbled/phonetically-adjacent-but-wrong
output (e.g. reference `前面向右轉就到金門縣政府了` decoded as
`透前腿轉心腱就多幾問冤真汙了`), even with beam search - almost certainly
because it was trained mostly on **synthesized TTS Hakka speech** for
textbook content, and doesn't generalize to this benchmark's real, mic-varied
recordings. A baseline that garbled would just reproduce the "beam
collapse"/"control-token leakage" failure mode from earlier in this README:
no real signal either way on the hypothesis. Every Hakka checkpoint trained
on **real** speech that could be found -
[`formospeech/whisper-large-v3-taiwanese-hakka`](https://huggingface.co/formospeech/whisper-large-v3-taiwanese-hakka)
(6 dialects, HAT-Vol2-derived; ~23% CER pre-fine-tune on its own domain per
the FSR-2025 challenge papers) and
[`formospeech/whisper-large-v2-taiwanese-hakka-v1`](https://huggingface.co/formospeech/whisper-large-v2-taiwanese-hakka-v1)
(single model, ~7-9% CER on Hakka radio news) - is **gated**: request access
on the model page, log in with `hf auth login` (or set `HF_TOKEN`) with an
account that's been granted access, then use it like any other checkpoint.
Since access approval and dialect match are both unknowns, **run
`--max-samples 20` first** and eyeball the hypotheses before committing to a
full run.

The `formospeech/*-v3-*` checkpoint additionally selects its dialect via an
*initial text prompt* rather than a language tag (its model card's own usage
example passes `prompt_ids=processor.get_prompt_ids(dialect_id)` to
`generate()`, where `dialect_id` is one of `htia_sixian`/`htia_hailu`/
`htia_dapu`/`htia_raoping`/`htia_zhaoan`/`htia_nansixian`). This is wired up
via `--dialect-prompt`; the script strips the resulting prefix back out of
the decoded text (`_strip_dialect_prompt_prefix`, mirroring the model card's
own `.replace(f" {dialect_id}", "")` post-processing). This benchmark's audio
doesn't document which dialect it is, so you may need to try a couple of
`--dialect-prompt` values on a small `--max-samples` run and keep whichever
gives the lower baseline CER. The `v2` checkpoint needs no such flag - it has
no per-dialect prompting, just plain `--language chinese --task transcribe`
like any other checkpoint here.

**1. Build the Hakka LM corpus** (writes only `char.txt`, per the no-word-
segmenter note above):

```
uv run prepare_hakka_lm_corpus.py --output-dir ./lm_corpus_hak
```

**2. Train the KenLM model:**

```
bash scripts/build_kenlm_models.sh ./lm_corpus_hak ./lm_hak
```

**3. Run the fusion eval.** With the gated v3 checkpoint (adjust
`--dialect-prompt` per the note above):

```
uv run eval_aishell_ngram_fusion.py --max-samples 20 \
    --dataset-repo slammax/formosan_asr_benchmark --dataset-config hakka \
    --text-column transcript --id-column audio_id --lm-holdout-frac 0.3 \
    --lang hak --schemes char --language chinese --lm-dir ./lm_hak \
    --asr-model formospeech/whisper-large-v3-taiwanese-hakka \
    --dialect-prompt htia_sixian
```

Or with the gated v2 checkpoint (no `--dialect-prompt` needed):

```
uv run eval_aishell_ngram_fusion.py --max-samples 20 \
    --dataset-repo slammax/formosan_asr_benchmark --dataset-config hakka \
    --text-column transcript --id-column audio_id --lm-holdout-frac 0.3 \
    --lang hak --schemes char --language chinese --lm-dir ./lm_hak \
    --asr-model formospeech/whisper-large-v2-taiwanese-hakka-v1
```

Once the baseline CER on that smoke run looks sane (comparable to the
~5-10% range seen on AISHELL-1/MDCC, not near 50%+), drop `--max-samples 20`
and let it run on the full ~3500-utterance eval partition.

**4. Analyze** exactly as before (auto-named `./logs/hak_ngram_fusion`):

```
uv run analyze_aishell_ngram_fusion.py --run-dir ./logs/hak_ngram_fusion
```

### Methodology notes

- **Metric.** For Chinese-family languages, alpha is always tuned to minimize
  CER (segmentation-tool independent) for *both* schemes; WER is not computed
  there because Chinese has no native word boundaries. For English
  (`--lang en`) the metric is WER — that is the WhisperLM-analogue quantity
  the negative control is about.
- **RER.** `(baseline_error - condition_error) / baseline_error`, computed on
  the eval slice; `alpha=0` reproduces the no-LM baseline for every order as a
  built-in sanity check.
- **Significance.** `analyze_aishell_ngram_fusion.py` runs a paired bootstrap
  (utterance-level resampling with replacement, recomputing corpus-level CER
  per resample) to distinguish "order 2 is nominally best" from "order 2 is
  significantly best" — orders whose CER-gap CI against the best order
  includes 0 are reported as statistically tied.
