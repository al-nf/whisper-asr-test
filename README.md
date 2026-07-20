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
