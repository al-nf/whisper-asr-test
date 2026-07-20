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

Results (raw/llm WER & CER, per-sample transcripts) are saved under
`./logs/llm_postprocess/`.
