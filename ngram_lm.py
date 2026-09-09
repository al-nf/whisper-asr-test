"""Shared helpers for the Mandarin (AISHELL-1), Cantonese (MDCC), Hakka, and
English (LibriSpeech) n-gram fusion experiments.

Two tokenization schemes are supported:

- "char": every remaining character is its own token (no segmentation).
- "word": word segmentation. Mandarin uses `jieba`; Cantonese uses
  `pycantonese.segment` (jieba's dictionary is Mandarin-specific); English
  uses whitespace split after `normalize_en_text`. Hakka ("hak") has no
  "word" entry: no maintained Hakka segmenter exists, so pass `--schemes char`.

`KenLMScorer` wraps a compiled KenLM binary (`.klm`, built by
`scripts/build_kenlm_models.sh` from `lmplz`/`build_binary`) and exposes a
single `avg_logprob` method that both the `prepare_*_lm_corpus.py` scripts
(via the tokenizers) and `eval_aishell_ngram_fusion.py` (via the scorer) rely
on, so corpus prep and rescoring can never drift out of sync on tokenization.

KenLM reports probabilities in log base 10 (see the KenLM README); everywhere
in this codebase we convert to natural log immediately on read so downstream
code (combining with Whisper's natural-log sequence scores) never has to
think about the base.
"""

import math
from typing import List

import jieba

# Silence jieba's "Building prefix dict" / "Loading model cost ..." chatter,
# which would otherwise interleave with tqdm progress bars.
jieba.setLogLevel(60)

LOG10_TO_LN = math.log(10)


def tokenize_char(text: str) -> List[str]:
    """Character tokenization. `text` is expected to already be whitespace-free
    (see `normalize_zh_text`); every remaining character becomes one token."""
    return list(text)


def tokenize_word_jieba(text: str) -> List[str]:
    """Jieba word segmentation (Mandarin). `cut_all=False` (default, "accurate
    mode") is used everywhere so corpus building and hypothesis scoring
    segment identically."""
    return [tok for tok in jieba.lcut(text, cut_all=False) if tok.strip()]


def tokenize_word_cantonese(text: str) -> List[str]:
    """`pycantonese.segment` word segmentation (Cantonese) - a jieba-styled
    DAG+HMM segmenter trained on Cantonese corpora (HKCanCor, rime-cantonese,
    Common Voice Cantonese, etc.) rather than jieba's Mandarin dictionary."""
    import pycantonese

    return [tok for tok in pycantonese.segment(text) if tok.strip()]


def tokenize_word_english(text: str) -> List[str]:
    """Whitespace word tokenization (English). `text` is expected to already
    be lowercased and punctuation-stripped (see `normalize_en_text`)."""
    return text.split()


def tokenize_char_nospace(text: str) -> List[str]:
    """Character tokenization that drops whitespace - the English analogue of
    `tokenize_char` on Chinese (where `normalize_zh_text` has already removed
    spaces, so `list(text)` is letters-only)."""
    return [c for c in text if not c.isspace()]


# Registry of {scheme: tokenizer} per language. `char` is shared; only the
# `word` segmenter differs. Both eval_aishell_ngram_fusion.py and the
# prepare_*_lm_corpus.py scripts select one of these via a `--lang` flag, so
# corpus-building and eval-time candidate tokenization always agree. "hak"
# deliberately has no "word" key - see module docstring.
TOKENIZERS_BY_LANG = {
    "zh": {"char": tokenize_char, "word": tokenize_word_jieba},
    "yue": {"char": tokenize_char, "word": tokenize_word_cantonese},
    "hak": {"char": tokenize_char},
    "en": {"char": tokenize_char_nospace, "word": tokenize_word_english},
}


def get_tokenizers(lang: str) -> "dict[str, callable]":
    try:
        return TOKENIZERS_BY_LANG[lang]
    except KeyError:
        raise ValueError(
            f"Unknown --lang '{lang}'; supported: {list(TOKENIZERS_BY_LANG)}"
        ) from None


# Backwards-compatible default (Mandarin/jieba) - most call sites should
# switch to `get_tokenizers(args.lang)` instead.
TOKENIZERS = TOKENIZERS_BY_LANG["zh"]


def normalize_zh_text(text: str) -> str:
    """Strip whitespace and non-word punctuation, mirroring the zh branch of
    `normalize_text` in eval.py / eval_llm_postprocess.py. Kept as a separate
    function (rather than importing from those scripts) so this module has no
    torch/whisper import dependency and stays cheap to import from the
    corpus-prep script."""
    import re

    return re.sub(r"[^\w]", "", text, flags=re.UNICODE)


def normalize_en_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Mirrors the non-zh
    branch of `normalize_text` in eval.py. Spaces are kept: English WER and
    word-n-gram tokenization both need them."""
    import re

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_lang(text: str, lang: str) -> str:
    if lang == "en":
        return normalize_en_text(text)
    return normalize_zh_text(text)


def is_lm_holdout(utt_id: str, holdout_frac: float, seed: int) -> bool:
    """Deterministic membership test used to self-partition a *single-split*
    dataset (e.g. formosan_asr_benchmark's Hakka subset, which - unlike
    AISHELL-1/MDCC - ships only one `test` split, with no separate
    train/transcript source to build the LM corpus from without touching the
    eval sentences).

    `prepare_hakka_lm_corpus.py` excludes every utterance where this returns
    True from the LM training text; `eval_aishell_ngram_fusion.py --lm-holdout-frac`
    evaluates ASR n-gram fusion on exactly (and only) those same utterances -
    so the two scripts must be called with the same `holdout_frac`/`seed` (the
    default of each matches the other). Uses a stable hash (md5, not Python's
    salted built-in `hash()`) so membership is reproducible across processes
    and across the two scripts/machines."""
    import hashlib

    digest = hashlib.md5(f"{seed}:{utt_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF  # deterministic float in [0, 1)
    return bucket < holdout_frac


class KenLMScorer:
    """Thin wrapper around a compiled KenLM model for whole-sequence scoring
    of a list of tokens (chars or jieba words)."""

    def __init__(self, path: str):
        import kenlm

        self.model = kenlm.Model(path)
        self.order = self.model.order

    def avg_logprob(self, tokens: List[str]) -> float:
        """Natural-log probability of `tokens`, normalized by token count.

        `bos=True, eos=True` scores the sequence as a complete utterance
        (matching how the LM corpus was built: one utterance per line), so
        short/long candidates are penalized/rewarded consistently with how
        the LM was trained.
        """
        if not tokens:
            return 0.0
        text = " ".join(tokens)
        log10_prob = self.model.score(text, bos=True, eos=True)
        return (log10_prob * LOG10_TO_LN) / len(tokens)
