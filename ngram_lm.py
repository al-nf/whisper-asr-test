"""Shared helpers for the AISHELL-1 n-gram fusion experiment.

Two tokenization schemes are supported everywhere in this experiment:

- "char": every Han character is its own token (no segmentation needed).
- "word": `jieba.lcut` word segmentation.

`KenLMScorer` wraps a compiled KenLM binary (`.klm`, built by
`scripts/build_kenlm_models.sh` from `lmplz`/`build_binary`) and exposes a
single `avg_logprob` method that both `prepare_aishell_lm_corpus.py` (via the
tokenizers) and `eval_aishell_ngram_fusion.py` (via the scorer) rely on, so
corpus prep and rescoring can never drift out of sync on tokenization.

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


def tokenize_word(text: str) -> List[str]:
    """Jieba word segmentation. `cut_all=False` (default, "accurate mode") is
    used everywhere so corpus building and hypothesis scoring segment
    identically."""
    return [tok for tok in jieba.lcut(text, cut_all=False) if tok.strip()]


TOKENIZERS = {
    "char": tokenize_char,
    "word": tokenize_word,
}


def normalize_zh_text(text: str) -> str:
    """Strip whitespace and non-word punctuation, mirroring the zh branch of
    `normalize_text` in eval.py / eval_llm_postprocess.py. Kept as a separate
    function (rather than importing from those scripts) so this module has no
    torch/whisper import dependency and stays cheap to import from the
    corpus-prep script."""
    import re

    return re.sub(r"[^\w]", "", text, flags=re.UNICODE)


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
