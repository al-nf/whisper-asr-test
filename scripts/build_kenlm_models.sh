#!/usr/bin/env bash
# Trains the 8 KenLM binaries (orders 2-5 x {char,word}) used by
# eval_aishell_ngram_fusion.py, from the corpora produced by
# prepare_aishell_lm_corpus.py.
#
# Order 1 (unigram) is deliberately excluded: KenLM's query/loading code
# (lm/model.cc) hard-requires at least a bigram model - `lmplz -o 1` can
# produce a unigram ARPA file, but `build_binary`/`kenlm.Model` refuse to
# load it ("This ngram implementation assumes at least a bigram model").
# The no-LM beam-search baseline in eval_aishell_ngram_fusion.py already
# serves as the effective "0th order" comparison point.
#
# Usage:
#   export KENLM_BIN_DIR=/path/to/kenlm/build/bin   # from setup_kenlm_jetson.sh
#   bash scripts/build_kenlm_models.sh [corpus_dir] [output_dir] [max_order] [min_order]
#
# Defaults: corpus_dir=./lm_corpus  output_dir=./lm  max_order=5  min_order=2
set -euo pipefail

CORPUS_DIR="${1:-./lm_corpus}"
OUTPUT_DIR="${2:-./lm}"
MAX_ORDER="${3:-5}"
MIN_ORDER="${4:-2}"

: "${KENLM_BIN_DIR:?Set KENLM_BIN_DIR to the directory containing lmplz/build_binary (see scripts/setup_kenlm_jetson.sh)}"
LMPLZ="${KENLM_BIN_DIR}/lmplz"
BUILD_BINARY="${KENLM_BIN_DIR}/build_binary"

for bin in "${LMPLZ}" "${BUILD_BINARY}"; do
    if [ ! -x "${bin}" ]; then
        echo "error: ${bin} not found or not executable" >&2
        exit 1
    fi
done

for scheme in char word; do
    corpus_file="${CORPUS_DIR}/${scheme}.txt"
    if [ ! -f "${corpus_file}" ]; then
        echo "error: ${corpus_file} not found - run prepare_aishell_lm_corpus.py first" >&2
        exit 1
    fi

    scheme_out_dir="${OUTPUT_DIR}/${scheme}"
    mkdir -p "${scheme_out_dir}"

    for order in $(seq "${MIN_ORDER}" "${MAX_ORDER}"); do
        arpa_path="${scheme_out_dir}/order${order}.arpa"
        klm_path="${scheme_out_dir}/order${order}.klm"

        echo "== Training ${scheme} order-${order} LM =="
        "${LMPLZ}" -o "${order}" --discount_fallback < "${corpus_file}" > "${arpa_path}"

        echo "== Binarizing ${scheme} order-${order} LM =="
        "${BUILD_BINARY}" "${arpa_path}" "${klm_path}"

        rm -f "${arpa_path}"
        echo "-> ${klm_path}"
    done
done

echo
echo "Done. Models written under ${OUTPUT_DIR}/{char,word}/order{${MIN_ORDER}..${MAX_ORDER}}.klm"
echo "Next: uv run eval_aishell_ngram_fusion.py --lm-dir ${OUTPUT_DIR} --orders $(seq -s' ' "${MIN_ORDER}" "${MAX_ORDER}")"
