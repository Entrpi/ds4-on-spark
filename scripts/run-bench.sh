#!/usr/bin/env bash
# scripts/run-bench.sh — run llama-benchy against a running ds4-server.
#
# Reproduces the methodology used in this repo's README to produce
# directly-comparable numbers to other OpenAI-compatible servers
# (llama.cpp, vLLM, SGLang).
#
# Requires: `uv` (https://docs.astral.sh/uv/), and a running ds4-server.

set -euo pipefail

PORT="${PORT:-8000}"
MODEL="${MODEL:-deepseek-v4-flash}"
# Path or HF id of tokenizer. Defaults to the converted safetensors checkpoint
# if you have it; otherwise points at the HF repo so llama-benchy fetches it.
TOKENIZER="${TOKENIZER:-${TOKENIZER_DIR:-/home/ent/models/deepseek-v4-flash-ds4-q2}}"

PP=(2048)
TG=(32 128 512)
DEPTH=(0 4096 16384)
CONCURRENCY=(1)
RUNS=3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --tokenizer) TOKENIZER="$2"; shift 2 ;;
        --pp) shift; PP=(); while [[ $# -gt 0 && "$1" != --* ]]; do PP+=("$1"); shift; done ;;
        --tg) shift; TG=(); while [[ $# -gt 0 && "$1" != --* ]]; do TG+=("$1"); shift; done ;;
        --depth) shift; DEPTH=(); while [[ $# -gt 0 && "$1" != --* ]]; do DEPTH+=("$1"); shift; done ;;
        --concurrency) shift; CONCURRENCY=(); while [[ $# -gt 0 && "$1" != --* ]]; do CONCURRENCY+=("$1"); shift; done ;;
        --runs) RUNS="$2"; shift 2 ;;
        --help|-h) sed -n '2,/^$/p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

command -v uvx >/dev/null 2>&1 || {
    echo "uvx not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
}

curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || {
    echo "ds4-server not running on :$PORT. Start it with scripts/start-server.sh." >&2
    exit 1
}

exec uvx --from git+https://github.com/eugr/llama-benchy llama-benchy \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --pp "${PP[@]}" \
    --tg "${TG[@]}" \
    --depth "${DEPTH[@]}" \
    --concurrency "${CONCURRENCY[@]}" \
    --runs "$RUNS" \
    --latency-mode generation
