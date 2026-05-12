#!/usr/bin/env bash
# scripts/smoke-test.sh — verify ds4 produces the expected first token on the
# canonical "capital of France" prompt. Fast end-to-end correctness check.
#
# Usage:
#   scripts/smoke-test.sh                         # uses default paths
#   scripts/smoke-test.sh --port 8000             # against a running ds4-server
#   scripts/smoke-test.sh --gguf /path/to/file    # against a specific GGUF

set -euo pipefail

DS4_SRC_DIR="${DS4_SRC_DIR:-$HOME/code/ds4}"
DS4_GGUF_DIR="${DS4_GGUF_DIR:-$HOME/gguf}"
GGUF_FILE="${GGUF_FILE:-DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}"
GGUF_PATH="${GGUF_PATH:-$DS4_GGUF_DIR/$GGUF_FILE}"
PORT=""
PROMPT='What is the capital of France? Answer in one sentence.'
EXPECT_RE='[Pp]aris'

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --gguf) GGUF_PATH="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --expect) EXPECT_RE="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,/^$/p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

if [[ -n "$PORT" ]]; then
    # HTTP path — talk to ds4-server.
    body=$(jq -n --arg p "$PROMPT" '{
        model: "deepseek-v4-flash",
        messages: [{role: "user", content: $p}],
        max_tokens: 64,
        stream: false
    }')
    out=$(curl -sS -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' --data "$body")
    text=$(echo "$out" | jq -r '.choices[0].message.content // ""')
else
    # CLI path — direct ds4 binary against GGUF.
    [[ -x "$DS4_SRC_DIR/ds4" ]] || { echo "ds4 binary not at $DS4_SRC_DIR/ds4" >&2; exit 1; }
    [[ -f "$GGUF_PATH" ]] || { echo "GGUF not at $GGUF_PATH" >&2; exit 1; }
    text=$( "$DS4_SRC_DIR/ds4" --cuda -m "$GGUF_PATH" -c 4096 -p "$PROMPT" 2>&1 | tail -20 )
fi

echo "$text"
echo "---"
if echo "$text" | grep -qE "$EXPECT_RE"; then
    echo "PASS — output matches /$EXPECT_RE/"
else
    echo "FAIL — output does not match /$EXPECT_RE/" >&2
    exit 1
fi
