#!/usr/bin/env bash
# scripts/start-server.sh — start ds4-server with sensible defaults for Spark.
#
# Backgrounds the process and disowns; logs to $HOME/ds4-server.log.
# Idempotent: refuses to start if a ds4-server is already running on the port.

set -euo pipefail

DS4_SRC_DIR="${DS4_SRC_DIR:-$HOME/code/ds4}"
DS4_GGUF_DIR="${DS4_GGUF_DIR:-$HOME/gguf}"
# Default to the 0731 checkpoint, falling back to the previous-generation
# file when only that one is installed (generation is detected from the
# file name, matching install.sh — a GGUF_FILE override wins either way).
LEGACY_GGUF_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
GGUF_FILE="${GGUF_FILE:-DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf}"
if [[ ! -f "$DS4_GGUF_DIR/$GGUF_FILE" && -f "$DS4_GGUF_DIR/$LEGACY_GGUF_FILE" ]]; then
    GGUF_FILE="$LEGACY_GGUF_FILE"
fi
GGUF_PATH="${GGUF_PATH:-$DS4_GGUF_DIR/$GGUF_FILE}"
MTP_FILE="${MTP_FILE:-DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf}"
MTP_PATH="${MTP_PATH:-$DS4_GGUF_DIR/$MTP_FILE}"
PORT="${PORT:-8000}"
CTX="${CTX:-32768}"
LOG="${LOG:-$HOME/ds4-server.log}"
USE_MTP=0
DRAFT=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --ctx)  CTX="$2"; shift 2 ;;
        --log)  LOG="$2"; shift 2 ;;
        --with-mtp) USE_MTP=1; shift ;;
        --draft) DRAFT="$2"; shift 2 ;;
        --help|-h) sed -n '2,/^$/p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "ds4-server already running on :$PORT" >&2
    curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -m json.tool 2>/dev/null || true
    exit 0
fi

[[ -x "$DS4_SRC_DIR/ds4-server" ]] || { echo "ds4-server not at $DS4_SRC_DIR" >&2; exit 1; }
[[ -f "$GGUF_PATH" ]] || { echo "GGUF not at $GGUF_PATH" >&2; exit 1; }

mtp_args=()
if [[ "$USE_MTP" -eq 1 ]]; then
    if [[ "$GGUF_FILE" == *-0731* ]]; then
        echo "--with-mtp: the 0731 checkpoint has no MTP head (replaced by the" >&2
        echo "DSpark stages); drop --with-mtp, or set GGUF_FILE to the pre-0731 base." >&2
        exit 2
    fi
    [[ -f "$MTP_PATH" ]] || { echo "MTP GGUF not at $MTP_PATH" >&2; exit 1; }
    mtp_args=(--mtp "$MTP_PATH" --mtp-draft "$DRAFT")
    echo "NOTE: MTP enabled. Our benchmarks show ~3-6% regression on Spark"
    echo "      — included for completeness; omit --with-mtp for fastest decode."
fi

nohup "$DS4_SRC_DIR/ds4-server" --cuda -m "$GGUF_PATH" \
    "${mtp_args[@]}" --port "$PORT" -c "$CTX" \
    > "$LOG" 2>&1 < /dev/null & disown
pid=$!
echo "ds4-server pid=$pid, log=$LOG, port=$PORT, ctx=$CTX"

for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "Server ready."
        curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -m json.tool 2>/dev/null || true
        exit 0
    fi
    sleep 2
done

echo "Server failed to come up within 120 s." >&2
tail -20 "$LOG" >&2
exit 1
