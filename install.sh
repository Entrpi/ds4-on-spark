#!/usr/bin/env bash
# install.sh — antirez/ds4 (DwarfStar 4) on NVIDIA DGX Spark (GB10 / SM121)
#
#   curl -sSL https://raw.githubusercontent.com/<owner>/ds4-on-spark/main/install.sh | bash
#   curl -sSL https://raw.githubusercontent.com/<owner>/ds4-on-spark/main/install.sh | bash -s -- --help
#
# What this script does (all steps idempotent — safe to re-run):
#
#   1. Verifies the host is a DGX Spark (or other GB10/SM121 system) with
#      CUDA 13 toolkit installed and >= 120 GiB free disk for the GGUFs.
#   2. Clones (or fast-forwards) the ds4 fork at tag v0.2.2 into $DS4_SRC_DIR.
#   3. Builds ds4, ds4-server, ds4-bench with CUDA_ARCH=sm_121.
#   4. Downloads the Q2 quantized GGUF (~81 GiB) + MTP GGUF (~3.6 GiB) from
#      antirez/deepseek-v4-gguf and the DSpark Q2K drafter (~6.5 GiB) from
#      bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF into $DS4_GGUF_DIR.
#   5. Runs a single-prompt smoke test against the canonical
#      "capital of France" prompt — expects "Paris" in the output.
#   6. Optionally (--start) serves the full DSpark speculative stack on
#      $DS4_PORT (lossless; yield-quench + kv-gate ride the v0.2.2 defaults).
#      --no-dspark serves plain continuous decode instead.
#
# The script makes NO changes outside:
#   - $DS4_SRC_DIR      (default ~/code/ds4)
#   - $DS4_GGUF_DIR     (default ~/gguf)
#   - the running ds4-server process (only if --start)
#
# License: MIT.  Source: https://github.com/entrpi/ds4-on-spark

set -euo pipefail

# ============================================================================
# 0. defaults + flag parsing
# ============================================================================

# Pin (updated 2026-07-24): the Entrpi/ds4 fork release tag v0.4.2.
# Everything in v0.4.0 (deep-decode substrate: head-group
# flash-decode for dense and indexed attention, aligned Q8_0 dense
# tier at all verify widths, indexer scorer token-loop, MoE gate_up
# expert dedup, speculation armed at every depth, the contributed
# queued-client zombie reap, the ds4-bench 32K+ fix) plus the v0.4.1
# quench recalibration: the speculation break-even guard now tracks
# v0.4's measured verify cost (2.22 -> 2.10; the v0.1.1-era guard was
# terminally quenching winners). Code-corpus decode 1.10x own plain,
# adversarial prose 1.04x (typical quench floor 0.95-0.97x, bounded
# learning debt; post-quench serving identical to plain). Deep
# stamps hold: 240K 57.3 ms/tok, 515K 59.9, 12K 36.6. v0.4.2 adds the
# community-contributed thinking-mode warm reuse on the continuous
# path (thinking turns reuse KV instead of cold re-prefilling, and
# thinking banks now persist to the disk KV tier). Standing
# release gates green on the tagged binary. The engine
# builds its aligned fast-path artifacts in-process at boot (since
# v0.2.2), so this installer's standalone server rides the full perf
# tier (stated in the boot log). Set
# DS4_REF=main + DS4_REPO=antirez/ds4 for the upstream engine without
# the fork's serving stack.
DS4_REPO="${DS4_REPO:-https://github.com/Entrpi/ds4.git}"
DS4_REF="${DS4_REF:-v0.4.2}"
DS4_SRC_DIR="${DS4_SRC_DIR:-$HOME/code/ds4}"
DS4_GGUF_DIR="${DS4_GGUF_DIR:-$HOME/gguf}"

CUDA_ARCH="${CUDA_ARCH:-sm_121}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc 2>/dev/null || echo 4)}"

HF_REPO="${HF_REPO:-antirez/deepseek-v4-gguf}"
# The imatrix-tuned q2 — the build all fork quality baselines were stamped on
# (the repo also hosts a plain chat-v2.gguf; the old default named it while
# claiming imatrix).
GGUF_FILE="${GGUF_FILE:-DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf}"
MTP_FILE="${MTP_FILE:-DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf}"

DSPARK_HF_REPO="${DSPARK_HF_REPO:-bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF}"
DSPARK_FILE="${DSPARK_FILE:-DSpark-drafter-Q2K-Q8.gguf}"

DS4_PORT="${DS4_PORT:-8000}"
DS4_CTX="${DS4_CTX:-32768}"

FORCE_HW=0
SKIP_BUILD=0
SKIP_DOWNLOAD=0
SKIP_MTP=0
SKIP_SMOKE=0
START_SERVER=0
# DSpark speculative decode is the default serving mode since v0.1.1 (suite
# mean 1.38x plain; net-positive per request via the yield-quench controller).
# --no-dspark serves plain continuous decode instead.
WITH_MTP=1
WITH_DSPARK=1

usage() {
    cat <<EOF
Usage: $0 [flags]

Flags:
  --help                  Show this help.
  --force                 Skip GB10/SM121 host check.
  --no-build              Skip clone + build (use existing $DS4_SRC_DIR/ds4*).
  --no-download           Skip GGUF download.
  --with-mtp              (default) Download the MTP speculative-decode GGUF.
  --with-dspark           (default) Serve with DSpark speculative decode —
                          downloads the ~6.5 GiB Q2K drafter from
                          $DSPARK_HF_REPO.
  --no-dspark             Serve plain continuous decode (no drafter download).
  --no-smoke              Skip post-install smoke test.
  --start                 Start ds4-server on :$DS4_PORT after install.
  --src-dir DIR           Where to put antirez/ds4 source (default: $DS4_SRC_DIR).
  --gguf-dir DIR          Where to put GGUF weights (default: $DS4_GGUF_DIR).
  --cuda-arch ARCH        nvcc -arch flag (default: $CUDA_ARCH).
  --jobs N                make -j N (default: $BUILD_JOBS).
  --port N                ds4-server port if --start (default: $DS4_PORT).
  --ctx N                 Allocated context size at server start (default: $DS4_CTX).

Environment variable equivalents:
  DS4_REPO DS4_REF DS4_SRC_DIR DS4_GGUF_DIR CUDA_ARCH BUILD_JOBS
  HF_REPO GGUF_FILE MTP_FILE DS4_PORT DS4_CTX
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --force) FORCE_HW=1; shift ;;
        --no-build) SKIP_BUILD=1; shift ;;
        --no-download) SKIP_DOWNLOAD=1; shift ;;
        --with-mtp) WITH_MTP=1; shift ;;
        --with-dspark) WITH_DSPARK=1; WITH_MTP=1; shift ;;
        --no-dspark) WITH_DSPARK=0; shift ;;
        --no-mtp) SKIP_MTP=1; shift ;;
        --no-smoke) SKIP_SMOKE=1; shift ;;
        --start) START_SERVER=1; shift ;;
        --src-dir) DS4_SRC_DIR="$2"; shift 2 ;;
        --gguf-dir) DS4_GGUF_DIR="$2"; shift 2 ;;
        --cuda-arch) CUDA_ARCH="$2"; shift 2 ;;
        --jobs) BUILD_JOBS="$2"; shift 2 ;;
        --port) DS4_PORT="$2"; shift 2 ;;
        --ctx) DS4_CTX="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; usage; exit 2 ;;
    esac
done

GGUF_PATH="$DS4_GGUF_DIR/$GGUF_FILE"
MTP_PATH="$DS4_GGUF_DIR/$MTP_FILE"
DSPARK_PATH="$DS4_GGUF_DIR/$DSPARK_FILE"

c_red()   { printf '\033[31m%s\033[0m' "$*"; }
c_green() { printf '\033[32m%s\033[0m' "$*"; }
c_yellow(){ printf '\033[33m%s\033[0m' "$*"; }
log() { printf '%s %s\n' "[$(date +%H:%M:%S)]" "$*"; }
die() { printf '\n%s %s\n' "$(c_red FATAL:)" "$*" >&2; exit 1; }
warn(){ printf '%s %s\n' "$(c_yellow WARN:)" "$*" >&2; }
ok()  { printf '%s %s\n' "$(c_green OK:)" "$*"; }

# ============================================================================
# 1. host verification
# ============================================================================

verify_host() {
    log "Verifying host..."

    local uname_m; uname_m=$(uname -m)
    if [[ "$uname_m" != "aarch64" ]] && [[ "$FORCE_HW" -eq 0 ]]; then
        die "Expected aarch64 (Grace+Blackwell); got $uname_m. Pass --force to skip."
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        die "nvidia-smi not found. Need NVIDIA driver installed."
    fi

    local gpu_info; gpu_info=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || true)
    if [[ -z "$gpu_info" ]]; then
        die "nvidia-smi failed to enumerate GPUs."
    fi
    log "GPU: $gpu_info"

    if ! echo "$gpu_info" | grep -qE 'compute_cap.*12\.1|GB10|Spark'; then
        if [[ "$FORCE_HW" -eq 0 ]]; then
            warn "Not detecting GB10 / SM12.1. ds4 may still work on other Blackwell SKUs."
            warn "Pass --cuda-arch sm_120 (or matching) and --force to proceed."
            die "Host check failed. Pass --force to skip."
        fi
    fi

    # CUDA toolkit
    local nvcc_bin="/usr/local/cuda/bin/nvcc"
    if [[ ! -x "$nvcc_bin" ]]; then
        nvcc_bin=$(command -v nvcc 2>/dev/null || true)
    fi
    if [[ -z "$nvcc_bin" ]] || [[ ! -x "$nvcc_bin" ]]; then
        die "nvcc not found. Install cuda-toolkit (we tested 13.0)."
    fi
    log "nvcc: $nvcc_bin"
    "$nvcc_bin" --version | head -4 | tail -1

    # Disk
    local free_gib
    free_gib=$(df -BG "$HOME" | awk 'NR==2 {gsub("G","",$4); print $4}')
    if (( free_gib < 120 )); then
        if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
            die "Need >= 120 GiB free under $HOME; have ${free_gib} GiB. Pass --no-download to skip GGUF, or free space."
        fi
        warn "Only ${free_gib} GiB free under $HOME; --no-download is set, continuing."
    fi
    ok "Host checks passed."
}

# ============================================================================
# 2. clone + build ds4
# ============================================================================

clone_and_build() {
    if [[ "$SKIP_BUILD" -eq 1 ]]; then
        log "Skipping clone + build (--no-build)."
        return
    fi
    log "Source dir: $DS4_SRC_DIR"
    if [[ ! -d "$DS4_SRC_DIR/.git" ]]; then
        mkdir -p "$(dirname "$DS4_SRC_DIR")"
        log "Cloning $DS4_REPO ..."
        git clone --depth 1 -b "$DS4_REF" "$DS4_REPO" "$DS4_SRC_DIR"
    else
        log "Fast-forwarding $DS4_SRC_DIR ..."
        (
            cd "$DS4_SRC_DIR"
            current_url=$(git remote get-url origin 2>/dev/null || echo "")
            if [[ "$current_url" != "$DS4_REPO" ]]; then
                log "Repointing origin: ${current_url:-<unset>} -> $DS4_REPO"
                git remote set-url origin "$DS4_REPO"
            fi
            git fetch --depth 1 origin "$DS4_REF"
            git reset --hard FETCH_HEAD
        )
    fi

    log "Building ds4, ds4-server, ds4-bench (CUDA_ARCH=$CUDA_ARCH, -j$BUILD_JOBS) ..."
    # As of upstream commit be43477 ("Standardize context length errors", 2026-05-15)
    # the default `make` target prints help instead of building. The named targets
    # are `make cuda CUDA_ARCH=...`, `make cuda-spark`, `make cuda-generic`,
    # `make cpu`. `make cuda-spark` now builds native sm_121 (fixed in ds4
    # commit dd157bd — it previously left `-arch` empty, ~25% slower prefill on
    # GB10). We still call `make cuda CUDA_ARCH=$CUDA_ARCH` here to preserve the
    # user-facing `--cuda-arch sm_NNN` flag for non-GB10 Blackwell SKUs.
    ( cd "$DS4_SRC_DIR" && make cuda -j"$BUILD_JOBS" CUDA_ARCH="$CUDA_ARCH" )

    for bin in ds4 ds4-server ds4-bench; do
        [[ -x "$DS4_SRC_DIR/$bin" ]] || die "Build did not produce $bin"
    done
    ok "Built: $DS4_SRC_DIR/{ds4,ds4-server,ds4-bench}"
}

# ============================================================================
# 3. download GGUFs (curl, resumable)
# ============================================================================

download_one() {
    local file="$1" dest="$2" repo="${3:-$HF_REPO}"
    local url="https://huggingface.co/$repo/resolve/main/$file"
    if [[ -f "$dest" ]]; then
        # Cheap completeness check: redownload only if HEAD content-length mismatches.
        local remote_size
        remote_size=$(curl -sI -L "$url" | awk -F': ' 'tolower($1)=="content-length"{print $2+0}' | tail -1)
        local local_size
        local_size=$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest")
        if [[ -n "$remote_size" ]] && [[ "$remote_size" == "$local_size" ]]; then
            ok "Already have $file ($local_size bytes)"
            return
        fi
        warn "Existing $file is $local_size B, expected $remote_size B — resuming."
    fi
    mkdir -p "$DS4_GGUF_DIR"
    log "Downloading $file from $HF_REPO ..."
    curl -L --fail --progress-bar -C - -o "$dest" "$url"
    ok "Downloaded $file"
}

download_models() {
    if [[ "$SKIP_DOWNLOAD" -eq 1 ]]; then
        log "Skipping GGUF download (--no-download)."
        return
    fi
    download_one "$GGUF_FILE" "$GGUF_PATH"
    if [[ "$WITH_MTP" -eq 1 ]] && [[ "$SKIP_MTP" -eq 0 ]]; then
        download_one "$MTP_FILE" "$MTP_PATH"
    fi
    if [[ "$WITH_DSPARK" -eq 1 ]]; then
        download_one "$DSPARK_FILE" "$DSPARK_PATH" "$DSPARK_HF_REPO"
    fi
}

# ============================================================================
# 4. smoke test
# ============================================================================

smoke_test() {
    if [[ "$SKIP_SMOKE" -eq 1 ]]; then
        log "Skipping smoke test (--no-smoke)."
        return
    fi
    [[ -f "$GGUF_PATH" ]] || { warn "$GGUF_PATH missing — skipping smoke test."; return; }

    log "Smoke test: 'capital of France' prompt ..."
    local out
    out=$( "$DS4_SRC_DIR/ds4" --cuda -m "$GGUF_PATH" -c 4096 \
           -p "What is the capital of France? Answer in one sentence." 2>&1 | tail -20 )
    echo "$out"
    if echo "$out" | grep -qi 'paris'; then
        ok "Smoke test PASSED — model produced 'Paris'."
    else
        die "Smoke test FAILED — 'Paris' not in output. See full output above."
    fi
}

# ============================================================================
# 5. install the ds4-serve launcher
# ============================================================================

install_launcher() {
    local src dst="$HOME/.local/bin/ds4-serve"
    src="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bin/ds4-serve"
    if [[ ! -f "$src" ]]; then
        # curl|bash install: fetch the launcher from the repo at the same ref.
        mkdir -p "$HOME/.local/bin"
        curl -fsSL "https://raw.githubusercontent.com/Entrpi/ds4-on-spark/main/bin/ds4-serve" -o "$dst" \
            || { warn "could not install ds4-serve launcher (offline?); full command is in the README."; return; }
    else
        mkdir -p "$HOME/.local/bin"
        cp "$src" "$dst"
    fi
    chmod +x "$dst"
    ok "Installed ds4-serve to $dst (full stack by default; just pass -c/--host/--port)."
    case ":$PATH:" in
        *":$HOME/.local/bin:"*) ;;
        *) warn "$HOME/.local/bin is not on PATH — add it to use 'ds4-serve' directly." ;;
    esac
}

# ============================================================================
# 6. optional: start server
# ============================================================================

start_server() {
    [[ "$START_SERVER" -eq 1 ]] || return
    [[ -f "$GGUF_PATH" ]] || die "$GGUF_PATH missing — cannot start server."

    local flags=()
    if [[ "$WITH_DSPARK" -eq 1 ]]; then
        [[ -f "$MTP_PATH" ]] || die "$MTP_PATH missing — --with-dspark needs the MTP GGUF."
        [[ -f "$DSPARK_PATH" ]] || die "$DSPARK_PATH missing — build it with gguf-tools/dspark_extract.py (README: DSpark)."
        log "Starting ds4-server with DSpark speculative decode (yield-quench + kv-gate ride the v0.2.2 defaults)."
    elif [[ "$WITH_MTP" -eq 1 ]] && [[ -f "$MTP_PATH" ]]; then
        flags+=(--no-dspark)
        log "Starting ds4-server with MTP-2 speculation (continuous batching)."
    else
        flags+=(--no-spec)
        log "Starting ds4-server (plain continuous decode; add --with-mtp or --with-dspark for speculation)."
    fi

    DS4_SRC_DIR="$DS4_SRC_DIR" DS4_GGUF_DIR="$DS4_GGUF_DIR" \
    GGUF_FILE="$GGUF_FILE" MTP_FILE="$MTP_FILE" DSPARK_FILE="$DSPARK_FILE" \
    nohup "$HOME/.local/bin/ds4-serve" ${flags[@]+"${flags[@]}"} \
        --port "$DS4_PORT" -c "$DS4_CTX" \
        > "$HOME/ds4-server.log" 2>&1 < /dev/null & disown
    local pid=$!
    log "ds4-server pid=$pid, log=$HOME/ds4-server.log (launcher: ds4-serve)"

    log "Waiting for /v1/models ..."
    local i
    for i in $(seq 1 60); do
        if curl -sf "http://127.0.0.1:$DS4_PORT/v1/models" >/dev/null 2>&1; then
            ok "Server up on http://127.0.0.1:$DS4_PORT"
            curl -s "http://127.0.0.1:$DS4_PORT/v1/models" | python3 -m json.tool 2>/dev/null || true
            return
        fi
        sleep 2
    done
    die "Server failed to come up within 120 s. Check $HOME/ds4-server.log."
}

# ============================================================================
# main
# ============================================================================

verify_host
clone_and_build
download_models
smoke_test
install_launcher
start_server

echo
ok "Done. Suggested next:"
echo "  $DS4_SRC_DIR/ds4-server --cuda -m $GGUF_PATH -c $DS4_CTX"
echo "  # then benchmark:"
echo "  uvx --from git+https://github.com/eugr/llama-benchy llama-benchy \\"
echo "      --base-url http://127.0.0.1:$DS4_PORT/v1 --model deepseek-v4-flash \\"
echo "      --pp 2048 --tg 32 128 --depth 0 4096 --latency-mode generation"
