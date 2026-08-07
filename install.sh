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
#   2. Clones (or fast-forwards) the ds4 fork at the pinned tag into $DS4_SRC_DIR.
#   3. Builds ds4, ds4-server, ds4-bench with `make cuda-spark` (GB10 default).
#      A non-GB10 Blackwell SKU builds with `make cuda CUDA_ARCH=<--cuda-arch>`.
#   4. Downloads the 0731 Q2 quantized GGUF (~87 GiB) from
#      antirez/deepseek-v4-gguf and the matching 0731 DSpark Q2K drafter
#      (~7 GiB) into $DS4_GGUF_DIR. The 0731 base has NO MTP head (the
#      single-block MTP module was replaced upstream by the DSpark
#      stages), so nothing MTP is downloaded for it. Previous-generation
#      weights are detected and offered for removal — prompted, optional,
#      and never silent (see --remove-old-weights / --keep-old-weights).
#   5. Runs a single-prompt smoke test against the canonical
#      "capital of France" prompt — expects "Paris" in the output.
#   6. Optionally (--start) serves the full DSpark speculative stack on
#      $DS4_PORT (lossless; yield-quench rides the launch defaults).
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

# Pin (updated 2026-08-08): the Entrpi/ds4 fork release tag v0.5.6 —
# the first-class API release on v0.5.5 (0731 weights, unchanged). The
# Anthropic Messages API and the OpenAI Responses API are now
# first-class surfaces of the batched engine: buffered and streaming
# requests on all four APIs run in the continuous batch, including
# thinking, stop sequences, and streaming tool calls, with streams
# starting at admission so first bytes arrive during prefill.
# Tool-calling agents get a real continuation contract: a finished
# tool call publishes a record binding it to the engine state that
# produced it, the tool-result follow-up continues from that state in
# place instead of re-ingesting the conversation, and stale state
# answers a clean 409 (resend full history, which always works).
# Also: stop-sequence hits report the matched text, max_tokens 0 means
# zero on every lane, server errors use each API's native shape with
# Retry-After where a retry makes sense, usage always matches timings
# (cache read/write detail included), and explicit admission bounds
# refuse honestly instead of queueing without limit. The new batched
# routing is the default; setting DS4_SERVER_CONT_ANTHROPIC=0,
# DS4_SERVER_CONT_RESPONSES=0, DS4_SERVER_CONT_TOOLS_ANTHROPIC=0 or
# DS4_SERVER_CONT_TOOLS_RESPONSES=0 restores the old serial routing
# per surface for one release only. Perf at parity with v0.5.5 (ABBA
# within noise at N=1/8/16; serving semantics pinned by unit oracles):
# the standing records and charts stand. Full release battery green on
# the tagged tree, including the 240k deep-serving gate (now run at
# shipped defaults, no tuned environment), the tool-eval suite
# (83/100/86/84, byte-identical to v0.5.5), and cross-box golden
# logprob vectors at zero deviation. Set DS4_REF=main +
# DS4_REPO=antirez/ds4 for the upstream engine without the fork's
# serving stack.
DS4_REPO="${DS4_REPO:-https://github.com/Entrpi/ds4.git}"
DS4_REF="${DS4_REF:-v0.5.6}"
DS4_SRC_DIR="${DS4_SRC_DIR:-$HOME/code/ds4}"
DS4_GGUF_DIR="${DS4_GGUF_DIR:-$HOME/gguf}"

CUDA_ARCH="${CUDA_ARCH:-sm_121}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc 2>/dev/null || echo 4)}"

HF_REPO="${HF_REPO:-antirez/deepseek-v4-gguf}"
# The 0731-refresh imatrix-tuned q2 (DeepSeek-V4-Flash-0731 weights, same
# ship recipe the fork quality baselines are stamped on). Overriding
# GGUF_FILE back to the pre-0731 name restores the full legacy behavior,
# including the MTP download — generation is detected from the file name.
GGUF_FILE="${GGUF_FILE:-DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf}"
MTP_FILE="${MTP_FILE:-DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf}"

DSPARK_HF_REPO="${DSPARK_HF_REPO:-bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF}"
DSPARK_FILE="${DSPARK_FILE:-DSpark-drafter-Q2K-Q8-0731.gguf}"

# Previous-generation weight files (the pre-0731 set). These are what the
# upgrade path detects and offers to remove: the old base is superseded, the
# old MTP head does not exist in the 0731 checkpoint (module replaced), and
# the old drafter was distilled against the old base's hidden states — all
# three are dead weight next to a 0731 install (~91 GiB total).
LEGACY_GGUF_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
LEGACY_MTP_FILE="DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
LEGACY_DSPARK_FILE="DSpark-drafter-Q2K-Q8.gguf"

DS4_PORT="${DS4_PORT:-8000}"
DS4_CTX="${DS4_CTX:-32768}"

FORCE_HW=0
SKIP_BUILD=0
IS_GB10=0          # set by verify_host; selects the post-build arch check only
SKIP_DOWNLOAD=0
SKIP_MTP=0
SKIP_SMOKE=0
START_SERVER=0
REMOVE_OLD=0
KEEP_OLD=0
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
                          downloads the ~7 GiB Q2K drafter from
                          $DSPARK_HF_REPO.
  --no-dspark             Serve plain continuous decode (no drafter download).
  --no-smoke              Skip post-install smoke test.
  --start                 Start ds4-server on :$DS4_PORT after install.
  --remove-old-weights    Delete previous-generation GGUFs without prompting.
                          When disk space is short they are removed BEFORE the
                          download; otherwise only after the new weights pass
                          the smoke test.
  --keep-old-weights      Keep previous-generation GGUFs, skip the prompt
                          (upgrade still fails early if they must go to make
                          the download fit).
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
        --remove-old-weights) REMOVE_OLD=1; shift ;;
        --keep-old-weights) KEEP_OLD=1; shift ;;
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

# Model generation, detected from the base file name so env overrides keep
# working in both directions. The 0731 checkpoint has no MTP head: the
# single-block MTP module was replaced upstream by the DSpark stages, so the
# legacy MTP GGUF must never be paired with a 0731 base.
if [[ "$GGUF_FILE" == *-0731* ]]; then
    MODEL_GEN="0731"
    MTP_SUPPORTED=0
else
    MODEL_GEN="legacy"
    MTP_SUPPORTED=1
fi

if [[ "$REMOVE_OLD" -eq 1 ]] && [[ "$KEEP_OLD" -eq 1 ]]; then
    echo "Pass at most one of --remove-old-weights / --keep-old-weights." >&2
    exit 2
fi

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

    if echo "$gpu_info" | grep -qE 'compute_cap.*12\.1|GB10|Spark'; then
        IS_GB10=1
    elif [[ "$FORCE_HW" -eq 0 ]]; then
        warn "Not detecting GB10 / SM12.1. ds4 may still work on other Blackwell SKUs."
        warn "Pass --cuda-arch sm_120 (or matching) and --force to proceed."
        die "Host check failed. Pass --force to skip."
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
        if [[ -f "$DS4_GGUF_DIR/$LEGACY_GGUF_FILE" ]] || [[ -f "$GGUF_PATH" ]]; then
            # Upgrade or partial install: the real space math (including the
            # optional old-weight removal) runs in upgrade_preflight.
            warn "Only ${free_gib} GiB free under $HOME — deferring to the upgrade space check."
        elif [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
            die "Need >= 120 GiB free under $HOME; have ${free_gib} GiB. Pass --no-download to skip GGUF, or free space."
        else
            warn "Only ${free_gib} GiB free under $HOME; --no-download is set, continuing."
        fi
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

    # As of upstream commit be43477 ("Standardize context length errors", 2026-05-15)
    # the default `make` target prints help instead of building. The named targets
    # are `make cuda CUDA_ARCH=...`, `make cuda-spark`, `make cuda-generic`,
    # `make cpu`.
    #
    # Target selection is driven by the RESOLVED CUDA_ARCH, never by the detected
    # hardware, so `--cuda-arch` keeps deciding the build on every non-GB10 SKU:
    #
    #   sm_121 / sm_121a  -> make cuda-spark   (GB10; also the default arch)
    #   anything else     -> make cuda CUDA_ARCH=$CUDA_ARCH   (unchanged path,
    #                        e.g. RTX PRO 6000 Blackwell via --cuda-arch sm_120)
    #
    # Field report (forum 378855 post 65, sword_fish): every install before the
    # v0.5.5 pin ran `make cuda CUDA_ARCH=sm_121`, which sets the sm_121a
    # gencode pair and DS4_CUDA_HAVE_MXF4 but not the Spark-specific defines, so
    # GB10 installs missed the Spark build entirely. `make cuda-spark` hardcodes
    # CUDA_ARCH=sm_121 itself, so we must not pass CUDA_ARCH to it, and it
    # builds -B (a partial rebuild across a compile-config change would silently
    # mix objects). Note for v0.5.6+: upstream measurement showed the old HBM
    # weight-read cache define changed memory planning, not kernel speed, and
    # its startup memory charge could starve deep boots — the cuda-spark target
    # no longer sets it, and the target remains the single source of truth for
    # what a Spark build enables.
    case "$CUDA_ARCH" in
        sm_121|sm_121a)
            log "Building ds4 for DGX Spark: make cuda-spark (Spark fast paths, -j$BUILD_JOBS) ..."
            ( cd "$DS4_SRC_DIR" && make cuda-spark -j"$BUILD_JOBS" )
            ;;
        *)
            log "Building ds4, ds4-server, ds4-bench (CUDA_ARCH=$CUDA_ARCH, -j$BUILD_JOBS) ..."
            if [[ "$IS_GB10" -eq 1 ]]; then
                warn "This host is a GB10 but --cuda-arch $CUDA_ARCH was requested, so the generic"
                warn "build is used and the Spark fast paths are compiled out. Drop --cuda-arch (or"
                warn "pass --cuda-arch sm_121) for the fast build."
            fi
            ( cd "$DS4_SRC_DIR" && make cuda -j"$BUILD_JOBS" CUDA_ARCH="$CUDA_ARCH" )
            ;;
    esac

    for bin in ds4 ds4-server ds4-bench; do
        [[ -x "$DS4_SRC_DIR/$bin" ]] || die "Build did not produce $bin"
    done

    # sword_fish's second suggestion: prove the binary matches the hardware
    # instead of trusting the make target. Non-fatal — a missing/!=0 cuobjdump
    # must never fail an otherwise good install.
    #
    # v0.5.6: the check is capture-based, never `| grep -q`. Under pipefail,
    # grep -q exits at the first match and cuobjdump dies of SIGPIPE (141), so
    # the old pipeline reported "no sm_121a SASS" precisely when the match WAS
    # found early in the dump — a false warning on every good fast build. An
    # EMPTY capture is also distinguished from a real mismatch: silence must
    # never verify anything.
    if [[ "$IS_GB10" -eq 1 ]]; then
        local cuobjdump="/usr/local/cuda/bin/cuobjdump"
        [[ -x "$cuobjdump" ]] || cuobjdump=$(command -v cuobjdump 2>/dev/null || true)
        if [[ -n "$cuobjdump" ]] && [[ -x "$cuobjdump" ]]; then
            local sass_arches
            sass_arches=$("$cuobjdump" "$DS4_SRC_DIR/ds4-server" 2>/dev/null | grep 'arch = ' || true)
            if [[ "$sass_arches" == *sm_121a* ]]; then
                ok "Verified: ds4-server carries sm_121a SASS."
            elif [[ -z "$sass_arches" ]]; then
                warn "could not read SASS arches from ds4-server (cuobjdump gave no output) — skipping the arch check."
            else
                warn "ds4-server does NOT carry sm_121a SASS (found: $(echo "$sass_arches" | tr '\n' ' ')). Expect a boot warning and reduced speed."
            fi
        fi
    fi

    ok "Built: $DS4_SRC_DIR/{ds4,ds4-server,ds4-bench}"
}

# ============================================================================
# 2b. upgrade path — previous-generation weight detection + optional removal
# ============================================================================

# Filled by upgrade_scan: absolute paths of legacy files present on disk.
LEGACY_PRESENT=()
LEGACY_BYTES=0

file_bytes() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0; }
gib() { awk -v b="$1" 'BEGIN{printf "%.1f", b/1073741824}'; }

upgrade_scan() {
    LEGACY_PRESENT=(); LEGACY_BYTES=0
    [[ "$MODEL_GEN" == "0731" ]] || return 0   # nothing is "old" for a legacy install
    local f
    for f in "$LEGACY_GGUF_FILE" "$LEGACY_MTP_FILE" "$LEGACY_DSPARK_FILE"; do
        local p="$DS4_GGUF_DIR/$f"
        if [[ -f "$p" ]]; then
            LEGACY_PRESENT+=("$p")
            LEGACY_BYTES=$(( LEGACY_BYTES + $(file_bytes "$p") ))
        fi
    done
    [[ ${#LEGACY_PRESENT[@]} -gt 0 ]] || return 0

    log "Upgrade: previous-generation weights found in $DS4_GGUF_DIR:"
    for p in "${LEGACY_PRESENT[@]}"; do
        printf '    %7s GiB  %s\n' "$(gib "$(file_bytes "$p")")" "${p##*/}"
    done
    log "They are dead weight next to a 0731 install ($(gib "$LEGACY_BYTES") GiB reclaimable):"
    log "  - the old base is superseded by the 0731 refresh;"
    log "  - the old MTP head has no 0731 counterpart (module replaced upstream)"
    log "    and must never be served with a 0731 base;"
    log "  - the old drafter was distilled against the old base and would crater"
    log "    speculative accept on 0731 (output stays correct, speed does not)."
}

# ask_remove_old "when" -> 0 = remove, 1 = keep. Honors the two flags; reads
# the answer from /dev/tty so `curl | bash` still prompts; non-interactive
# runs without a flag keep the files (safe default) and say so.
ask_remove_old() {
    local when="$1"
    [[ "$REMOVE_OLD" -eq 1 ]] && return 0
    [[ "$KEEP_OLD"  -eq 1 ]] && return 1
    # A controlling terminal must actually OPEN — permission bits on the
    # /dev/tty node pass even in detached (cron/CI) contexts.
    if ( : < /dev/tty ) 2>/dev/null; then
        local ans=""
        printf 'Remove the %s previous-generation file(s) (%s GiB) %s? [y/N] ' \
            "${#LEGACY_PRESENT[@]}" "$(gib "$LEGACY_BYTES")" "$when" > /dev/tty
        read -r ans < /dev/tty || true
        [[ "$ans" =~ ^[Yy] ]]
    else
        warn "Non-interactive and neither --remove-old-weights nor --keep-old-weights given — keeping old weights."
        return 1
    fi
}

remove_old_weights() {
    local p
    for p in "${LEGACY_PRESENT[@]}"; do
        log "Removing ${p##*/} ($(gib "$(file_bytes "$p")") GiB)"
        rm -f -- "$p"
    done
    ok "Previous-generation weights removed ($(gib "$LEGACY_BYTES") GiB reclaimed)."
    LEGACY_PRESENT=(); LEGACY_BYTES=0
}

# Decide whether the download fits with the old weights still on disk. If it
# does not, removal (consented) happens BEFORE the download; the resumable
# curl means a failed download can always be re-run.
upgrade_preflight() {
    [[ "$SKIP_DOWNLOAD" -eq 0 ]] || return 0
    [[ ${#LEGACY_PRESENT[@]} -gt 0 ]] || return 0

    local need=0 free_b
    # Bytes still to download: new base + drafter, minus whatever is present.
    [[ -f "$GGUF_PATH"   ]] || need=$(( need + 94489280512 ))  # ~88 GiB ceiling
    if [[ "$WITH_DSPARK" -eq 1 ]] && [[ ! -f "$DSPARK_PATH" ]]; then
        need=$(( need + 8589934592 ))                          # ~8 GiB ceiling
    fi
    free_b=$( (df -B1 "$DS4_GGUF_DIR" 2>/dev/null || true) | awk 'NR==2{print $4}')
    free_b=${free_b:-0}

    if (( free_b >= need )); then
        return 0   # fits alongside; the removal offer comes after the smoke test
    fi
    warn "Free space under $DS4_GGUF_DIR is $(gib "$free_b") GiB; the remaining download needs ~$(gib "$need") GiB."
    if (( free_b + LEGACY_BYTES < need )); then
        die "Even removing the old weights ($(gib "$LEGACY_BYTES") GiB) will not fit the download. Free space and re-run."
    fi
    log "Removing the previous-generation weights first would make it fit."
    if ask_remove_old "now, before the download"; then
        remove_old_weights
    else
        die "Not enough space with the old weights kept. Re-run with --remove-old-weights (or free space) to upgrade."
    fi
}

# After the new weights exist and the smoke test passed, offer the (optional)
# cleanup — never before the new base is proven to answer.
upgrade_cleanup_offer() {
    [[ ${#LEGACY_PRESENT[@]} -gt 0 ]] || return 0
    if [[ "$SKIP_SMOKE" -eq 1 ]] && [[ "$REMOVE_OLD" -eq 0 ]]; then
        warn "Smoke test was skipped — keeping old weights (remove manually or re-run with --remove-old-weights)."
        return 0
    fi
    if ask_remove_old "now that the new weights passed the smoke test"; then
        remove_old_weights
    else
        log "Keeping previous-generation weights (re-run with --remove-old-weights to reclaim $(gib "$LEGACY_BYTES") GiB)."
    fi
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
    log "Downloading $file from $repo ..."
    curl -L --fail --progress-bar -C - -o "$dest" "$url" || return 1
    ok "Downloaded $file"
}

download_models() {
    if [[ "$SKIP_DOWNLOAD" -eq 1 ]]; then
        log "Skipping GGUF download (--no-download)."
        return
    fi
    download_one "$GGUF_FILE" "$GGUF_PATH"
    if [[ "$MTP_SUPPORTED" -eq 0 ]]; then
        if [[ "$WITH_MTP" -eq 1 ]] && [[ "$SKIP_MTP" -eq 0 ]]; then
            log "MTP: nothing to download — the 0731 base has no MTP head (module replaced upstream by the DSpark stages)."
        fi
    elif [[ "$WITH_MTP" -eq 1 ]] && [[ "$SKIP_MTP" -eq 0 ]]; then
        download_one "$MTP_FILE" "$MTP_PATH"
    fi
    if [[ "$WITH_DSPARK" -eq 1 ]]; then
        if ! download_one "$DSPARK_FILE" "$DSPARK_PATH" "$DSPARK_HF_REPO"; then
            warn "DSpark drafter $DSPARK_FILE not downloadable from $DSPARK_HF_REPO (not published yet?)."
            warn "Continuing without it — the server will boot plain; re-run the installer later to pick it up."
            rm -f -- "$DSPARK_PATH"
        fi
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
    [[ -f "$GGUF_PATH" ]] || { warn "$GGUF_PATH missing — skipping smoke test."; return 0; }

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
            || { warn "could not install ds4-serve launcher (offline?); full command is in the README."; return 0; }
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
    # `|| return` (no argument) propagates the FAILED test's status 1, and
    # under set -e that killed every non---start install at the last step —
    # the script exited 1 after doing everything right. Explicit 0.
    [[ "$START_SERVER" -eq 1 ]] || return 0
    [[ -f "$GGUF_PATH" ]] || die "$GGUF_PATH missing — cannot start server."

    local flags=()
    if [[ "$WITH_DSPARK" -eq 1 ]] && [[ -f "$DSPARK_PATH" ]]; then
        log "Starting ds4-server with DSpark speculative decode (MTP head dropped when a drafter is armed)."
    elif [[ "$MTP_SUPPORTED" -eq 1 ]] && [[ "$WITH_MTP" -eq 1 ]] && [[ -f "$MTP_PATH" ]]; then
        flags+=(--no-dspark)
        log "Starting ds4-server with MTP-2 speculation (continuous batching)."
    else
        flags+=(--no-spec)
        if [[ "$MTP_SUPPORTED" -eq 0 ]] && [[ ! -f "$DSPARK_PATH" ]]; then
            log "Starting ds4-server plain — no 0731 drafter on disk and the 0731 base has no MTP fallback."
        else
            log "Starting ds4-server (plain continuous decode; add --with-mtp or --with-dspark for speculation)."
        fi
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
            # Field report 378855/65 (sword_fish): the server can be up, answer
            # correctly, and still be a build that serves at a fraction of the
            # advertised speed. The serving binary is the only thing that can
            # see its own compile config, and since ds4 v0.5.5 it says so at
            # boot. Surface that verdict in the install output — a cuobjdump
            # arch check CANNOT catch this (both configs carry sm_121a SASS).
            # Match the v0.5.5 advisory wording AND the split v0.5.6+ wordings
            # (generic build / HBM-cache-only) — see ds4_server.c around the
            # ds4_cuda_spark_build_mismatch call.
            if grep -qE 'built without the DGX Spark configuration|binary is a generic CUDA build|Spark HBM weight cache is compiled out' "$HOME/ds4-server.log" 2>/dev/null; then
                warn "==================================================================="
                warn "ds4-server reports it was built WITHOUT the DGX Spark configuration"
                warn "and will serve well below the advertised speed on this GB10."
                warn "This is an installer/build defect, not a hardware problem. Fix:"
                warn "    cd $DS4_SRC_DIR && make cuda-spark -j$BUILD_JOBS"
                warn "then restart with: ds4-serve"
                warn "==================================================================="
            elif [[ "$IS_GB10" -eq 1 ]]; then
                ok "Verified: no build-mismatch advisory in the boot log (DGX Spark fast paths active)."
            fi
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
upgrade_scan
upgrade_preflight
clone_and_build
download_models
smoke_test
upgrade_cleanup_offer
install_launcher
start_server

echo
ok "Done. Suggested next:"
echo "  $DS4_SRC_DIR/ds4-server --cuda -m $GGUF_PATH -c $DS4_CTX"
echo "  # then benchmark:"
echo "  uvx --from git+https://github.com/eugr/llama-benchy llama-benchy \\"
echo "      --base-url http://127.0.0.1:$DS4_PORT/v1 --model deepseek-v4-flash \\"
echo "      --pp 2048 --tg 32 128 --depth 0 4096 --latency-mode generation"
