#!/bin/bash
# Make room in the persistent HuggingFace cache for a large checkpoint.
#
# Usage (inside the ci_sglang container, where /sgl-data is the runner's
# persistent cache mount):
#   ensure_hf_cache_space.sh <required_gib> [keep_substring ...]
#
# Why this exists: /sgl-data/hf-cache is shared by every AMD job on the runner
# and nothing evicts whole checkpoints from it, so it fills with complete
# models and the next multi-hundred-GB download dies with
# "OSError: [Errno 28] No space left on device" partway through -- after tens of
# minutes of transfer, and only visible by reading the job log. This has been
# papered over twice by repointing CACHE_HOST at a fresh empty directory
# (#26642, #26905), which frees space by abandoning the old cache rather than
# managing it.
#
# Stale partial downloads are already handled by
# scripts/ci/utils/cleanup_hf_cache.py, which this delegates to rather than
# reimplementing; that script is wired into the CUDA runner prep but nothing on
# AMD called it before. What is genuinely new here is reporting free space and,
# when partial-download cleanup is not enough, evicting whole checkpoints
# least-recently-used first.
#
# Evicted checkpoints are re-downloaded by whichever job needs them next, so
# this trades another job's download time for this one's ability to run at all.
# On a cache that cannot hold every large model at once, something has to give;
# LRU order at least means the victim is whatever has gone longest unused.
#
# Never fails the job. A wrong size estimate should not block a run that would
# have fit, and if it truly does not fit the download reports that itself --
# now against a log that says exactly how much room there was.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REQUIRED_GIB="${1:?required free space in GiB}"
shift || true
KEEP_SUBSTRINGS=("$@")

HF_CACHE="${HF_HOME:-/sgl-data/hf-cache}/hub"

# Anything touched this recently may belong to a job running right now on a
# runner that shares this cache mount; deleting it out from under an in-flight
# download or weight load would turn our space problem into their crash. Matches
# the staleness window cleanup_hf_cache.py uses for partial downloads.
PROTECT_RECENT_MINUTES=120

avail_gib() {
    df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'
}

report() {
    echo "=== HF cache space ($1) ==="
    df -h "$HF_CACHE" 2>/dev/null || df -h /sgl-data 2>/dev/null || true
    echo "==========================="
}

is_kept() {
    local dir_name="$1" keep
    for keep in ${KEEP_SUBSTRINGS+"${KEEP_SUBSTRINGS[@]}"}; do
        [[ "$dir_name" == *"$keep"* ]] && return 0
    done
    return 1
}

ensure_hf_cache_space() {
    if [[ ! -d "$HF_CACHE" ]]; then
        echo "HF cache $HF_CACHE does not exist yet; nothing to free."
        return 0
    fi

    report "before"
    local avail
    avail=$(avail_gib "$HF_CACHE")
    echo "Need ${REQUIRED_GIB} GiB, have ${avail:-unknown} GiB free."
    if [[ -z "$avail" ]]; then
        echo "WARNING: could not read free space from df; skipping cleanup."
        return 0
    fi
    if (( avail >= REQUIRED_GIB )); then
        echo "✓ Enough free space already; nothing evicted."
        return 0
    fi

    # Abandoned partial downloads first -- they are never useful to anyone.
    python3 "${SCRIPT_DIR}/../utils/cleanup_hf_cache.py" || true
    avail=$(avail_gib "$HF_CACHE")
    echo "Free space after stale-artifact cleanup: ${avail} GiB."

    if (( avail >= REQUIRED_GIB )); then
        echo "✓ Reclaimed enough from partial downloads; no checkpoint evicted."
        report "after"
        return 0
    fi

    # Least-recently-modified checkpoints first.
    local evicted=0 dir name
    while IFS= read -r dir; do
        [[ -z "$dir" ]] && continue
        name=$(basename "$dir")
        if is_kept "$name"; then
            echo "  keep    $name (requested by caller)"
            continue
        fi
        echo "  evict   $name"
        rm -rf "$dir" 2>/dev/null || { echo "    failed to remove; skipping"; continue; }
        evicted=$((evicted + 1))
        avail=$(avail_gib "$HF_CACHE")
        echo "    free space now ${avail} GiB"
        (( avail >= REQUIRED_GIB )) && break
    done < <(find "$HF_CACHE" -maxdepth 1 -type d -name 'models--*' \
                 -mmin "+${PROTECT_RECENT_MINUTES}" -printf '%T@ %p\n' 2>/dev/null \
             | sort -n | cut -d' ' -f2-)

    report "after"
    if (( avail >= REQUIRED_GIB )); then
        echo "✓ Freed enough space after evicting ${evicted} checkpoint(s)."
    else
        echo "WARNING: only ${avail} GiB free after evicting ${evicted} checkpoint(s),"
        echo "         short of the ${REQUIRED_GIB} GiB requested. The download may"
        echo "         still fail; the runner cache likely needs more capacity than"
        echo "         the set of large checkpoints this fleet serves."
    fi
    return 0
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    ensure_hf_cache_space "$@"
fi
