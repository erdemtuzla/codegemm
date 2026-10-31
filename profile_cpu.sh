#!/usr/bin/env bash
# Optional launcher: collect CPU counters only during selected inference phases.
# Example:
# PERF_REPORT=benchmarks/pi_decode_perf.txt bash profile_cpu.sh \
#   --model-path packed_hf/MODEL --prompt-lengths 512 --perf-phase decode \
#   --output benchmarks/pi_decode_profile.json
set -euo pipefail

if ! command -v perf >/dev/null 2>&1; then
    echo 'perf is not installed; ordinary run_benchmark.py runs do not need it.' >&2
    exit 1
fi

benchmark_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
perf_report=${PERF_REPORT:-benchmarks/perf_cpu.txt}
if [[ -e "$perf_report" ]]; then
    echo "Report already exists; choose another PERF_REPORT: $perf_report" >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$perf_report")"
perf_events='{cache-references,cache-misses}'
if [[ -n "${PERF_EVENTS:-}" ]]; then
    perf_events=$PERF_EVENTS
fi

perf_tmp=$(mktemp -d)
cleanup() {
    rm -f -- "$perf_tmp/control" "$perf_tmp/ack"
    rmdir -- "$perf_tmp"
}
trap cleanup EXIT
mkfifo -- "$perf_tmp/control" "$perf_tmp/ack"

# Hold both FIFOs open before launching perf, preventing open-order deadlocks.
exec {benchmark_control_fd}<>"$perf_tmp/control"
exec {benchmark_ack_fd}<>"$perf_tmp/ack"

perf stat \
    --delay=-1 \
    --control="fifo:$perf_tmp/control,$perf_tmp/ack" \
    -e "$perf_events" \
    -o "$perf_report" \
    -- python "$benchmark_dir/run_benchmark.py" \
        --backend cpu --mode latency \
        --perf-control "$perf_tmp/control" \
        --perf-ack "$perf_tmp/ack" \
        "$@"

echo "Counter report: $perf_report"
echo 'Use ordinary unprofiled runs for latency comparisons.'
