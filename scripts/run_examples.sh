#!/usr/bin/env bash
#
# Run every example in examples/ headlessly and report pass/fail per system.
#
# One command to verify the package end to end: each of the six modelled
# systems (M1-M6) plus the L1 adaptive-control example. No GUI, no MuJoCo
# viewer, no network; the whole set finishes in well under a minute.
#
# Usage:
#   scripts/run_examples.sh            # run them all, quiet unless something fails
#   scripts/run_examples.sh --verbose  # also print each example's own output
#   scripts/run_examples.sh --list     # list what would run, and exit
#
# Exit status is 0 only if every example passed. An example that needs an
# optional extra that is not installed is reported SKIP and does not fail
# the run.
#
# The interpreter is `uv run --extra dev --extra l1 python` when uv is available,
# else `python3`. The `l1` extra is asked for so the L1 example actually runs
# rather than reporting SKIP under this script's own documented invocation.
# Override the interpreter with e.g. PYTHON=/path/to/python.
#
set -euo pipefail

# Resolve repo root from this script's location, so it works from anywhere.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VERBOSE=0
LIST=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose) VERBOSE=1; shift ;;
        --list) LIST=1; shift ;;
        -h|--help) sed -n '2,21p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# label|description|script
EXAMPLES=(
    "M1|quadrotor|examples/quadrotor.py"
    "M2|quadrotor + cable-suspended payload|examples/quadrotor_cspayload.py"
    "M3|multi-quad + point-mass payload|examples/multi_quad_cspayload.py"
    "M4|multi-quad + rigid-body payload|examples/multi_quad_rigidbody.py"
    "M5|quadrotor + flexible cable|examples/quadrotor_flexcable.py"
    "M6|multi-quad + flexible hose|examples/multi_quad_hose.py"
    "L1|adaptive control (needs the 'l1' extra)|examples/quadrotor_geometric_l1.py"
)

if [[ "$LIST" -eq 1 ]]; then
    for entry in "${EXAMPLES[@]}"; do
        IFS='|' read -r label desc script <<< "$entry"
        printf '  %-3s %-40s %s\n' "$label" "$desc" "$script"
    done
    exit 0
fi

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_CMD=("$PYTHON")
elif command -v uv >/dev/null 2>&1; then
    PYTHON_CMD=(uv run --extra dev --extra l1 python)
else
    PYTHON_CMD=(python3)
fi

echo ">> Running ${#EXAMPLES[@]} examples with: ${PYTHON_CMD[*]}"
echo

PASSED=0
SKIPPED=0
FAILED=()

for entry in "${EXAMPLES[@]}"; do
    IFS='|' read -r label desc script <<< "$entry"
    printf '  %-3s %-40s ' "$label" "$desc"

    start=$SECONDS
    # `set -e` must not abort on a failing example: we want the full report.
    set +e
    output="$("${PYTHON_CMD[@]}" "$script" 2>&1)"
    status=$?
    set -e
    elapsed=$((SECONDS - start))

    case "$status" in
        0) printf 'PASS  %3ds\n' "$elapsed"; PASSED=$((PASSED + 1)) ;;
        # 2 is the examples' agreed "optional extra missing" status.
        2) printf 'SKIP  %3ds\n' "$elapsed"; SKIPPED=$((SKIPPED + 1))
           echo "$output" | sed 's/^/       /' ;;
        *) printf 'FAIL  %3ds\n' "$elapsed"; FAILED+=("$label ($script)")
           echo "$output" | sed 's/^/       /' ;;
    esac

    if [[ "$VERBOSE" -eq 1 && "$status" -eq 0 ]]; then
        echo "$output" | sed 's/^/       /'
    fi
done

echo
echo ">> ${PASSED} passed, ${SKIPPED} skipped, ${#FAILED[@]} failed"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    for f in "${FAILED[@]}"; do
        echo "   FAILED: $f"
    done
    exit 1
fi
