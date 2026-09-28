#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible entry point. Pair generation and model evaluation are
# now separate so saved pairs can be reused without scanning maps again.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_subtask_pair_generation.sh" "$@"
