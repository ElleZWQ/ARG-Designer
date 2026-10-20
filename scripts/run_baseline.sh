#!/usr/bin/env bash
# Single-agent baselines: vanilla / cot / sc_cot, per-domain.
# Self-consistency (sc_cot) samples N=3 per question (paper doesn't specify N,
# so this value is recorded in each run's manifest for reproducibility).
#
# Usage:
#   ./scripts/run_baseline.sh gsm8k vanilla
#   ./scripts/run_baseline.sh gsm8k sc_cot
set -euo pipefail

DOMAIN="${1:-gsm8k}"
METHOD="${2:-vanilla}"     # vanilla | cot | sc_cot
LLM_NAME="GPT-5.6-Sol-joybuilder"
BATCH_SIZE=5              # stay low to avoid LLM rate limits
NUM_SAMPLES=3             # self-consistency sampling count (sc_cot only)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${REPO_ROOT}"
exec ${PYTHON} experiment/baseline.py \
    --domain "${DOMAIN}" \
    --method "${METHOD}" \
    --llm_name "${LLM_NAME}" \
    --batch_size "${BATCH_SIZE}" \
    --num_samples "${NUM_SAMPLES}"
