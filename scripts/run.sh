#!/usr/bin/env bash
# ARG-Designer GSM8K reproduction runner.
# Single entry point: choose the stage (or "all") and parameters are driven by
# the variables below. No server credentials live here — this repo is public.
#
# Usage:
#   ./scripts/run.sh gsm8k stage1
#   ./scripts/run.sh gsm8k stage2
#   ./scripts/run.sh gsm8k stage3
#   ./scripts/run.sh gsm8k all
set -euo pipefail

# ------------------- Tunables -------------------------------
DOMAIN="${1:-gsm8k}"          # dataset domain (only gsm8k wired up for now)
STAGE="${2:-all}"             # stage1 | stage2 | stage3 | all
LLM_NAME="GPT-5.6-Sol-joybuilder"  # LLM backend model name
SEED=42                      # fixed seed for reproducibility
BATCH_SIZE=4                 # data-gen / training batch size
NUM_ITERATIONS=10            # Stage 1 train set = iterations * batch_size
EVAL_BATCH_SIZE=5            # Stage 3 parallel eval batch (keep low to avoid LLM rate limits)
OUTPUT_ROOT="output/${DOMAIN}" # unified artifact root (under repo root)
# ------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GSM8K_DIR="${REPO_ROOT}/experiment/${DOMAIN}"
PYTHON="${PYTHON:-python}"     # use $PYTHON env var if set, else "python"
export HF_HUB_OFFLINE=1        # force offline: read cached HF models, never hit the network
export TRANSFORMERS_OFFLINE=1 # same for transformers library

stamp() { date +"%Y%m%d-%H%M%S"; }

stage1_dir="${OUTPUT_ROOT}/stage1_cold_start_$(stamp)"
stage2_dir="${OUTPUT_ROOT}/stage2_finetune_$(stamp)"
stage3_dir="${OUTPUT_ROOT}/stage3_eval_$(stamp)"

run_stage1() {
  echo "== Stage 1: cold start =="
  ( cd "${GSM8K_DIR}" && ${PYTHON} cold_start_${DOMAIN}.py \
      --llm_name "${LLM_NAME}" \
      --seed "${SEED}" \
      --batch_size "${BATCH_SIZE}" \
      --num_iterations "${NUM_ITERATIONS}" \
      --output_dir "${REPO_ROOT}/${stage1_dir}" )
}

run_stage2() {
  echo "== Stage 2: finetune =="
  local cold="${REPO_ROOT}/output/${DOMAIN}/$(ls -1 "${REPO_ROOT}/output/${DOMAIN}" | grep "^stage1_cold_start_" | sort | tail -1)"
  ( cd "${GSM8K_DIR}" && ${PYTHON} finetune_${DOMAIN}.py \
      --pretrain \
      --llm_name "${LLM_NAME}" \
      --seed "${SEED}" \
      --batch_size "${BATCH_SIZE}" \
      --output_dir "${REPO_ROOT}/${stage2_dir}" \
      --cold_start_dir "${cold}" )
}

run_stage3() {
  echo "== Stage 3: evaluate =="
  local latest_stage1
  latest_stage1="$(ls -1 "${REPO_ROOT}/output/${DOMAIN}" 2>/dev/null | grep "^stage1_cold_start_" | sort | tail -1)"
  if [[ -z "${latest_stage1}" ]]; then
    echo "No stage1 output found; run stage1 first." >&2
    exit 1
  fi
  local cold_split="${REPO_ROOT}/output/${DOMAIN}/${latest_stage1}/task_split_${DOMAIN}.json"
  # The finetune script writes its model relative to its own CWD (experiment/<domain>),
  # so look there for the finetuned model dir.
  local model_dir="${GSM8K_DIR}/output/$(ls -1 "${GSM8K_DIR}/output" 2>/dev/null | grep "^${DOMAIN}_finetuned_model_" | sort | tail -1)"
  ( cd "${GSM8K_DIR}" && ${PYTHON} evaluate_${DOMAIN}.py \
      --model_path "${model_dir}" \
      --task_split_path "${cold_split}" \
      --llm_name "${LLM_NAME}" \
      --seed "${SEED}" \
      --eval_batch_size "${EVAL_BATCH_SIZE}" \
      --output_dir "${REPO_ROOT}/${stage3_dir}" )
}

case "${STAGE}" in
  stage1) run_stage1 ;;
  stage2) run_stage2 ;;
  stage3) run_stage3 ;;
  all)
    run_stage1
    run_stage2
    run_stage3
    ;;
  *)
    echo "Unknown stage '${STAGE}'. Use: stage1 | stage2 | stage3 | all" >&2
    exit 1
    ;;
esac

echo "Done: ${DOMAIN} ${STAGE}"
