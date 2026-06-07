#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <model-name> <endpoint-url>"
    echo "Example: $0 Llama-3.2-1B-Instruct-FP8 http://localhost:8000"
    exit 1
fi

MODEL_NAME="$1"
ENDPOINT="$2"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LM_EVAL_DIR="${SCRIPT_DIR}/metrics/lm-eval"
GUIDELLM_DIR="${SCRIPT_DIR}/metrics/guidellm"

mkdir -p "$LM_EVAL_DIR" "$GUIDELLM_DIR"

echo "=== Benchmarking: ${MODEL_NAME} ==="
echo "Endpoint: ${ENDPOINT}"

# Discover the model name from the vLLM /v1/models endpoint
SERVED_MODEL=$(curl -s "${ENDPOINT}/v1/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
echo "Served model ID: ${SERVED_MODEL}"

# Resolve HF tokenizer: use base model name since the served model ID is a local path
BASE_MODEL=$(echo "$MODEL_NAME" | sed -E 's/-(W[48]A1[68]|FP8|NVFP4|Baseline)$//')
if echo "$BASE_MODEL" | grep -q "Llama"; then
    TOKENIZER="meta-llama/${BASE_MODEL}"
elif echo "$BASE_MODEL" | grep -q "Qwen"; then
    TOKENIZER="Qwen/${BASE_MODEL}"
else
    TOKENIZER="$BASE_MODEL"
fi
echo "Tokenizer: ${TOKENIZER}"

echo ""
echo "--- lm-eval (wikitext perplexity) ---"
lm_eval \
    --model local-completions \
    --model_args "model=${SERVED_MODEL},base_url=${ENDPOINT}/v1/completions,tokenizer_backend=huggingface,tokenizer=${TOKENIZER},num_concurrent=1,max_retries=3" \
    --tasks wikitext \
    --num_fewshot 0 \
    --output_path "${LM_EVAL_DIR}/${MODEL_NAME}" \
    --log_samples

echo ""
echo "--- guidellm (throughput / latency) ---"
guidellm \
    --target "${ENDPOINT}/v1" \
    --model "${SERVED_MODEL}" \
    --rate-type sweep \
    --max-seconds 120 \
    --output-path "${GUIDELLM_DIR}/${MODEL_NAME}.json"

echo ""
echo "=== Done: ${MODEL_NAME} ==="
echo "Results:"
echo "  lm-eval:  ${LM_EVAL_DIR}/${MODEL_NAME}/"
echo "  guidellm: ${GUIDELLM_DIR}/${MODEL_NAME}.json"
