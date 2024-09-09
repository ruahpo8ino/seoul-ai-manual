export HF_HOME="/workspace/backup/model/"
export VLLM_ATTENTION_BACKEND=FLASHINFER
python3 -m vllm.entrypoints.api_server \
    --host 0.0.0.0 --port 8000 \
    --dtype "auto" \
    --model google/gemma-2-9b-it \
    -tp 1 --gpu-memory-utilization 0.7 --max-model-len 4096 --disable-sliding-window