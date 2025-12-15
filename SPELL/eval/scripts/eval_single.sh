#!bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD='spawn'

PROJ_DIR="SPELL/eval"

TP=0.7
TOP_P=0.95
TOP_K=-1

TASKS_LIST=("docmath" "frames" "2wikimqa" "hotpotqa" "musique" "longbench-v2") 

JUDGE_MDOEL="gpt-oss-120b"
VERIFIER_HOST="<your_api_host>"
VERIFIER_PORT="<your_api_port>"
API_BASE="http://${VERIFIER_HOST}:${VERIFIER_PORT}/v1"

N_SAMPLES=8

MAX_OUTPUT_LEN=20000

MODEL_NAME=Qwen3-30B-A3B-Thinking-2507
MODEL_PATH="<your_model_path>/${MODEL_NAME}"

mkdir -p ${PROJ_DIR}/results

for MAX_INPUT_LEN in 16384 100000
do
    # test all
    SAVE_NAME="${MODEL_NAME}_I${MAX_INPUT_LEN}_O${MAX_OUTPUT_LEN}_N${N_SAMPLES}"

    python ${PROJ_DIR}/generate.py \
        --input_dir "${PROJ_DIR}/data" \
        --save_dir "${PROJ_DIR}/results" \
        --save_file ${SAVE_NAME} \
        --model "${MODEL_PATH}" \
        --tokenizer "${MODEL_PATH}" \
        --tasks "${TASKS_LIST[@]}" \
        --n_sampling ${N_SAMPLES} \
        --temperature ${TP} \
        --top_p ${TOP_P} \
        --max_input_len ${MAX_INPUT_LEN} \
        --max_output_len ${MAX_OUTPUT_LEN} \
        --gpu_memory_utilization 0.9 \
        --top_k ${TOP_K} \
        --split ${N_SAMPLES} 

    python ${PROJ_DIR}/verify.py \
        --save_dir "${PROJ_DIR}/results" \
        --save_file ${SAVE_NAME} \
        --model "${JUDGE_MDOEL}" \
        --tasks "${TASKS_LIST[@]}" \
        --temperature 0.0 \
        --n_proc 200 \
        --top_p 1.0 \
        --max_input_len 8192 \
        --max_output_len 8192 \
        --top_k -1 \
        --api_key "EMPTY" \
        --api_base ${API_BASE} 

done
