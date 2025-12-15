
BASE_MODEL_NAME=Qwen3-30B-A3B-Thinking-2507
HF_MODEL_PATH="YOUR_BASE_MODEL_PATH"

MODEL_NAME=Qwen3-30B-A3B-Thinking-2507-SPELL

for STEP in 70 ;
do

    CKPT_PATH="YOUR_CHECKPOINT_PATH"
    LOCAL_DIR="${CKPT_PATH}/${MODEL_NAME}/global_step_${STEP}/actor"
    TARGET_DIR="${CKPT_PATH}/${MODEL_NAME}/global_step_${STEP}/huggingface"

    python model_merger.py \
        --backend "fsdp" \
        --hf_model_path $HF_MODEL_PATH \
        --local_dir $LOCAL_DIR \
        --target_dir $TARGET_DIR

    cp $HF_MODEL_PATH/merges.txt $TARGET_DIR
    cp $HF_MODEL_PATH/generation_config.json $TARGET_DIR
    cp $HF_MODEL_PATH/tokenizer_config.json $TARGET_DIR
    cp $HF_MODEL_PATH/tokenizer.json $TARGET_DIR
    cp $HF_MODEL_PATH/vocab.json $TARGET_DIR

done