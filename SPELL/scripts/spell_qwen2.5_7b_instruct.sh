set -x
export HYDRA_FULL_ERROR=1
ulimit -n 65535

MODEL_PATH=<YOUR_MODEL_PATH>
MODEL_NAME=Qwen2.5-7B-Instruct

PROJ_DIR=SPELL

LOG_DIR=${PROJ_DIR}/logs
CHECKPOINT_DIR=${PROJ_DIR}/checkpoints

mkdir -p ${LOG_DIR}
mkdir -p ${CHECKPOINT_DIR}

export PROJ_NAME="SPELL"

# rule-bases rm type
export DOC_QA_METRIC="sub_em_strict"

export rollout_mode="sync"
export rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
fi

GLOBAL_BS=128 # train batch size
GEN_BS=480 # gen batch size
MINI_BS=128  # minibatch size

TP=0.7
TOP_P=0.95
ENTROPY_COE=0.0
KL_COE=0.0
RO=8 # responder group size 
LR=2e-6
CLIP_HIGH=0.2

# questioner config
MAPPING_FUNC=gaussian # 'gaussian' 'reverse' 'rzero'
BAD_CASE_RATIO=1 # bad case ratio for questioner, set to 1 to ablate it
CACHE_SIZE=3

# verifier config
JUDGE_RO=8 # verifier group size 
VERIFIER_UPDATE_RATIO=0.125 # set to 1/n 
VERIFIER_LABEL=maj_cons
UPDATE_VERIFIER=True 

SP_SIZE=1
TP_SIZE=1

MAX_TOKENS=32000
MAX_TOENS_PER=32000

MAX_INPUT=18000
MAX_OUTPUT=4096 # for reasoning models, set to 10k or longger

PREFIX_NAME=${MODEL_NAME}-${MAPPING_FUNC}-N${RO}-Verifier-N${JUDGE_RO}
EXPERIMENT_NAME=${PREFIX_NAME}

TEST_DATA_16K=${PROJ_DIR}/dataset/test.parquet

DOCMATH_DATA_16K=${PROJ_DIR}/dataset/docmath_qa/train.parquet

ULTRA_FINEWEB_16K=${PROJ_DIR}/dataset/ultra_fineweb/train.parquet

VAL_ROLLOUT_SAVE_DIR=${LOG_DIR}/test_gen
TRAIN_ROLLOUT_SAVE_DIR=${LOG_DIR}/train_gen

mkdir -p ${VAL_ROLLOUT_SAVE_DIR}
mkdir -p ${TRAIN_ROLLOUT_SAVE_DIR}

python -m verl.trainer.main_ray_spell \
    data.train_files=["${DOCMATH_DATA_16K}","${ULTRA_FINEWEB_16K}"] \
    data.val_files=["${TEST_DATA_16K}"] \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    reward_model.reward_manager="thread" \
    data.prompt_key=prompt \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.max_prompt_length=${MAX_INPUT} \
    data.max_response_length=${MAX_OUTPUT} \
    data.gen_batch_size=${GEN_BS} \
    data.train_batch_size=${GLOBAL_BS} \
    data.use_cache.enable=True \
    data.use_cache.cached_lower=0.01 \
    data.use_cache.cached_higher=1.0 \
    data.use_cache.cache_size=${CACHE_SIZE} \
    data.n_docs.docmath=[5] \
    data.n_docs.ultrafineweb=[5] \
    data.tasks.docmath=['docmath_qa'] \
    data.tasks.ultrafineweb=['doc_general_qa','doc_mc'] \
    data.questioner_bad_case_ratio=${BAD_CASE_RATIO} \
    data.filter_questioner_prompts=True \
    algorithm.reward_combined_function=max \
    algorithm.domain_sampling.enable=True \
    algorithm.domain_sampling.update_weights=False \
    algorithm.domain_sampling.init_weights=[1,1,1] \
    algorithm.domain_sampling.init_weight_method=predefined \
    algorithm.questioner.reward_type=${MAPPING_FUNC} \
    algorithm.questioner.group=batch \
    algorithm.questioner.update=True \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=10 \
    algorithm.filter_groups.metric=score \
    algorithm.self_verification.enable=True \
    algorithm.self_verification.label_type=${VERIFIER_LABEL} \
    algorithm.self_verification.tasks=['doc_general_qa','docmath_qa','doc_mc'] \
    algorithm.self_verification.update=${UPDATE_VERIFIER} \
    algorithm.self_verification.reward_type=most \
    algorithm.self_verification.n=${JUDGE_RO} \
    algorithm.self_verification.update_ratio=${VERIFIER_UPDATE_RATIO} \
    algorithm.self_verification.update_lower_bound=0.51 \
    algorithm.self_verification.update_upper_bound=0.99 \
    actor_rollout_ref.rollout.n=${RO} \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COE} \
    algorithm.kl_ctrl.kl_coef=${KL_COE} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COE} \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_HIGH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOENS_PER} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOENS_PER} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOENS_PER} \
    actor_rollout_ref.model.path=${MODEL_PATH}/${MODEL_NAME} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BS} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_TOKENS} \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=${TOP_P} \
    actor_rollout_ref.rollout.temperature=${TP} \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=${TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${TP} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SP_SIZE}  \
    trainer.logger=['console'] \
    trainer.project_name=${PROJ_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.validation_data_dir=${VAL_ROLLOUT_SAVE_DIR}/${EXPERIMENT_NAME} \
    trainer.rollout_data_dir=${TRAIN_ROLLOUT_SAVE_DIR}/${EXPERIMENT_NAME}  \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.default_local_dir=${CHECKPOINT_DIR}/${EXPERIMENT_NAME} \
    trainer.resume_mode=auto \
    trainer.total_epochs=1000 "${@:1}" \
    | tee ${LOG_DIR}/${EXPERIMENT_NAME}.log 
