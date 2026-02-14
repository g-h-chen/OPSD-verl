#!/bin/bash
set -x

export RAY_MASTER_PORT=6379
export RAY_DASHBOARD_PORT=8265
export MASTER_PORT=34235
export TF_CPP_MIN_LOG_LEVEL=3

# Set up the Ray temporary directory
export RAY_TMPDIR=/dev/shm/ray_wwy
rm -rf ${RAY_TMPDIR}
mkdir -p ${RAY_TMPDIR}

# Set the task name
CURRENT_PATH=$(pwd)
PROJECT_NAME=internvl3_5_14b_custom
TASK_NAME=$(basename "$0")
TASK_NAME="${TASK_NAME%.*}"
echo "TASK_NAME: $TASK_NAME"
echo "PROJECT_NAME: $PROJECT_NAME"

export OUTPUT_PATH=${CURRENT_PATH}/verl_internvl_work_dirs/${PROJECT_NAME}/${TASK_NAME}
export TENSORBOARD_DIR=${OUTPUT_PATH}/tensorboard
export JOBLOG=${OUTPUT_PATH}/training.log
RAY_WORKING_DIR="${OUTPUT_PATH}/ray_working_dir/"

# Create output directory if it does not exist
mkdir -p ${OUTPUT_PATH}
mkdir -p ${TENSORBOARD_DIR}
mkdir -p ${RAY_WORKING_DIR}

# Set up environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TRITON_CACHE_DIR="/dev/shm/triton_wwy/"
export VLLM_CACHE_ROOT="/dev/shm/vllmca_wwy/"

echo "start ray worker" &>> ${JOBLOG}

if [ "$RANK" -eq 0 ]; then
    ray start --head  --port=$RAY_MASTER_PORT --dashboard-host=0.0.0.0 --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus 8
    echo "Main node finished"
else
    sleep 30
    echo "Node started"
    ray start --address="$MASTER_ADDR:$RAY_MASTER_PORT" --num-gpus 8 --block
fi

echo "submit ray job" &>> ${JOBLOG}

sleep 30
ray status
ray list nodes
echo $MASTER_ADDR

# Dynamically compute the number of nodes and total GPU engines
NUM_GPUS_PER_NODE=8
MICRO_TRAIN_BATCH_SIZE=8
MICRO_ROLLOUT_BATCH_SIZE=8
ROLLOUT_BATCH_SIZE=512
N_SAMPLES_PER_PROMPT=16
TENSOR_PARALLEL=1
SEQUENCE_PARALLEL=4
PPO_MINI_BATCH_SIZE=32
RAY_ADDRESS="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}"

NPROC_PER_NODE=8
use_dynamic_bsz=True

# ===== Submit the ray job =====
ray job submit --address=${RAY_ADDRESS} \
    -- python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${CURRENT_PATH}/MMPR-Tiny/mmpr_tiny.parquet \
    data.val_files=${CURRENT_PATH}/verl_data/geo3k/test.parquet \
    data.train_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.max_prompt_length=8192 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    data.truncation='error' \
    data.image_key=images \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=OpenGVLab/InternVL3_5-14B-MPO \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SEQUENCE_PARALLEL} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_ROLLOUT_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_PARALLEL} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=${N_SAMPLES_PER_PROMPT} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_TRAIN_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SEQUENCE_PARALLEL} \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.default_local_dir=${OUTPUT_PATH} \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${TASK_NAME} \
    trainer.n_gpus_per_node=${NPROC_PER_NODE} \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=10 \
    trainer.test_freq=5000 \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=${OUTPUT_PATH}/rollouts \
    trainer.total_epochs=1 2>&1 | tee ${JOBLOG}
