#!/bin/bash
set -x

export BIN="<YOUR_PYTHON_EXE_PATH>"
export PROJ_DIR=SPELL

export MASTER_IP_FILE="${PROJ_DIR}/.ray_head.txt"
echo "Master IP file for this job will be: ${MASTER_IP_FILE}"

cleanup() {
    echo "Performing cleanup for JOB_ID: ${JOB_ID}..."
    if [ "${NODE_RANK:-99}" -eq 0 ]; then
        if [ -f "${MASTER_IP_FILE}" ]; then
            rm -f "${MASTER_IP_FILE}"
            echo "Cleaned up master IP file: ${MASTER_IP_FILE}"
        fi
    fi
}

trap cleanup EXIT INT TERM

if ! command -v ip &> /dev/null; then
    apt-get update && apt-get install -y iproute2
fi

export NNODES=${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
export PORT=6231
export GPUS_PER_NODE=8

rollout_mode="sync"
rollout_name="vllm" 
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
fi

export PROJ_NAME="SPIRE"

export DOC_QA_METRIC="sub_em_strict"


cd ${PROJ_DIR}

export MASTER_IP=""

if [ $NODE_RANK -eq 0 ]; then
    echo "This is the HEAD node. Discovering and publishing IP..."
    
    MASTER_IP=$(ip addr show eth0 | grep "inet " | awk '{print $2}' | cut -d'/' -f1)
    echo "Head node IP address is: ${MASTER_IP}"

    TMP_IP_FILE="${MASTER_IP_FILE}.tmp"
    echo "${MASTER_IP}" > "${TMP_IP_FILE}"
    mv "${TMP_IP_FILE}" "${MASTER_IP_FILE}"
    echo "IP published to unique file: ${MASTER_IP_FILE}"

else
    echo "This is a WORKER node. Waiting for head node's IP to be published..."
    
    timeout 600s bash -c "while [ ! -f \"${MASTER_IP_FILE}\" ]; do echo 'Waiting...'; sleep 3; done"
    if [ $? -ne 0 ]; then
        echo "Error: Timed out after 600 seconds waiting for the master IP file. Exiting."
        exit 1
    fi

    MASTER_IP=$(cat "${MASTER_IP_FILE}")
    echo "Successfully retrieved head node IP: ${MASTER_IP}"
fi

if [ -z "${MASTER_IP}" ]; then
    echo "Error: MASTER_IP is empty after synchronization step. Cannot proceed."
    exit 1
fi

check_ray_status() {
    until ${BIN}/ray status >/dev/null 2>&1; do
        echo "Waiting for Ray cluster to be fully ready..."
        sleep 5
    done
}

if [ $NODE_RANK -eq 0 ]; then
    echo "Starting Ray HEAD node on ${MASTER_IP}..."
    ${BIN}/ray start --head --port=${PORT} --node-ip-address=${MASTER_IP}

    check_ray_status
    echo "Ray head node started successfully."

    echo "Starting the training script on the head node..."
    bash ${PROJ_DIR}/scripts/spire_qwen2.5_32b_instruct.sh

else
    echo "Starting Ray WORKER node, connecting to ${MASTER_IP}:${PORT}..."
    
    while true; do
        timeout 60s ${BIN}/ray start --address="${MASTER_IP}:${PORT}"
        if [ $? -eq 0 ]; then
            echo "Successfully joined Ray cluster."
            break
        fi
        echo "Failed to connect to master at ${MASTER_IP}, retrying in 5 seconds..."
        sleep 5
    done
    
    check_ray_status
    echo "Worker node has successfully joined the cluster and is ready."

    echo "Worker node is now idle and waiting for tasks."
    sleep infinity
fi

wait