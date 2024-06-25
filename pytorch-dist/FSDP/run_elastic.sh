#!/bin/bash
set -e
set -x

HOST_NODE_ADDR=${1:-'127.0.0.1'}
IS_MASTER=${2:-'0'}
JOB_ID=12315

master_cmd=''
if [ $IS_MASTER = '1' ];then
	master_cmd='--node-rank 0'
fi

export TORCH_LOGS=all
export HF_ENDPOINT=https://hf-mirror.com
# export WUICHAK_DEBUG=1

torchrun --nnodes 1:4 \
	--nproc_per_node 2 \
	${master_cmd} \
	--rdzv-backend=c10d \
	--rdzv-endpoint=$HOST_NODE_ADDR \
	--rdzv-id=$JOB_ID \
	--max-restarts=2 \
	T5_training.py --epochs 5
