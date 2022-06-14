#!/bin/bash
ORION_VGPU_PER_NODE=2

export ORION_RES_GROUP=NULL
export ORION_VGPU=$ORION_VGPU_PER_NODE

TF_WORKER_ID=1 python main.py 2>&1 | tee worker_1.log &
TF_WORKER_ID=0 python main.py 2>&1 | tee worker_0.log
