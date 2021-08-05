#!/bin/bash
CUDA_VER=${1:-11.0}
MODE=${1:-0} # could be 0 1 2 3 4 5 6
rm multi_thread_*
mv /usr/lib/orion /usr/lib/orion-bak
nvcc -lnccl nccl_thread_all_api_all_mode_in_one.cu -o multi_thread_all_api_all_mode_in_one
mv /usr/lib/orion-bak /usr/lib/orion
echo "./multi_thread_all_api_all_mode_in_one $ORION_VGPU ${MODE}"
