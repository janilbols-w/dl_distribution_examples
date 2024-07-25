#!/bin/bash
NUM_GPUS=${NUM_GPUS:-'2'}
rm demo || echo do nothing
# export LIBRARY_PATH=$LIBRARY_PATH:/opt/hpcx/ompi/lib/
# export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/opt/hpcx/ompi/include/

nvcc -I/usr/local/openmpi/include/ -L/usr/local/openmpi/lib/ -lmpi -lnccl main_nccl.cu -o demo

echo "mpirun -np $NUM_GPUS --allow-run-as-root demo"

