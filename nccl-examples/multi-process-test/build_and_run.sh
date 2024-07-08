#!/bin/bash
rm demo

# lib not found issue
# export LIBRARY_PATH=$LIBRARY_PATH:/PATH/TO/libmpi.so
# export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/PATH/TO/mpi.h

nvcc -I/usr/local/openmpi/include/ -L/usr/local/openmpi/lib/ -lmpi -lnccl main_nccl.cpp -o demo

echo "mpirun -np $ORION_VGPU --allow-run-as-root demo"
