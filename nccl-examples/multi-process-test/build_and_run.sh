#!/bin/bash
rm demo
nvcc -I/usr/local/openmpi/include/ -L/usr/local/openmpi/lib/ -lmpi -lnccl main_nccl.cpp -o demo

echo "mpirun -np $ORION_VGPU --allow-run-as-root demo"
