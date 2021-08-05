#!/bin/bash
rm demo
# nvcc naive-group-mode.cu -o demo -lnccl
nvcc naive-reduce-test.cu -o demo -lnccl 
