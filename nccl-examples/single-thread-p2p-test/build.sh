#!/bin/bash
rm test
nvcc naive-p2p-test.cu -o p2p-test -lnccl
