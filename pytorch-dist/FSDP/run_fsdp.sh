#!/bin/bash
torchrun --nnodes 1 --nproc_per_node 2  \
	T5_training.py --epochs 3
