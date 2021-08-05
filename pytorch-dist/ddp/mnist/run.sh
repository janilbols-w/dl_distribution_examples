#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=$ORION_VGPU mnist-ddp.py --backend nccl
