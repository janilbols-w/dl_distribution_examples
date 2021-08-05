#!/bin/bash
DATA_PATH=${1:-'/root/ImageNet_ILSVRC2012_reduced'}
python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ${DATA_PATH}
