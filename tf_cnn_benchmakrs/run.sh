#!/bin/bash
log_path="./log"
mkdir $log_path
model="alexnet"
batch_size=32
export ORION_VGPU=4
export ORION_GMEM=30000
export ORION_RATIO=100
for i in 1 2 3 4 5
do
	mv /usr/lib/orion-bak /usr/lib/orion
	horovodrun -np $ORION_VGPU -H localhost:$ORION_VGPU \
            python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
            --model $model \
            --batch_size $batch_size \
            --num_batches 100 \
            --variable_update horovod 2>&1 | tee $log_path/${model}_orion_$i.log
	sleep 20
	mv /usr/lib/orion /usr/lib/orion-bak
 	horovodrun -np $ORION_VGPU -H localhost:$ORION_VGPU \
            python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
            --model $model \
            --batch_size $batch_size \
            --num_batches 100 \
            --variable_update horovod 2>&1 | tee $log_path/${model}_native_$i.log
	sleep 20
done
