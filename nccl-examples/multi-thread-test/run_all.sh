#!/bin/bash
for i in `seq 7`
do
	mode=$(( i - 1 ))
	./multi_thread_all_api_all_mode_in_one ${ORION_VGPU} ${mode}
	echo done with multi_thread_all_api_all_mode_in_one ${ORION_VGPU} ${mode}
	sleep 5
done
