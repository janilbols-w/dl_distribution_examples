# README
This tensorflow MultiWorkerMirroredStrategy example refers to official tensorflow distribution tutorials: 
https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras

For more information, you could go to the official site.

## Overview

This example allows you to run '2' workers with MultiWorkerMirroredStrategy, to train a simple network on mnist.

./
├── README.md
├── main.py
├── mnist_setup.py
└── run.sh


- main.py : main entry for all workers  
for each workers, simply run the command below
```bash
TF_WORKER_ID=<WID> python main.py
```

- mnist_setup.py  
this is a python script to define mnist models

- run.sh  
this file allows you to run a quickstart within one terminal

## Quickstart
```bash
# default settings will running a 2+2gpus tf.dist.MultiWorkerMirroredStrategy test on mnist
# log for worker 0 will NOT be display, but store in worker_0.log
./run.sh
```
