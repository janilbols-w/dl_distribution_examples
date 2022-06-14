# =======================================================
#
# Date:                 2022 JUNE 10th
# Author:               wanghuize@virtaitech.com
# TF version:           tensorflow-2.6.2
#
# ===================== Quickstart ======================
#
# To start each worker, run with cmd as below:
#       TF_WORKER_ID=<WID> python main.py
#
# =======================================================
#
# To run more workers, please modify TF_CONFIG
#
# =======================================================

import os
import json
import tensorflow as tf
import numpy as np

def mnist_dataset(batch_size):
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  # The `x` arrays are in uint8 and have values in the [0, 255] range.
  # You need to convert them to float32 with values in the [0, 1] range.
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
  return train_dataset

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model


# configure TF env
per_worker_batch_size = 64
TF_WORKER_ID=int(os.environ['TF_WORKER_ID'])
tf_config = {
    'cluster': {
        'worker': ['localhost:33456', 'localhost:43456']
    },
    'task': {'type': 'worker', 'index': TF_WORKER_ID}
}
os.environ['TF_CONFIG'] = json.dumps(tf_config)

num_workers = len(tf_config['cluster']['worker'])


# Init

# Change to tf.distribute.experimental.MultiWorkerMirroredStrategy()
# when you are running with a lower version of tensorflow
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# dataset
global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_dataset(global_batch_size)

# build model
with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = build_and_compile_cnn_model()

# train model
multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)
