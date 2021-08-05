
# coding: utf-8

# ##### Copyright 2019 The TensorFlow Authors.

# In[1]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# # Custom training with tf.distribute.Strategy

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/distribute/custom_training"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/distribute/custom_training.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/distribute/custom_training.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/distribute/custom_training.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# This tutorial demonstrates how to use [`tf.distribute.Strategy`](https://www.tensorflow.org/guide/distributed_training) with custom training loops. We will train a simple CNN model on the fashion MNIST dataset. The fashion MNIST dataset contains 60000 train images of size 28 x 28 and 10000 test images of size 28 x 28.
# 
# We are using custom training loops to train our model because they give us flexibility and a greater control on training. Moreover, it is easier to debug the model and the training loop.

# In[2]:


# Import TensorFlow
import tensorflow as tf

# Helper libraries
import numpy as np
import os
import pdb

import multiprocessing
import psutil
import ctypes

print(tf.__version__)
#tf.executing_eagerly()


# ## Download the fashion MNIST dataset

# In[3]:


#fashion_mnist = tf.keras.datasets.fashion_mnist

#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Adding a dimension to the array -> new shape == (28, 28, 1)
# We are doing this because the first layer in our model is a convolutional
# layer and it requires a 4D input (batch_size, height, width, channels).
# batch_size dimension will be added later on.
#train_images = train_images[..., None]
#test_images = test_images[..., None]

# Getting the images in [0, 1] range.
#train_images = train_images / np.float32(255)
#test_images = test_images / np.float32(255)


# ## Create a strategy to distribute the variables and the graph

# How does `tf.distribute.MirroredStrategy` strategy work?
# 
# *   All the variables and the model graph is replicated on the replicas.
# *   Input is evenly distributed across the replicas.
# *   Each replica calculates the loss and gradients for the input it received.
# *   The gradients are synced across all the replicas by summing them.
# *   After the sync, the same update is made to the copies of the variables on each replica.
# 
# Note: You can put all the code below inside a single scope. We are dividing it into several code cells for illustration purposes.
# 

# In[4]:


# If the list of devices is not specified in the
# `tf.distribute.MirroredStrategy` constructor, it will be auto-detected.
strategy = tf.distribute.MirroredStrategy()


# In[5]:


print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))


# ## Setup input pipeline

# Export the graph and the variables to the platform-agnostic SavedModel format. After your model is saved, you can load it with or without the scope.

# In[6]:


#BUFFER_SIZE = len(train_images)
BUFFER_SIZE = 6000

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 1


# Create the datasets and distribute them:

# In[7]:


#train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
#test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE) 

#train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
#test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


# ## Create the model
# 
# Create a model using `tf.keras.Sequential`. You can also use the Model Subclassing API to do this.

# In[8]:

print("WUICHAK: define create model")
def create_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
    ])

  return model


# In[9]:


# Create a checkpoint directory to store the checkpoints.
#checkpoint_dir = './training_checkpoints'
#checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


# ## Define the loss function
# 
# Normally, on a single machine with 1 GPU/CPU, loss is divided by the number of examples in the batch of input.
# 
# *So, how should the loss be calculated when using a `tf.distribute.Strategy`?*
# 
# * For an example, let's say you have 4 GPU's and a batch size of 64. One batch of input is distributed
# across the replicas (4 GPUs), each replica getting an input of size 16.
# 
# * The model on each replica does a forward pass with its respective input and calculates the loss. Now, instead of dividing the loss by the number of examples in its respective input (BATCH_SIZE_PER_REPLICA = 16), the loss should be divided by the GLOBAL_BATCH_SIZE (64).
# 
# *Why do this?*
# 
# * This needs to be done because after the gradients are calculated on each replica, they are synced across the replicas by **summing** them.
# 
# *How to do this in TensorFlow?*
# * If you're writing a custom training loop, as in this tutorial, you should sum the per example losses and divide the sum by the GLOBAL_BATCH_SIZE: 
# `scale_loss = tf.reduce_sum(loss) * (1. / GLOBAL_BATCH_SIZE)`
# or you can use `tf.nn.compute_average_loss` which takes the per example loss,
# optional sample weights, and GLOBAL_BATCH_SIZE as arguments and returns the scaled loss.
# 
# * If you are using regularization losses in your model then you need to scale
# the loss value by number of replicas. You can do this by using the `tf.nn.scale_regularization_loss` function.
# 
# * Using `tf.reduce_mean` is not recommended. Doing so divides the loss by actual per replica batch size which may vary step to step.
# 
# * This reduction and scaling is done automatically in keras `model.compile` and `model.fit`
# 
# * If using `tf.keras.losses` classes (as in the example below), the loss reduction needs to be explicitly specified to be one of `NONE` or `SUM`. `AUTO` and `SUM_OVER_BATCH_SIZE`  are disallowed when used with `tf.distribute.Strategy`. `AUTO` is disallowed because the user should explicitly think about what reduction they want to make sure it is correct in the distributed case. `SUM_OVER_BATCH_SIZE` is disallowed because currently it would only divide by per replica batch size, and leave the dividing by number of replicas to the user, which might be easy to miss. So instead we ask the user do the reduction themselves explicitly.
# * If `labels` is multi-dimensional, then average the `per_example_loss` across the number of elements in each sample. For example, if the shape of `predictions` is `(batch_size, H, W, n_classes)` and `labels` is `(batch_size, H, W)`, you will need to update `per_example_loss` like: `per_example_loss /= tf.cast(tf.reduce_prod(tf.shape(labels)[1:]), tf.float32)`
# 

# In[10]:

print("WUICHAK: define loss")
with strategy.scope():
  # Set reduction to `none` so we can do the reduction afterwards and divide by
  # global batch size.
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True,
      reduction=tf.keras.losses.Reduction.NONE)
  def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)


# ## Define the metrics to track loss and accuracy
# 
# These metrics track the test loss and training and test accuracy. You can use `.result()` to get the accumulated statistics at any time.

# In[11]:

print("WUICHAK: define accuracy")
with strategy.scope():
  test_loss = tf.keras.metrics.Mean(name='test_loss')

  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')

# ## Training loop

# In[12]:


# model, optimizer, and checkpoint must be created under `strategy.scope`.
print("WUICHAK: create model")
with strategy.scope():
  model = create_model()

  optimizer = tf.keras.optimizers.Adam()

#  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)


# In[13]:


def train_step(inputs):
  images, labels = inputs
  this_thread = multiprocessing.current_process()
  print('WUICHAK: worker %s: PID=%s' % (this_thread.name, this_thread.pid), "threaing id: " , ctypes.CDLL('libc.so.6').syscall(186))
  tp = psutil.Process(this_thread.pid)
  print('WUICHAK: worker %s: memory_info=%s' % (this_thread.name, tp.memory_info().rss/1024/1024), 
          "threaing id: " , ctypes.CDLL('libc.so.6').syscall(186))
  #print(os.system( "pstree -p " + str(os.getpid())))
  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = compute_loss(labels, predictions)

  print("WUICHAK: ", "threaing id: ", ctypes.CDLL('libc.so.6').syscall(186) ,"train_step tape.gradient")
  gradients = tape.gradient(loss, model.trainable_variables)
  print("WUICHAK: ", "threaing id: " , ctypes.CDLL('libc.so.6').syscall(186) , "apply_gradients")
  optimizer.apply_gradients(zip(gradients, model.trainable_variables),experimental_aggregate_gradients=True)
  print("WUICHAK: ", "threaing id: " , ctypes.CDLL('libc.so.6').syscall(186) ,"update state")
  train_accuracy.update_state(labels, predictions)
  print("WUICHAK: ", "threaing id: " , ctypes.CDLL('libc.so.6').syscall(186) ,"train_step loss - ", loss)
  return loss 

def test_step(inputs):
  pass
'''  
  images, labels = inputs

  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss.update_state(t_loss)
  test_accuracy.update_state(labels, predictions)
'''

# In[14]:


# `run` replicates the provided computation and runs it
# with the distributed input.
#@tf.function
def distributed_train_step(dataset_inputs,num_batch):
  print("WUICHAK: ", "threaing id: " , ctypes.CDLL('libc.so.6').syscall(186), "start strategy.run - batch%d"%num_batch)
  #pdb.set_trace()
  per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
  print("WUICHAK: ", "threaing id: " , ctypes.CDLL('libc.so.6').syscall(186), "end strategy.run - batch%d"%num_batch)
  #pdb.set_trace()
  print("WUICHAK: ", "threaing id: " , ctypes.CDLL('libc.so.6').syscall(186), "start strategy.reduce - batch%d"%num_batch)
  rt = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)
  print("WUICHAK: ", "threaing id: " , ctypes.CDLL('libc.so.6').syscall(186), "end strategy.reduce - batch%d"%num_batch)
  print("WUICHAK: ", "threaing id: " , ctypes.CDLL('libc.so.6').syscall(186), "distributed_train_step return",rt)
  return rt

#@tf.function
def distributed_test_step(dataset_inputs):
  pass
  #return strategy.run(test_step, args=(dataset_inputs,))

print("WUICHAK: ", "threaing id: " , ctypes.CDLL('libc.so.6').syscall(186), "START TRAIN LOOP")
for epoch in range(EPOCHS):
  # TRAIN LOOP
  total_loss = 0.0
  num_batches = 0
  TEST_N_BATCHES = 5000
  for i in range(TEST_N_BATCHES):
    print("WUICHAK: ======================================")
    print("WUICHAK: ", "threaing id: " , ctypes.CDLL('libc.so.6').syscall(186), "const input")
    x = (tf.zeros([64,28,28,1],dtype=float),tf.ones([64,],dtype=float))
    print("WUICHAK: ", "threaing id: " , ctypes.CDLL('libc.so.6').syscall(186), "start dist_tran_step - batch%d"%(num_batches))
    total_loss += distributed_train_step(x,num_batches)
    print("WUICHAK: ", "threaing id: " , ctypes.CDLL('libc.so.6').syscall(186), "end dist_tran_step - batch%d"%(num_batches))
    num_batches += 1
    if num_batches == 2:
        break
  train_loss = total_loss / num_batches


'''
  # TEST LOOP
  for x in test_dist_dataset:
    pass
    #distributed_test_step(x)

  if epoch % 2 == 0:
    checkpoint.save(checkpoint_prefix)

  template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
              "Test Accuracy: {}")
  print (template.format(epoch+1, train_loss,
                         train_accuracy.result()*100, test_loss.result(),
                         test_accuracy.result()*100))

  #rest_loss.reset_states()
  #train_accuracy.reset_states()
  #rest_accuracy.reset_states()
'''
