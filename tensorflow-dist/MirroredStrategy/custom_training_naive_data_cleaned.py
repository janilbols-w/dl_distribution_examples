# coding: utf-8
# Import TensorFlow
import tensorflow as tf
import numpy as np
import os, pdb, multiprocessing, psutil, ctypes

print(tf.__version__)

strategy = tf.distribute.MirroredStrategy()
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

#BUFFER_SIZE = len(train_images)
BUFFER_SIZE = 6000
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 1

# ## Create the model
# Create a model using `tf.keras.Sequential`. You can also use the Model Subclassing API to do this.
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

print("WUICHAK: define accuracy")
with strategy.scope():
  test_loss = tf.keras.metrics.Mean(name='test_loss')

  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')

# ## Training loop
# model, optimizer, and checkpoint must be created under `strategy.scope`.
print("WUICHAK: create model")
with strategy.scope():
  model = create_model()
  optimizer = tf.keras.optimizers.Adam()

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

print("WUICHAK: ", "threaing id: " , ctypes.CDLL('libc.so.6').syscall(186), "START TRAIN LOOP")
for epoch in range(EPOCHS):
  # TRAIN LOOP
  total_loss = 0.0
  num_batches = 0
  N_BATCHES = 5000
  for i in range(N_BATCHES):
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
