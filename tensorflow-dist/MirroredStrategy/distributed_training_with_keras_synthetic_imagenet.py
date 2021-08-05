# Import TensorFlow and TensorFlow Datasets

import tensorflow_datasets as tfds
import tensorflow as tf
import time
import os
from tensorflow.keras.optimizers import Adam

start_timestamp = time.time()
print(tf.__version__)

# datasets, info = tfds.load(name='imagenet_a', with_info=True, as_supervised=True)
# imagenet_train = datasets['test']

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# You can also do info.splits.total_num_examples to get the total
# number of examples in the dataset.

# num_train_examples = info.splits['test'].num_examples
# num_test_examples = num_train_examples

EPOCHS = 6
ITERATIONS = 12

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

dataset = tf.data.Dataset.from_tensor_slices((tf.random.uniform([ITERATIONS*BATCH_SIZE, 224, 224, 3]), tf.random.uniform([ITERATIONS*BATCH_SIZE, 1])))
# target = tf.data.Dataset.from_tensor_slices(tf.random.uniform([ITERATIONS*BATCH_SIZE, 1], minval=0, maxval=999, dtype=tf.int64))

# def scale(image, label):
#   image = tf.cast(image, tf.float32)
#   image /= 255

#   return image, label

train_dataset = dataset.batch(BATCH_SIZE)
eval_dataset = dataset.batch(BATCH_SIZE)


with strategy.scope():
  model = tf.keras.applications.ResNet50(weights=None)
  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])


# Define the checkpoint directory to store the checkpoints

checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# Function for decaying the learning rate.
# You can define any decay function you need.
def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 3 and epoch < 7:
    return 1e-4
  else:
    return 1e-5

# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))

class PrintPerformance(tf.keras.callbacks.Callback):
  def on_epoch_begin(self, epoch, logs=None):
    self.epoch_start_time = time.time()
  def on_epoch_end(self, epoch, logs=None):
    epoch_end_time = time.time()
    print('\Training Speed for epoch {} is {} img/sec'.format(epoch + 1,
            ITERATIONS*BATCH_SIZE/(epoch_end_time-self.epoch_start_time)))

            
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR(),
    PrintPerformance()
]

model.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks)

# ======================================================

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

eval_loss, eval_acc = model.evaluate(eval_dataset)

print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))

end_timestampe = time.time()

print("TOTAL TIME USAGE ", end_timestampe - start_timestamp)


print("success")
exit()
