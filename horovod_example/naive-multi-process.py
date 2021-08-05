import tensorflow as tf
import horovod.tensorflow as hvd
hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

a = tf.ones([1000,1000,1,3])*hvd.rank()
for i in range(100000):
  a = tf.add(a,a)
print('rank%d'%hvd.rank(), a.device)
