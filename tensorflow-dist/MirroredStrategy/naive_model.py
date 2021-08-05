from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf
import json


gpus = tf.config.experimental.list_physical_devices('GPU')

batch_size = 100

strategy = tf.distribute.MirroredStrategy()

data = load_iris()
features = data.data
labels = data.target

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

with strategy.scope():

    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer="adam", metrics=["accuracy"])


model.fit(x_train, y_train, batch_size=batch_size, epochs=5000, verbose=2)
result = model.evaluate(x_test, y_test, verbose=2)
