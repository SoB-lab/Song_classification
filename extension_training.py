import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


model = load_model('my_model_2')

data_dir_train = 'picture_database_seperated/train/'

batch_size = 16
im_width = 480
im_height = 620

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_train,
    validation_split=0.2,
    subset="training",
    seed=123,
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(im_width, im_height)
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_train,
    validation_split=0.2,
    subset="validation",
    seed=123,
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(im_width, im_height)
)


epochs = 20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs)

model.save('my_model_2')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

import pandas
pandas.DataFrame(history.history).to_csv("history5.csv")

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()