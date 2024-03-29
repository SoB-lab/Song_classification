import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


batch_size = 16
img_height = 480
img_width = 620

data_dir_test = 'picture_database_seperated/test/'

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_test,
  image_size=(img_height, img_width),
  color_mode='rgb',
  batch_size=batch_size)

model = load_model('my_model_2')

predictions = np.array([])
Y_pred = np.array([])
labels =  np.array([])
Y_true = np.array([])
for x, y in test_ds:
    predictions = model.predict(x,batch_size=16)
    y_pred = np.argmax(predictions, axis=1)
    Y_pred = np.append(y_pred, Y_pred)
    Y_true = np.append(y, Y_true)

print(tf.math.confusion_matrix(labels=Y_true, predictions=Y_pred).numpy())

results = model.evaluate(test_ds)
print(results)

loss, acc = model.evaluate(test_ds, verbose=2)
print(loss, acc)