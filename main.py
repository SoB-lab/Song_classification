'''Created with python 3.8 or above, with tensorflow 2.4
Run CNN model for music genre classification for 3 different classes of 3000 samples each
Fine-tuned for the following model of two convolutionnal layers, two Maxpooling and two densification before linearization
'''
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


batch_size = 16


def CNN_model2(model, num_classes):
    model.add(layers.Conv2D(32, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.SpatialDropout2D(0.4))
    model.add(layers.Conv2D(64, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.SpatialDropout2D(0.4))
    '''model.add(layers.Conv2D(128, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.SpatialDropout2D(0.5))
    model.add(layers.Conv2D(64, kernel_size=(4,4), strides=(2,2), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))'''
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.6))
    #model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

print("---------------------------------------------------------------------")
print("Create datasets")
print("---------------------------------------------------------------------")

im_width = 480
im_height = 620

data_dir_train = 'picture_database_seperated/train/'
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

print(type(train_ds), type(val_ds))

num_classes = 3

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("---------------------------------------------------------------------")
print("creation of the model")
print("---------------------------------------------------------------------")



#Removal of the data augmentation because of the image artefact that are created and modify the genre of the songs
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1. / 255),
    #layers.experimental.preprocessing.RandomZoom(height_factor=(0.2, 0.3)),
    #layers.experimental.preprocessing.RandomTranslation(height_factor=0.0, width_factor=0.2),
    #layers.experimental.preprocessing.RandomFlip(mode="horizontal")
])

model = Sequential(data_augmentation)
model = CNN_model2(model, num_classes)


opti = tf.keras.optimizers.RMSprop(
    learning_rate=0.0005
)


model.compile(optimizer=opti,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



# Following line only use during tuning to early stop the fitting in case of over/underfitting
#earlystop_callback = EarlyStopping(monitor = 'val_accuracy',
                                   #min_delta = 0.001,
                                   #patience=8)
print("---------------------------------------------------------------------")
print("run the model")
print("---------------------------------------------------------------------")


epochs = 20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  #callbacks=[earlystop_callback]
)

model.summary()


#Save the model for potential reuse or testing
model.save('my_model')

#Print the summary of each trainable and non trainable parameters of each layers
model.summary()

print("---------------------------------------------------------------------")
print("Return metrics")
print("---------------------------------------------------------------------")


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

import pandas
#Save the metrics from history
pandas.DataFrame(history.history).to_csv("history.csv")


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


