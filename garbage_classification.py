import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import opendatasets as od
import tensorflow as tf
from tensorflow import keras
from keras import layers

#od.download("https://www.kaggle.com/datasets/vencerlanz09/plastic-paper-garbage-bag-synthetic-images")

dataset_dir = "plastic-paper-garbage-bag-synthetic-images/Bag Classes/Bag Classes"

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(150, 150),
    batch_size=32,
    validation_split=0.2,
    subset='training', 
    seed=123
)



validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(150, 150),
    batch_size=32,
    validation_split=0.2,
    subset='validation', 
    seed=123
)

def model_maker():
  input = keras.Input(shape=(150, 150, 3))
  x = layers.Conv2D(32, 3, activation="relu", padding="same")(input)
  x = layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = layers.MaxPooling2D(3)(x)
  x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = layers.MaxPooling2D(3)(x)
  x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = layers.MaxPooling2D(3)(x)
  x = layers.Conv2D(32, 3, activation="relu", padding="valid")(x)
  x = layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = layers.MaxPooling2D(3)(x)
  x = layers.Flatten()(x)
  x = layers.Dense(256, activation="relu")(x)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(128, activation="relu")(x)
  x = layers.Dropout(0.5)(x)
  output = layers.Dense(3, activation="softmax")(x)

  model = keras.Model(input, output)
  return model

model = model_maker()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train_dataset, validation_data=validation_dataset, epochs=10)

model.save('garbage_classification.h5')
print("Model saved as 'garbage_classification.h5'")
