import os
import random
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import keras
import numpy as np
from PIL import Image

def load_images(train_path, val_path, test_path):
    train_paths = [train_path+'/rock',train_path+'/paper',train_path+'/scissors']
    val_paths = [val_path+'/rock',val_path+'/paper',val_path+'/scissors']
    test_paths = [test_path+'/rock',test_path+'/paper',test_path+'/scissors']
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []
    for i, path in enumerate(train_paths):
        for file in os.listdir(path):
            if file.endswith(".png"):
                img = Image.open(os.path.join(path, file)).convert('L') # converts image to grayscale
                img = img.resize(IMAGE_SHAPE) # resize image
                X_train.append(np.asarray(img))
                y_train.append(i)
    data = list(zip(X_train, y_train))
    random.shuffle(data)
    X_train, y_train = zip(*data)
    print("loaded rock data")
    for i, path in enumerate(val_paths):
        for file in os.listdir(path):
            if file.endswith(".png"):
                img = Image.open(os.path.join(path, file)).convert('L') # converts image to grayscale
                img = img.resize(IMAGE_SHAPE) # resize image
                X_val.append(np.asarray(img))
                y_val.append(i)
    data = list(zip(X_val, y_val))
    random.shuffle(data)
    X_val, y_val = zip(*data)
    print("loaded paper data")
    for i, path in enumerate(test_paths):
        for file in os.listdir(path):
            if file.endswith(".png"):
                img = Image.open(os.path.join(path, file)).convert('L') # converts image to grayscale
                img = img.resize(IMAGE_SHAPE) # resize image
                X_test.append(np.asarray(img))
                y_test.append(i)
    data = list(zip(X_test, y_test))
    random.shuffle(data)
    X_test, y_test = zip(*data)
    print("loaded scissors data")
    return np.asarray(X_train), np.asarray(y_train), np.asarray(X_val), np.asarray(y_val), np.asarray(X_test), np.asarray(y_test)


def train_model(X_train, y_train, X_val, y_val):
    X_train = tf.expand_dims(X_train, -1)
    X_val = tf.expand_dims(X_val, -1)

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(60,
                                           60,
                                           1)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )
    num_classes = 3

    model = Sequential([
        layers.Reshape((60, 60, 1)),
        data_augmentation,
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, name="outputs")
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )


    version = 1
    while True:
        model_file = "model_{}.h5".format(version)
        if not os.path.exists(model_file):
            break
        version += 1

    model.save(model_file)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

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

SEED = 42
EPOCHS = 5
IMAGE_SHAPE = (60,60)
random.seed(SEED)
np.random.seed(SEED)
X_train, y_train, X_val, y_val, X_test, y_test = load_images('../Dataset/multi_otsu','../Dataset/validation_otsu','../Dataset/testing_otsu')
print(f"X_train.shape: {X_train.shape} - y_train.shape: {y_train.shape} - class dist: {np.bincount(y_train)}\n" +
      f"X_val.shape: {X_val.shape} - y_val.shape: {y_val.shape} - class dist: {np.bincount(y_val)}\n" +
      f"X_test.shape: {X_test.shape} - y_test.shape: {y_test.shape} - class dist: {np.bincount(y_test)}")
train_model(X_train,y_train,X_val,y_val)

