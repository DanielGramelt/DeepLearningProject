import os
import random
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import multiprocessing
import keras
import numpy as np
from PIL import Image

def load_images(train_path, val_path, seed):

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        validation_split=None,
        subset=None,
        seed=seed,
        shuffle=True,
        color_mode='grayscale',
        image_size=IMAGE_SHAPE,
        batch_size=BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_path,
        validation_split=None,
        subset=None,
        seed=seed,
        shuffle=True,
        color_mode='grayscale',
        image_size=IMAGE_SHAPE,
        batch_size=BATCH_SIZE)

    class_names = train_ds.class_names
    print(class_names)
    return train_ds, val_ds


def train_model(train_ds, val_ds):

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(HEIGHT,
                                           WIDTH,
                                           1)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )
    num_classes = 3

    model = Sequential([
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
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )


    version = 1
    while True:
        model_file = "models/model_{}.h5".format(version)
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


BATCH_SIZE = 32
EPOCHS = 15
HEIGHT = 60
WIDTH = 60
IMAGE_SHAPE = (HEIGHT, WIDTH)

def train_and_save_model(seed):

    random.seed(seed)
    np.random.seed(seed)
    train, val = load_images('../Dataset/multi_otsu', '../Dataset/validation_otsu', seed)
    train_model(train, val)

if __name__ == '__main__':
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.map(train_and_save_model, range(123, 123 * 11, 123))