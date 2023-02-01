import os
import tensorflow as tf
from tensorflow import keras

def get_models(model_path):
    models = []
    for filename in os.listdir(model_path):
        if filename.endswith(".h5"):
            model = keras.models.load_model(os.path.join(model_path, filename))
            models.append(model)
    return models


def predict_images(model, image_path):
    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        image_path,
        validation_split=None,
        subset=None,
        color_mode='grayscale',
        image_size=IMAGE_SHAPE,
        batch_size=BATCH_SIZE
    )

    predictions = model.predict(test_data)
    return tf.argmax(predictions, axis=1)



IMAGE_SHAPE = (60,60)
BATCH_SIZE = 32
