import os
from math import sqrt

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, accuracy_score


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
        shuffle=False,
        color_mode='grayscale',
        image_size=IMAGE_SHAPE,
        batch_size=BATCH_SIZE
    )

    predictions = model.predict(test_data)
    return tf.argmax(predictions, axis=1)

def draw_graphs():
    return -1

def calculate_mcc(tp, tn, fp, fn):
    try:
        return (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    except Exception as e:
        return 0

def get_best_model(models,predictions,truth):
    return -1

def get_matrix_values(truth, prediction):
    matrix = confusion_matrix(y_true=truth, y_pred=prediction)

    # fp,tp,tn,fn for each class
    tp_rock = matrix[0, 0]
    tp_paper = matrix[1, 1]
    tp_scissors = matrix[2, 2]
    tp_total = tp_rock + tp_paper + tp_scissors

    tn_rock = np.sum(matrix) - (tp_rock + matrix[0, 1] + matrix[0, 2])
    tn_paper = np.sum(matrix) - (tp_paper + matrix[1, 0] + matrix[1, 2])
    tn_scissors = np.sum(matrix) - (tp_scissors + matrix[2, 0] + matrix[2, 1])

    fp_rock = matrix[0, 1] + matrix[0, 2]
    fp_paper = matrix[1, 0] + matrix[1, 2]
    fp_scissors = matrix[2, 0] + matrix[2, 1]

    fn_rock = matrix[1, 0] + matrix[2, 0]
    fn_paper = matrix[0, 1] + matrix[2, 1]
    fn_scissors = matrix[0, 2] + matrix[1, 2]

    # confusion matrix for each class
    confusion_matrix_rock = np.array([[tn_rock, fp_rock], [fn_rock, tp_rock]])
    confusion_matrix_paper = np.array([[tn_paper, fp_paper], [fn_paper, tp_paper]])
    confusion_matrix_scissors = np.array([[tn_scissors, fp_scissors], [fn_scissors, tp_scissors]])

    labels = ['rock', 'paper', 'scissors']

    fig, ax = plt.subplots()

    image = ax.imshow(matrix, cmap='Blues')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, matrix[i][j], ha='center', va='center', color='black')

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title('Confusion Matrix')
    fig.colorbar(image)
    plt.show()

    return matrix, [tp_rock,tp_paper,tp_scissors],[tn_rock,tn_paper,tn_scissors],[fp_rock,fp_paper,fp_scissors],[fn_rock,fn_paper,fn_scissors]

IMAGE_SHAPE = (60,60)
BATCH_SIZE = 32
TEST_DATA_PATH = '../Dataset/testing_otsu'
if __name__ == '__main__':
    num_paper = len([f for f in os.listdir(TEST_DATA_PATH+'/paper') if os.path.isfile(os.path.join(TEST_DATA_PATH+'/paper', f))])
    num_rock = len([f for f in os.listdir(TEST_DATA_PATH + '/rock') if
                     os.path.isfile(os.path.join(TEST_DATA_PATH + '/rock', f))])
    num_scissors = len([f for f in os.listdir(TEST_DATA_PATH + '/scissors') if
                     os.path.isfile(os.path.join(TEST_DATA_PATH + '/scissors', f))])
    truth = np.concatenate([np.full((num_paper),0),np.full((num_rock),1),np.full((num_scissors-1),2)])
    models = get_models('models/otsu_models60x60')
    predictions = []
    mcc_rock = []
    mcc_paper = []
    mcc_scissors = []
    mcc_total = []
    accuracy_total = []
    for model in models:
        prediction = predict_images(model, TEST_DATA_PATH)
        predictions.append(prediction)
        matrix, tp, tn, fp, fn = get_matrix_values(truth, prediction)
        mcc_rock.append(calculate_mcc(tp=tp[0], tn=tn[0], fp=fp[0], fn=fn[0]))
        mcc_paper.append(calculate_mcc(tp=tp[1], tn=tn[1], fp=fp[1], fn=fn[1]))
        mcc_scissors.append(calculate_mcc(tp=tp[2], tn=tn[2], fp=fp[2], fn=fn[2]))
        accuracy_total.append(accuracy_score(y_true=truth, y_pred=prediction))

    for i in range(0, len(mcc_rock)):
        mcc_total.append((mcc_rock[i] + mcc_paper[i] + mcc_scissors[i])/3)

    max_mcc_value = np.amax(mcc_total)
    max_mcc_index = mcc_total.index(max_mcc_value)
    print("the best mcc is: "+str(max_mcc_value)+" on model_"+str(max_mcc_index+1)+".h5")
    max_acc_value = np.amax(accuracy_total)
    max_acc_index = accuracy_total.index(max_acc_value)
    print("the best accuracy is: " + str(max_acc_value) + " on model_" + str(max_acc_index + 1) + ".h5")

#    model = get_best_model(models)
#    matrix_diagram = draw_confusion_matrix(model)