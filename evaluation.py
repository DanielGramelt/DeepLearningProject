import os
from math import sqrt

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support


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


def plotMetrics(truth, pred):
    true_rock = truth.copy()
    true_paper = truth.copy()
    true_scissors = truth.copy()
    pred_rock = pred.copy()
    pred_paper = pred.copy()
    pred_scissors = pred.copy()
    for i in range(len(true_rock)):
        if true_rock[i] != 0:
            true_rock[i] = 1
    for i in range(len(true_paper)):
        if true_paper[i] != 1:
            true_paper[i] = 0
    for i in range(len(true_scissors)):
        if true_scissors[i] != 2:
            true_scissors[i] = 1
    for i in range(len(pred_rock)):
        if pred_rock[i] != 0:
            pred_rock[i] = 1
    for i in range(len(pred_paper)):
        if pred_paper[i] != 1:
            pred_paper[i] = 0
    for i in range(len(pred_scissors)):
        if pred_scissors[i] != 2:
            pred_scissors[i] = 1
    matrix = confusion_matrix(y_true=truth,y_pred=pred)

    #fp,tp,tn,fn for each class
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

    #confusion matrix for each class
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

    accuracy_rock = accuracy_score(y_true=true_rock,y_pred=pred_rock)
    accuracy_paper = accuracy_score(y_true=true_paper,y_pred=pred_paper)
    accuracy_scissors = accuracy_score(y_true=true_scissors,y_pred=pred_scissors)
    accuracy_total = accuracy_score(y_true=truth,y_pred=pred)
    accuracy_array = [accuracy_rock,accuracy_paper,accuracy_scissors]
    print(accuracy_rock)
    print(accuracy_paper)
    print(accuracy_scissors)
    print(accuracy_total)
    precision_array, recall_array, f1_array, _ = precision_recall_fscore_support(y_true=truth,y_pred=pred)
    precision_rock = precision_array[0]
    precision_paper = precision_array[1]
    precision_scissors = precision_array[2]

    recall_rock = recall_array[0]
    recall_paper = recall_array[1]
    recall_scissors = recall_array[2]

    f1_rock = f1_array[0]
    f1_paper = f1_array[1]
    f1_scissors = f1_array[2]

    mcc_rock = calculate_mcc(tp_rock,tn_rock,fp_rock,fn_rock)
    mcc_paper = calculate_mcc(tp_paper,tn_paper,fp_paper,fn_paper)
    mcc_scissors = calculate_mcc(tp_scissors,tn_scissors,fp_scissors,fn_scissors)
    mcc_array = [mcc_rock,mcc_paper,mcc_scissors]

    fig, ax = plt.subplots(figsize=(15,5))
    x = np.arange(3)
    ax.bar(x - 0.15, accuracy_array, 0.15, label='Accuracy',color='#FFFB7A')
    ax.bar(x, precision_array, 0.15, label='Precision',color='#65EBBB')
    ax.bar(x + 0.15, f1_array, 0.15, label='F1 score',color='#E365EB')
    ax.bar(x+0.3, recall_array, 0.15, label='Recall',color='#7398FF')
    ax.bar(x + 0.45, mcc_array, 0.15, label='MCC',color='#FF9966')
    ax.set_xticks(x)
    ax.set_xticklabels(['rock', 'paper', 'scissors'])
    ax.set_ylabel('Score')
    ax.legend()
    plt.show()


    return
def draw_graphs(matrix):
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


def calculate_mcc(tp, tn, fp, fn):
    try:
        return (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    except Exception as e:
        return 0

def get_best_model(mcc_total,accuracy_total,f1_total):
    max_mcc_value = np.amax(mcc_total)
    max_mcc_index = mcc_total.index(max_mcc_value)
    print("the best mcc is: " + str(max_mcc_value) + " on model_" + str(max_mcc_index + 1) + ".h5")
    max_acc_value = np.amax(accuracy_total)
    max_acc_index = accuracy_total.index(max_acc_value)
    print("the best accuracy is: " + str(max_acc_value) + " on model_" + str(max_acc_index + 1) + ".h5")
    max_f1_value = np.amax(f1_total)
    max_f1_index = f1_total.index(max_f1_value)
    print("the best f1 is: " + str(max_f1_value) + " on model_" + str(max_f1_index + 1) + ".h5")
    return max_mcc_index

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
    f1_rock = []
    f1_paper = []
    f1_scissors = []
    f1_total = []
    precision_rock = []
    precision_paper = []
    precision_scissors = []
    recall_rock = []
    recall_paper = []
    recall_scissors = []
    accuracy_rock = []
    accuracy_paper = []
    accuracy_scissors = []
    accuracy_total = []
    for model in models:
        prediction = predict_images(model, TEST_DATA_PATH)
        predictions.append(prediction)
        matrix, tp, tn, fp, fn = get_matrix_values(truth, prediction)
        mcc_rock.append(calculate_mcc(tp=tp[0], tn=tn[0], fp=fp[0], fn=fn[0]))
        mcc_paper.append(calculate_mcc(tp=tp[1], tn=tn[1], fp=fp[1], fn=fn[1]))
        mcc_scissors.append(calculate_mcc(tp=tp[2], tn=tn[2], fp=fp[2], fn=fn[2]))
        accuracy_total.append(accuracy_score(y_true=truth, y_pred=prediction))
        precision_array, recall_array, f1_array, _ = precision_recall_fscore_support(y_true=truth, y_pred=prediction)
        precision_rock.append(precision_array[0])
        precision_paper.append(precision_array[1])
        precision_scissors.append(precision_array[2])

        recall_rock.append(recall_array[0])
        recall_paper.append(recall_array[1])
        recall_scissors.append(recall_array[2])

        f1_rock.append(f1_array[0])
        f1_paper.append(f1_array[1])
        f1_scissors.append(f1_array[2])

    for i in range(0, len(mcc_rock)):
        mcc_total.append((mcc_rock[i] + mcc_paper[i] + mcc_scissors[i])/3)
        f1_total.append((f1_rock[i] + f1_paper[i] + f1_scissors[i])/3)

    best_index = get_best_model(mcc_total, accuracy_total, f1_total)
    plotMetrics(truth, predictions[best_index].numpy())
