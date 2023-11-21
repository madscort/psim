import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, matthews_corrcoef
from sklearn.metrics import average_precision_score
import numpy as np
import csv

def read_predictions(file_path):
    y_true, y_pred, y_prob = [], [], []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            y_true.append(int(row[0]))
            y_pred.append(int(row[1]))
            y_prob.append(float(row[2]))
    return y_true, y_pred, y_prob

def calculate_mcc_f1(y_true, y_pred):
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return mcc, f1

def plot_curves(models):
    plt.figure(figsize=(18, 6))

    # ROC and AUC
    plt.subplot(1, 3, 1)
    for model_name, file_path in models.items():
        y_true, y_pred, y_prob = read_predictions(file_path)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Precision-Recall
    plt.subplot(1, 3, 2)
    for model_name, file_path in models.items():
        y_true, _, y_prob = read_predictions(file_path)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    # MCC-F1
    plt.subplot(1, 3, 3)
    mcc_values, f1_values = [], []
    for model_name, file_path in models.items():
        y_true, y_pred, _ = read_predictions(file_path)
        mcc, f1 = calculate_mcc_f1(y_true, y_pred)
        mcc_values.append(mcc)
        f1_values.append(f1)
    plt.scatter(mcc_values, f1_values)
    for i, model_name in enumerate(models.keys()):
        plt.annotate(model_name, (mcc_values[i], f1_values[i]))
    plt.xlabel('MCC')
    plt.ylabel('F1 Score')
    plt.title('MCC vs F1 Score')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    models = {
        'Baseline': 'data/visualization/performance/mappy_performance.tsv',
        'CNN_model': 'data/visualization/performance/inception_performance.tsv',
        'Transformer_model': 'data/visualization/performance/small_transformer_performance.tsv'
    }

    plot_curves(models)