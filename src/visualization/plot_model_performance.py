import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import average_precision_score
from matplotlib.font_manager import FontProperties
import numpy as np
import csv

matplotlib.rcParams['font.family'] = 'Helvetica Neue'

def read_predictions(file_path):
    y_true, y_pred, y_prob = [], [], []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            y_true.append(int(row[0]))
            y_pred.append(int(row[1]))
            y_prob.append(float(row[2]))
    return y_true, y_pred, y_prob

def plot_confusion_matrix(y_true, y_pred, ax, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, values_format='d')
    ax.set_title(f'Confusion Matrix: {model_name}')

def plot_curves(models):
    num_models = len(models)
    plt.figure(figsize=(18, 6 + 4 * num_models))

    # ROC and AUC
    plt.subplot(1, 2, 1)
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
    plt.subplot(1, 2, 2)
    for model_name, file_path in models.items():
        y_true, _, y_prob = read_predictions(file_path)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(models):
    num_models = len(models)
    plt.figure(figsize=(12, 4 * num_models))

    max_val = 0
    for _, file_path in models.items():
        y_true, y_pred, _ = read_predictions(file_path)
        cm = confusion_matrix(y_true, y_pred)
        max_val = max(max_val, cm.max())

    for i, (model_name, file_path) in enumerate(models.items()):
        y_true, y_pred, _ = read_predictions(file_path)
        cm = confusion_matrix(y_true, y_pred)
        ax = plt.subplot(1, 3, i + 1)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(ax=ax, values_format='d', cmap='Blues', colorbar=False)
        ax.set_title(f'{model_name}', fontsize=20)
        ax.set_xlabel('Predicted label', fontsize=20)
        ax.set_ylabel('True label', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        for text in disp.text_.ravel():
            text.set_fontsize(20)
        im = disp.im_
        im.set_clim(0, max_val)

    plt.tight_layout()
    plt.savefig('reports/figures/confusion.pdf')
    #plt.show()

if __name__ == "__main__":

    # models = {
    #     'Baseline': 'data/visualization/performance/mappy_performance_v02.tsv',
    #     'CNN-model': 'data/visualization/performance/inception_performance_v02.tsv'
    # }

    models = {
        'Satellite pfam-based': 'data/visualization/performance/v02/validation/pfama_single_v02_small_8dhbpn5q_performance.tsv',
        'Satellite protein': 'data/visualization/performance/v02/validation/mmdb_single_v02_small_ofkn1eok_performance.tsv',
        'General protein profile ': 'data/visualization/performance/v02/validation/alldb_v02_small_iak7l6eg_performance.tsv'
    }

    #plot_curves(models)
    plot_confusion_matrices(models)
    