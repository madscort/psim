import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def read_scores_from_file(pos_file, neg_file):
    # Read scores from positive file
    with open(pos_file, 'r') as f:
        positive_results = []
        for line in f:
            res = [float(res) for res in line.strip().split("\t")]
            positive_results.append(res)
    

    # Read scores from negative file
    with open(neg_file, 'r') as f:
        negative_results = []
        for line in f:
            res = [float(res) for res in line.strip().split("\t")]
            negative_results.append(res)

    max_positive = [max(res) for res in positive_results]
    max_negative = [max(res) for res in negative_results]
    max_scores = max_positive + max_negative
    max_labels = [1]*len(max_positive) + [0]*len(max_negative)

    return max_scores, max_labels

def plot_roc_curve(scores, labels, label):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

# File paths for your sets (you can extend this list)
sets = [
    {"pos": "models/hmm_model/phage_10/prediction/positive/phage_10_pos_hmmsearch_result.tsv", "neg": "models/hmm_model/phage_10/prediction/negative/phage_10_neg_hmmsearch_result.tsv", "label": "0.1"},
    {"pos": "models/hmm_model/phage_25/prediction/positive/phage_25_pos_hmmsearch_result.tsv", "neg": "models/hmm_model/phage_25/prediction/negative/phage_25_neg_hmmsearch_result.tsv", "label": "0.25"},
    {"pos": "models/hmm_model/phage_50/prediction/positive/phage_50_pos_hmmsearch_result.tsv", "neg": "models/hmm_model/phage_50/prediction/negative/phage_50_neg_hmmsearch_result.tsv", "label": "0.5"},
    {"pos": "models/hmm_model/phage_90/prediction/positive/phage_90_pos_hmmsearch_result.tsv", "neg": "models/hmm_model/phage_90/prediction/negative/phage_90_neg_hmmsearch_result.tsv", "label": "0.9"},
]

if __name__ == "__main__":
    plt.figure(figsize=(10, 8))

    for s in sets:
        scores, labels = read_scores_from_file(s["pos"], s["neg"])
        plot_roc_curve(scores, labels, s["label"])

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(title='Phage fraction of negative test-data',loc="lower right")
    plt.show()
