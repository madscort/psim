import mappy as mp
from pathlib import Path
from tempfile import TemporaryDirectory
from dotenv import find_dotenv, load_dotenv
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
import seaborn as sns

# fasta_glob = Path("data/processed/10_datasets/phage_25_fixed_25000_reduced_90_ws/satellite_sequences/").glob("*.fna")
# a = mp.Aligner(["data/processed/10_datasets/phage_25_fixed_25000_reduced_90_ws/.tmp/sat_contigs.fna"])  # load or build index

np.random.seed(1)

dataset_id = "prophage_95_fixed_25000_ps_minimal_90_ws"
version_id = "prophage_95_fixed_25000_ps_minimal_90_ws"

dataset = Path("data/processed/10_datasets") / dataset_id
sampletable = Path(dataset) / "sampletable.tsv"
all_sequences = Path(dataset) / "dataset.fna"
individual_sequences = Path(dataset) / "sequences"

validation_root = Path("models/mappy_model")
validation_root.mkdir(parents=True, exist_ok=True)
iteration_root = Path(validation_root, version_id)
iteration_root.mkdir(parents=True, exist_ok=True)

# Split positive and negetative sets separately
df_sampletable = pd.read_csv(sampletable, sep="\t", header=None, names=['id', 'type', 'label'])

df_neg = df_sampletable.loc[df_sampletable['label'] == 0]
df_pos = df_sampletable.loc[df_sampletable['label'] == 1]

train_neg, test_neg = train_test_split(df_neg, stratify=df_neg['type'], test_size=0.2)
train_pos, test_pos = train_test_split(df_pos, stratify=df_pos['type'], test_size=0.2)

#train_ids = np.concatenate([train_neg['id'].values, train_pos['id'].values])
# Only positive test set
train_ids = train_pos['id'].values

test_ids = np.concatenate([test_neg['id'].values, test_pos['id'].values])
# # Only positive test set
# test_ids = test_pos['id'].values

with TemporaryDirectory() as tmp:
    # Create fasta for training data
    train_fasta = Path(tmp) / "train.fna"
    with open(train_fasta, "w") as f:
        for id in train_ids:
            with open(individual_sequences / f"{id}.fna") as g:
                f.write(g.read())
    
    # Load as index:
    a = mp.Aligner(str(train_fasta))  # load or build index
    if not a:
        raise Exception("ERROR: failed to load/build index")
    
    # Align each test sequence to the index
    max_scores = []
    max_labels = []
    label_predict = []
    for id in test_ids:
        test_id_path = individual_sequences / f"{id}.fna"
        max_score = 0
        match_id = None
        for name, seq, qual in mp.fastx_read(str(test_id_path)):
            # align sequence to index
            for hit in a.map(seq): # traverse alignments
                if hit.mapq > max_score:
                    max_score = hit.mapq
                    match_id = hit.ctg
        if match_id is not None:
            label_predict.append(df_sampletable.loc[df_sampletable['id'] == match_id]['label'].values[0])
        else:
            label_predict.append(0)
        max_scores.append(max_score)
        max_labels.append(df_sampletable.loc[df_sampletable['id'] == id]['label'].values[0])

# Get roc curve with scikit learn

fpr, tpr, thresholds = roc_curve(max_labels, max_scores)
roc_auc = auc(fpr, tpr)

# Plot using matplotlib

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
Path(iteration_root, "result_metrics", "roc_curve.png").parent.mkdir(parents=True, exist_ok=True)
plt.savefig(Path(iteration_root, "result_metrics", "roc_curve.png"))
plt.close()
# Get precision recall curve with scikit learn

precision, recall, thresholds = precision_recall_curve(max_labels, max_scores)
pr_auc = auc(recall, precision)

# Plot using matplotlib
plt.figure()
plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.legend(loc="lower right")
Path(iteration_root, "result_metrics", "pr_curve.png").parent.mkdir(parents=True, exist_ok=True)
plt.savefig(Path(iteration_root, "result_metrics", "pr_curve.png"))
plt.close

# Get precision recall and f1 score with scikit learn
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold}")

precision, recall, f1, _ = precision_recall_fscore_support(max_labels, [1 if score > optimal_threshold else 0 for score in max_scores], average='binary')

# Create confusion matrix
label_predict = [1 if score > optimal_threshold else 0 for score in max_scores]
conf_matrix = confusion_matrix(max_labels, label_predict)

# Plot using seaborn
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap="Blues", cbar=False,
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Save the figure
conf_matrix_path = Path(iteration_root, "result_metrics", "conf_matrix.png")
plt.savefig(conf_matrix_path)
plt.close()

print("Threshold based:")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")

precision, recall, f1, _ = precision_recall_fscore_support(max_labels, label_predict, average='binary')
print("Zero default:")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")

# Save results to file
with open(Path(iteration_root, "result_metrics", "metrics.tsv"), "w") as f:
    print(f"Optimal threshold: {optimal_threshold}", file = f)
    print(f"Precision: {precision}", file = f)
    print(f"Recall: {recall}", file = f)
    print(f"F1: {f1}", file = f)