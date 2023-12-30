import mappy as mp
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

def mapq2prob(quality_score):
    return 10 ** (-quality_score / 10)

# 2023-10-31 mads

# This baseline model creates a minimap index of the positive training data
# ie. the satellite sequences, and then maps the test sequencens to this.
# The best match score for each prediction is then used for a ROC curve.
# The optimal threshold is calculated from this, and used for the rest of the
# metrics.

np.random.seed(1)

dataset_id = "dataset_v02"

dataset = Path("data/processed/10_datasets") / dataset_id
training_table = Path(dataset) / "train.tsv"
test_table = Path(dataset) / "test.tsv"

individual_train_sequences = Path(dataset) / "train" / "sequences"
individual_test_sequences = Path(dataset) / "test" / "sequences"

iteration_root = Path("data/vizualization/score_distribution/type/mappy_base/")
tax = Path("data/processed/03_taxonomical_annotation/ps_tax_info.tsv")
tax_df = pd.read_csv(tax, sep="\t", header=0, names=["id", "family", "genus", "species"])

outf = Path("data/visualization/score_distribution/type/mappy_base/")
outf.mkdir(parents=True, exist_ok=True)
outfn = outf / "mappy_all_types.tsv"

# Load training test split

df_training = pd.read_csv(training_table, sep="\t", header=0, names=['id', 'type', 'label'])
df_test = pd.read_csv(test_table, sep="\t", header=0, names=['id', 'type', 'label'])

train_ids = df_training.loc[df_training['label'] == 1]['id'].values
test_ids = df_test['id'].values

type_labels = df_test["type"].values.tolist()

with TemporaryDirectory() as tmp:
    # Create fasta for training data
    train_fasta = Path(tmp) / "train.fna"
    with open(train_fasta, "w") as f:
        for id in train_ids:
            with open(individual_train_sequences / f"{id}.fna") as g:
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
        test_id_path = individual_test_sequences / f"{id}.fna"
        max_score = 0
        match_id = None
        for name, seq, qual in mp.fastx_read(str(test_id_path)):
            # align sequence to index
            for hit in a.map(seq): # traverse alignments
                if hit.mapq > max_score:
                    max_score = hit.mapq
                    match_id = hit.ctg
        if match_id is not None:
            label_predict.append(df_training.loc[df_training['id'] == match_id]['label'].values[0])
        else:
            label_predict.append(0)
        max_scores.append(max_score)
        max_labels.append(df_test.loc[df_test['id'] == id]['label'].values[0])

# taxfamily = tax_df["family"].values.tolist()
# tax_count = Counter(taxfamily)

taxspecies = tax_df["species"].values.tolist()
tax_count = Counter(taxspecies)

common_tax = tax_count.most_common(20)
toptax = [x[0] for x in common_tax]

with open(outfn, "w") as fout:
    for sample_n, mapq in enumerate(max_scores):
        # if df_test['label'][sample_n] == 0:
        #     continue
        prob = 1 - mapq2prob(mapq)
        label = df_test['label'][sample_n]

        # Label
        #pred_id = df_test['label'][sample_n]
        
        # Type
        pred_id = type_labels[sample_n]
        if df_test['label'][sample_n] == 0:
            if pred_id.startswith("pro"):
                pred_id = "Prophage"
            elif pred_id.startswith("meta"):
                pred_id = "Metagenomic"
            elif pred_id.startswith("host"):
                pred_id = "Host"
        
        print(pred_id,
              f"{prob:.4f}",
              int(label),
              sep="\t",
              file=fout)


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