import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, recall_score
def bootstrap_confidence_interval(y_true, y_pred, y_pred_prob=None, metric_func=None, n_bootstraps=1000, ci=95, rng_seed=1):
    rng = np.random.RandomState(rng_seed)
    bootstrapped_scores = []

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue

        if metric_func == roc_auc_score:
            if y_pred_prob is not None:
                score = metric_func(y_true[indices], y_pred_prob[indices])
            else:
                raise ValueError("y_pred_prob is required for ROC AUC score")
        else:
            score = metric_func(y_true[indices], y_pred[indices])

        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    lower = np.percentile(sorted_scores, (100 - ci) / 2)
    upper = np.percentile(sorted_scores, 100 - (100 - ci) / 2)
    mean_score = np.mean(sorted_scores)

    return mean_score, (lower, upper)

def main_2():
    
    output_metrics = Path("data/visualization/performance/v02/confidence/none.tsv")
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    
    inception = pd.read_csv("data/visualization/performance/inception_performance_v02.tsv",
                              sep="\t", header=None, names=['true', 'pred', 'prob_1','logit_0', 'logit_1'])
    mappy = pd.read_csv("data/visualization/performance/mappy_performance_v02.tsv",
                              sep="\t", header=None, names=['true', 'pred', 'prob_1'])
    
    linear = pd.read_csv("data/visualization/performance/v02/logistic_alldb_c1_v02_performance.txt",
                               sep="\t", header=None, names=['true', 'pred', 'prob_1'])

    transformer = pd.read_csv("data/visualization/performance/v02/validation/alldb_v02_small_iak7l6eg_performance.tsv",
                              sep="\t", header=None, names=['true', 'pred', 'prob_1','logit_0', 'logit_1'])
    
    satellite_finder = pd.read_csv("data/visualization/performance/satellite_finder/satellite_finder_performance.tsv",
                              sep="\t", header=None, names=['true', 'pred'])

    y_true_all = inception['true'].values
    y_true_transformer = transformer['true'].values

    models = {
        'CNN': {'y_pred': inception['pred'].values, 'y_pred_prob': inception['prob_1'].values},
        'Baseline-alignment': {'y_pred': mappy['pred'].values, 'y_pred_prob': mappy['prob_1'].values},
        'Baseline-linear': {'y_pred': linear['pred'].values, 'y_pred_prob': linear['prob_1'].values},
        'Transformer': {'y_pred': transformer['pred'].values, 'y_pred_prob': transformer['prob_1'].values},
        'Satellite Finder': {'y_pred': satellite_finder['pred'].values}
    }

    # Calculate metrics and confidence intervals for each model
    metric_names = ["Accuracy", "F1", "Precision", "ROC AUC", "Recall"]
    all_results = {model: [] for model in models}
    with open(output_metrics, "w") as fout:
        for model_name, predictions in models.items():
            if model_name == 'Transformer':
                y_true = y_true_transformer
            else:
                y_true = y_true_all
            y_pred = predictions['y_pred']
            
            # w roc_auc_score
            #metrics = [accuracy_score, f1_score, precision_score, roc_auc_score, recall_score]
            
            # w/o roc_auc_score
            metrics = [accuracy_score, f1_score, precision_score, recall_score]


            for metric in metrics:
                if metric == roc_auc_score:
                    y_pred_prob = predictions['y_pred_prob']
                    mean, ci = bootstrap_confidence_interval(y_true, y_pred, y_pred_prob, metric)
                else:
                    mean, ci = bootstrap_confidence_interval(y_true, y_pred, metric_func=metric)
                print(f"{model_name}\t{metric.__name__}\t{mean}\t{ci[0]}\t{ci[1]}", file=fout)
                # all_results[model_name].append((mean, ci))
            


    # # Plotting
    # plt.figure(figsize=(12, 8))
    # for i, metric in enumerate(metric_names):
    #     for j, model_name in enumerate(models):
    #         mean, ci = all_results[model_name][i]
    #         #plt.errorbar(j + i * 0.2, mean, yerr=[[mean - ci[0]], [ci[1] - mean]], fmt='o', capsize=5, label=f'{model_name} - {metric}' if i == 0 else "")
    #         plt.errorbar(j + i * 0.2, mean, yerr=[[mean - ci[0]], [ci[1] - mean]])
    
    # plt.xticks(np.arange(len(models)) + 0.2, models.keys())
    # plt.ylabel('Metric Value')
    # plt.title('Comparison of Evaluation Metrics Across Models')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.show()


def main_1():
    # Open tsv with: 
    predictions = pd.read_csv("data/visualization/performance/v02/alldb_single_v02_small_performance.tsv",
                              sep="\t", header=None, names=['true', 'pred', 'prob_1','logit_0', 'logit_1'])
    
    # predictions = pd.read_csv("data/visualization/performance/mappy_performance_v02.tsv",
    #                           sep="\t", header=None, names=['true', 'pred', 'prob_1'])

    y_true = predictions['true'].values
    y_pred = predictions['pred'].values
    y_prob = predictions['prob_1'].values

    print(y_true[0])
    print(y_pred[0])
    print(y_prob[0])

    # # calculate metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_prob)
    recall = recall_score(y_true, y_pred)

    # print("Accuracy: ", acc)
    # print("F1: ", f1)
    # print("Precision: ", precision)
    # print("Recall: ", recall)
    # print("AUC: ", auc)

    print("Original ROC area: {:0.3f}".format(roc_auc_score(y_true, y_prob)))

    n_bootstraps = 1000
    rng_seed = 1  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)
        print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
        confidence_lower, confidence_upper))

    plt.hist(bootstrapped_scores, bins=50)
    plt.title('Histogram of the bootstrapped ROC AUC scores')
    plt.show()


if __name__ == "__main__":
    main_2()
