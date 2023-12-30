import hydra
import wandb
import sys
from collections import Counter
from pathlib import Path
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from scipy.stats import t

from src.data.LN_data_module import FixedLengthSequenceModule
from src.models.LNSequenceModule import SequenceModule

rename = {'test_acc': 'accuracy_score', 'test_f1': 'f1_score', 'test_precision': 'precision_score', 'test_auc': 'roc_auc_score', 'test_recall': 'recall_score'}

def calculate_means_and_cis_to_file(metrics_list, file_path):
    # Initialize a dictionary to hold our aggregated metrics
    aggregated_metrics = {}
    
    # Loop through each metric in the first fold to initialize the keys and lists
    for key in metrics_list[0].keys():
        aggregated_metrics[key] = []

    # Now populate the lists with values from all folds
    for metrics in metrics_list:
        for key, value in metrics.items():
            aggregated_metrics[key].append(value)

    # Number of observations (folds)
    n = len(metrics_list)
    # Significance level for 95% confidence
    alpha = 0.05
    # Degrees of freedom
    df = n - 1
    # Critical t value for two tails
    t_crit = abs(t.ppf(alpha/2, df))

    # Calculate the mean and confidence intervals
    with open(file_path, 'w') as fout:
        for metric, values in aggregated_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            ci_half_width = t_crit * (std_val / np.sqrt(n))
            lower_ci = mean_val - ci_half_width
            upper_ci = mean_val + ci_half_width
            
            print('Transformer', rename[metric], mean_val, lower_ci, upper_ci, sep='\t', file=fout)

def custom_stratify(df, stratify_col, small_class_threshold=10):

    small_classes = df[stratify_col].value_counts()[df[stratify_col].value_counts() < small_class_threshold].index
    df_small = df[df[stratify_col].isin(small_classes)]
    df_large = df[~df[stratify_col].isin(small_classes)]
    train_idx_large, test_idx_large = train_test_split(df_large.index, stratify=df_large[stratify_col], test_size=0.1, random_state=1)
    train_idx_small, test_idx_small = train_test_split(df_small.index, test_size=0.1, random_state=1)
    train_idx = train_idx_large.union(train_idx_small)
    test_idx = test_idx_large.union(test_idx_small)

    return train_idx, test_idx

@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    seed_everything(1)

    output_metrics = Path("data/visualization/performance/v02/confidence/cross_transformer.tsv")
    output_metrics_raw = Path("data/visualization/performance/v02/confidence/cross_transformer_raw.tsv")
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    dataset_root = Path("data/processed/10_datasets/")
    dataset = Path(dataset_root, cfg.dataset)
    sampletables = [pd.read_csv(dataset / f"{split}.tsv", sep="\t", header=0, names=["id", "type", "label"]) for split in ['train', 'val', 'test']]
    df_sampletable = pd.concat(sampletables, ignore_index=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    fold_results = []

    for fold, (fit_idx, test_idx) in enumerate(skf.split(df_sampletable['id'], df_sampletable['type'])):
        print(f"Fold {fold}")
        if fold == 2:
            break
        train_idx, val_idx = custom_stratify(df_sampletable.iloc[fit_idx], 'type')

        
        data_module = FixedLengthSequenceModule(dataset=dataset,
                                                train_indices=train_idx,
                                                val_indices=val_idx,
                                                test_indices=test_idx,
                                                return_type=cfg.model.data.return_type,
                                                num_workers=cfg.num_workers,
                                                batch_size=cfg.batch_size,
                                                pad_pack=cfg.model.data.pad_pack,
                                                use_saved=cfg.model.data.use_saved)
        data_module.setup()

        # Populate vocabulary size and max sequence length if used.
        if 'vocab_size' in cfg.model.params:
            cfg.model.params.vocab_size = data_module.vocab_size
        if 'max_seq_length' in cfg.model.params:
            cfg.model.params.max_seq_length = data_module.max_seq_length
        
        class_weights = data_module.class_weights
        steps_per_epoch = data_module.steps_per_epoch
        vocab_map = data_module.vocab_map

        wandb_logger = WandbLogger(project=cfg.project,
                                   config=OmegaConf.to_container(cfg,resolve=True),
                                   name=f"{cfg.run_name}_fold{fold+1}",
                                   group=cfg.run_group)

        model = SequenceModule(model_config=cfg.model,
                               lr=cfg.optimizer.lr,
                           batch_size=cfg.batch_size,
                           steps_per_epoch=steps_per_epoch,
                           optimizer=cfg.optimizer.name,
                           warmup=cfg.optimizer.warmup,
                           class_weights=class_weights,
                           vocab_map=vocab_map)

        early_stop_callback = EarlyStopping(
                                        monitor='val_loss',
                                        min_delta=0.00,
                                        patience=5,
                                        verbose=False,
                                        mode='min')

        checkpoint_path = Path(str(wandb_logger.experiment.dir), f"fold{fold}")
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path,
                                              monitor='val_loss', mode='min')

        trainer = Trainer(accelerator=cfg.accelerator,
                      devices=cfg.devices,
                      max_epochs=cfg.epochs,
                      logger=wandb_logger,
                      callbacks=[LearningRateMonitor(logging_interval='step'),
                                 checkpoint_callback,
                                 early_stop_callback])

        trainer.fit(model, datamodule=data_module)

        outputs = trainer.test(ckpt_path="best", datamodule=data_module)
        fold_results.append(outputs[0])
        wandb.finish()
    
    metrics = ['test_acc', 'test_f1', 'test_precision', 'test_auc', 'test_recall']
    with open(output_metrics_raw, "w") as fin:
        for n, result in enumerate(fold_results):
            for metric in metrics:
                print(f"Transformer_{n}\t{rename[metric]}\t{result[metric]}", file=fin)
        
    calculate_means_and_cis_to_file(fold_results, output_metrics)

if __name__ == "__main__":
    main()