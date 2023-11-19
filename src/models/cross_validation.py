import hydra
import wandb
import sys
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

from src.data.LN_data_module import FixedLengthSequenceModule
from src.models.LNSequenceModule import SequenceModule


@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    seed_everything(1)


    dataset_root = Path("data/processed/10_datasets/")
    dataset = Path(dataset_root, cfg.dataset)
    sampletable = Path(dataset, "sampletable.tsv")  

    # Initialize the StratifiedKFold object
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    # Load sampletable
    df_sampletable = pd.read_csv(sampletable, sep="\t", header=None, names=['id', 'type', 'label'])
    
    fold_results = []

    for fold, (fit_idx, test_idx) in enumerate(skf.split(df_sampletable['id'], df_sampletable['type'])):
        
        wandb_logger = WandbLogger(project=cfg.project,
                               config=OmegaConf.to_container(cfg,
                                                             resolve=True),
                               name=f"{cfg.run_name}_fold{fold+1}",
                               group=cfg.run_group)


        # split the fit_idx into train and val
        train_idx, val_idx = train_test_split(fit_idx, stratify=df_sampletable.iloc[fit_idx]['type'], test_size=0.2)

        
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
        
        print(outputs)
        # y_hat = outputs[0]['y_hat'].cpu().numpy()
        # y = outputs[0]['y'].cpu().numpy()

        # # Save predictions and ground truth labels for this fold
        # fold_results.append((y_hat, y))
        wandb.finish()

    # Save the results to disk
    # with open(Path('data/visualization/cross_validation/cross_val_results.pkl').absolute(), 'wb') as f:
    #     pickle.dump(fold_results, f)

if __name__ == "__main__":
    main()