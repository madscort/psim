import hydra
import wandb
import pickle
import pandas as pd
import sys
from pathlib import Path
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from src.data.LN_data_module import FixedLengthSequenceModule
from src.models.LNSequenceModule import SequenceModule
from pytorch_lightning.callbacks import ModelCheckpoint

@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")

def main(cfg: DictConfig):
    project = cfg.project
    accelerator = cfg.accelerator
    devices = cfg.devices
    dataset = cfg.dataset
    model_type = cfg.model.type
    model_name = cfg.model.name
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    epochs = cfg.epochs
    activation_fn = cfg.model.activation_fn
    alt_dropout_rate = cfg.model.alt_dropout_rate
    fc_dropout_rate = cfg.model.fc_dropout_rate
    batchnorm = cfg.model.batchnorm
    fc_num = cfg.model.fc_num
    kernel_size_1 = cfg.model.kernel_size_1
    kernel_size_2 = cfg.model.kernel_size_2
    kernel_size_3 = cfg.model.kernel_size_3
    optimizer = cfg.optimizer.name
    lr = cfg.optimizer.lr
    num_inception_layers = cfg.model.num_inception_layers
    out_channels = cfg.model.out_channels
    kernel_size_b1 = cfg.model.kernel_size_b1
    kernel_size_b2 = cfg.model.kernel_size_b2
    keep_b3 = cfg.model.keep_b3
    keep_b4 = cfg.model.keep_b4
    group_name = "cv_CNNInception"
    
    dataset_root = Path("data/processed/10_datasets/")
    dataset = Path(dataset_root, dataset)
    sampletable = Path(dataset, "sampletable.tsv")  

    # Initialize the StratifiedKFold object
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    # Load sampletable
    df_sampletable = pd.read_csv(sampletable, sep="\t", header=None, names=['id', 'type', 'label'])
    
    fold_results = []

    for fold, (fit_idx, test_idx) in enumerate(skf.split(df_sampletable['id'], df_sampletable['type'])):
        
        wandb_logger = WandbLogger(project=project,
                               config=OmegaConf.to_container(cfg,
                                                             resolve=True),
                               name=f"{model_name}_fold{fold+1}",
                               group=group_name)

        # split the fit_idx into train and val
        train_idx, val_idx = train_test_split(fit_idx, stratify=df_sampletable.iloc[fit_idx]['type'], test_size=0.2)

        data_module = FixedLengthSequenceModule(dataset=dataset,
                                                train_indices=train_idx,
                                                val_indices=val_idx,
                                                test_indices=test_idx,
                                                num_workers=num_workers,
                                                batch_size=batch_size)

        model = SequenceModule(model_name,
                            lr=lr,
                            optimizer=optimizer,
                            activation_fn=activation_fn,
                            fc_dropout_rate=alt_dropout_rate,
                            batchnorm=batchnorm,
                            fc_num=fc_num,
                            # BasicCNN only:
                            alt_dropout_rate=alt_dropout_rate,
                            kernel_size=(kernel_size_1,kernel_size_2,kernel_size_3),
                            # Inception only:
                            num_inception_layers=num_inception_layers,
                            out_channels=out_channels,
                            kernel_size_b1=kernel_size_b1,
                            kernel_size_b2=kernel_size_b2,
                            keep_b3=keep_b3,
                            keep_b4=keep_b4)

        early_stop_callback = EarlyStopping(
                                        monitor='val_loss',
                                        min_delta=0.00,
                                        patience=5,
                                        verbose=False,
                                        mode='min')

        checkpoint_path = Path(str(wandb_logger.experiment.dir), f"fold{fold}")
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path,
                                              monitor='val_loss', mode='min')

        trainer = Trainer(accelerator=accelerator,
                            devices=devices,
                            max_epochs=epochs,
                            logger=wandb_logger,
                            callbacks=[LearningRateMonitor(logging_interval='step'),
                                        early_stop_callback,
                                        checkpoint_callback])

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