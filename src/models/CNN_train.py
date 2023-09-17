import hydra
import wandb
from pathlib import Path
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelSummary
from src.data.LN_data_module import FixedLengthSequenceModule
from src.models.LNSequenceModule import SequenceModule

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
    conv_num = cfg.model.conv_num
    kernel_size_1 = cfg.model.kernel_size_1
    kernel_size_2 = cfg.model.kernel_size_2
    optimizer = cfg.optimizer.name
    lr = cfg.optimizer.lr
    
    dataset_root = Path("data/processed/10_datasets/")
    dataset = Path(dataset_root, dataset)
    data_module = FixedLengthSequenceModule(dataset=dataset,
                                            num_workers=num_workers,
                                            batch_size=batch_size)
    wandb_logger = WandbLogger(project=project,
                               config=OmegaConf.to_container(cfg,
                                                             resolve=True),
                               name=model_name,
                               group=model_type)

    model = SequenceModule(model_name,lr=lr, optimizer=optimizer, activation_fn=activation_fn, alt_dropout_rate=alt_dropout_rate, fc_dropout_rate=fc_dropout_rate, batchnorm=batchnorm, fc_num=fc_num, conv_num=conv_num, kernel_size=(kernel_size_1,kernel_size_2,3))

    early_stop_callback = EarlyStopping(
                                        monitor='val_loss',
                                        min_delta=0.00,
                                        patience=5,
                                        verbose=False,
                                        mode='min')
                                            
    trainer = Trainer(accelerator=accelerator,
                      devices=devices,
                      max_epochs=epochs,
                      logger=wandb_logger,
                      callbacks=[LearningRateMonitor(logging_interval='step'),
                                 early_stop_callback,
                                 ModelSummary(max_depth=10)])
    
    trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)

    wandb.finish()

if __name__ == "__main__":
    main()
