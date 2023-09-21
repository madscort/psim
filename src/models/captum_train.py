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
from pytorch_lightning.callbacks import ModelSummary

from src.data.LN_data_module import FixedLengthSequenceModule
from src.models.CNN_captum import CaptumModule

@hydra.main(config_path="../../configs", config_name="captum_config", version_base="1.2")
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
    lr = cfg.optimizer.lr
    dataset_return_type = cfg.data.return_type
    pad_pack = cfg.data.pad_pack
    use_saved = cfg.data.use_saved

    seed_everything(1)
    
    dataset_root = Path("data/processed/10_datasets/")
    dataset = Path(dataset_root, dataset)
    data_module = FixedLengthSequenceModule(dataset=dataset,
                                            return_type=dataset_return_type,
                                            num_workers=num_workers,
                                            batch_size=batch_size,
                                            pad_pack=pad_pack,
                                            use_saved=use_saved)
    wandb_logger = WandbLogger(project=project,
                               config=OmegaConf.to_container(cfg,
                                                             resolve=True),
                               name=model_name,
                               group=model_type)

    data_module.setup()
    model = CaptumModule(lr=lr)

    early_stop_callback = EarlyStopping(monitor='val_loss',
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
