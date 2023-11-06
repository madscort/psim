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

from src.data.LN_data_module import FixedLengthSequenceModule
from src.models.LNSequenceModule import SequenceModule

@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):

    seed_everything(1)
    
    dataset_root = Path("data/processed/10_datasets/")
    dataset = Path(dataset_root, cfg.dataset)
    data_module = FixedLengthSequenceModule(dataset=dataset,
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
                               config=OmegaConf.to_container(cfg,
                                                             resolve=True),
                               name=cfg.run_name,
                               group=cfg.run_group)
    model = SequenceModule(model_config=cfg.model,
                           lr=cfg.optimizer.lr,
                           batch_size=cfg.batch_size,
                           steps_per_epoch=steps_per_epoch,
                           optimizer=cfg.optimizer.name,
                           class_weights=class_weights,
                           vocab_map=vocab_map)

    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        min_delta=0.00,
                                        patience=5,
                                        verbose=False,
                                        mode='min')
                                            
    trainer = Trainer(accelerator=cfg.accelerator,
                      devices=cfg.devices,
                      max_epochs=cfg.epochs,
                      logger=wandb_logger,
                      callbacks=[LearningRateMonitor(logging_interval='step'),
                                 early_stop_callback])
    
    trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)

    wandb.finish()

if __name__ == "__main__":
    main()
