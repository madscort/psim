import hydra
from pathlib import Path
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import torch
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
    # Populate vocabulary size if embedding layer is used.
    if 'vocab_size' in cfg.model.params:
        cfg.model.params.vocab_size = data_module.vocab_size

    model = SequenceModule.load_from_checkpoint(checkpoint_path=Path("psim/xe6gkzog/checkpoints/epoch=0-step=148.ckpt").absolute())
    model.eval()

if __name__ == "__main__":
    main()
