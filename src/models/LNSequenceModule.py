import torch
import sys
import wandb
import torch.nn as nn
import pytorch_lightning as pl
from hydra.utils import instantiate
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR
from torchmetrics.functional import accuracy
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from omegaconf import DictConfig

class SequenceModule(pl.LightningModule):
    def __init__(self,
                 model_config: DictConfig,
                 lr: float,
                 batch_size: int,
                 warmup: bool,
                 steps_per_epoch: int,
                 optimizer: str,
                 class_weights: torch.Tensor,
                 vocab_map: dict,
                 fold_num: int = None):
        super(SequenceModule, self).__init__()

        self.fold_num = fold_num
        self.model = instantiate(model_config.params)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.lr = lr
        self.batch_size = batch_size
        self.warmup = warmup
        self.steps_per_epoch = steps_per_epoch
        self.optimizer = optimizer
        self.test_y_hat = []
        self.test_y = []
        self.vocab_map = vocab_map
        self.save_hyperparameters(logger=False)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        _, loss, acc = self._get_preds_loss_accuracy(batch)
        if self.fold_num is not None:
            wandb.log({f"train_CV{self.fold_num}_loss" : loss.item(),
                    f"train_CV{self.fold_num}_acc" : acc.item()})
            self.log('train_loss', loss, prog_bar=True, logger=False, batch_size=self.batch_size)
            self.log('train_acc', acc, prog_bar=True, logger=False, batch_size=self.batch_size)
        else:
            self.log('train_loss', loss, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log('train_acc', acc, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_accuracy(batch)
        if self.fold_num is not None:
            wandb.log({f"val_CV{self.fold_num}_loss" : loss.item(),
                    f"val_CV{self.fold_num}_acc" : acc.item()})
            self.log('val_loss', loss, prog_bar=True, logger=False, batch_size=self.batch_size)
            self.log('val_acc', acc, prog_bar=True, logger=False, batch_size=self.batch_size)
        else:
            self.log('val_loss', loss, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log('val_acc', acc, prog_bar=True, logger=True, batch_size=self.batch_size)
        return preds
    
    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.test_outputs = []
        return
    
    def test_step(self, batch, batch_idx) -> None:
        super().test_step()
        x, y = batch
        y_hat = self(x)
        self.test_outputs.append({'y_true': y, 'y_hat': y_hat})
        return

    def on_test_epoch_end(self):
        # concatenate all y_true and y_hat from outputs of test_step
        y_true = torch.cat([x['y_true'] for x in self.test_outputs], dim=0)
        y_hat = torch.cat([x['y_hat'] for x in self.test_outputs], dim=0)

        # convert probabilities to predicted labels
        y_pred_prob = torch.nn.functional.softmax(y_hat, dim=1)
        y_pred = y_pred_prob.argmax(dim=1)

        y_true = y_true.cpu()
        y_pred_prob = y_pred_prob.cpu()
        y_pred_prob_class_1 = y_pred_prob[:, 1] 
        y_pred = y_pred.cpu()
        y_hat = y_hat.cpu()
        
        # calculate metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='binary')
        precision = precision_score(y_true, y_pred, average='binary')
        auc = roc_auc_score(y_true, y_pred_prob_class_1)
        
        # log metrics
        self.log('test_acc', torch.tensor(acc, dtype=torch.float32), batch_size=self.batch_size)
        self.log('test_f1', torch.tensor(f1, dtype=torch.float32), batch_size=self.batch_size)
        self.log('test_precision', torch.tensor(precision, dtype=torch.float32), batch_size=self.batch_size)
        self.log('test_auc', torch.tensor(auc, dtype=torch.float32), batch_size=self.batch_size)
        
        # log ROC curve in wandb
        y_true_np = y_true.numpy()
        y_hat_np = y_pred_prob.numpy()
        
        # log to wandb if experiment has attribute log:
        if hasattr(self.logger.experiment, 'log'):
            self.logger.experiment.log({"roc": wandb.plot.roc_curve(y_true_np, y_hat_np, labels=["Class 0", "Class 1"], classes_to_plot=[1])})

            self.logger.experiment.log({"performance": wandb.Table(columns=["accuracy", "f1", "precision", "auc"],
                        data=[[acc, f1, precision, auc]])})

        self.trainer.results = self.test_outputs
        del self.test_outputs
        return

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        else:
            raise ValueError('Optimizer not supported')
        
        if self.warmup:
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.lr, # The peak LR to achieve after warmup
                epochs=self.trainer.max_epochs, # Total number of epochs
                steps_per_epoch=self.steps_per_epoch, # Number of batches in one epoch
                pct_start=0.1, # The percentage of the cycle spent increasing the LR
                anneal_strategy='cos', # How to anneal the LR (options: 'cos' or 'linear')
                final_div_factor=1e4, # The factor to reduce the LR at the end
            )
        else:
            scheduler = LambdaLR(optimizer, lambda epoch: 0.95 ** epoch)

        monitor = 'val_loss'
        return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step',
                    },
                    'monitor': monitor,
                }
    
    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        loss = self.criterion(logits, y)
        acc = accuracy(preds, y, 'binary')
        return preds, loss, acc

    def on_save_checkpoint(self, checkpoint):
        if self.vocab_map is not None:
            checkpoint['vocabulary_map'] = self.vocab_map
        else:
            checkpoint['vocabulary_map'] = {'no_vocab_map': 0}
    
    def on_load_checkpoint(self, checkpoint):
        self.vocab_map = checkpoint['vocabulary_map']