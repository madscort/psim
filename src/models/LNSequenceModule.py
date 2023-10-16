import torch
import wandb
import torch.nn as nn
import pytorch_lightning as pl
from hydra.utils import instantiate

from torchmetrics.functional import accuracy
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from omegaconf import DictConfig

class SequenceModule(pl.LightningModule):
    def __init__(self,
                 model_config: DictConfig,
                 lr: float,
                 optimizer: str,
                 fold_num: int = None):
        super(SequenceModule, self).__init__()

        self.fold_num = fold_num
        self.model = instantiate(model_config.params)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer = optimizer
        self.test_y_hat = []
        self.test_y = []
        self.save_hyperparameters(logger=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _, loss, acc = self._get_preds_loss_accuracy(batch)
        if self.fold_num is not None:
            wandb.log({f"train_CV{self.fold_num}_loss" : loss,
                       f"train_CV{self.fold_num}_acc" : acc})
            self.log('train_loss', loss, logger=False, batch_size=batch[1].shape[0])
            self.log('train_acc', acc, prog_bar=True, logger=False, batch_size=batch[1].shape[0])
        else:
            self.log('train_loss', loss, batch_size=batch[1].shape[0])
            self.log('train_acc', acc, prog_bar=True, batch_size=batch[1].shape[0])
        return loss
    
    def validation_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_accuracy(batch)
        if self.fold_num is not None:
            wandb.log({f"val_CV{self.fold_num}_loss" : loss,
                       f"val_CV{self.fold_num}_acc" : acc})
            self.log('val_loss', loss, logger=False, batch_size=batch[1].shape[0])
            self.log('val_acc', acc, prog_bar=True, logger=False, batch_size=batch[1].shape[0])
        else:
            self.log('val_loss', loss, prog_bar=True, batch_size=batch[1].shape[0])
            self.log('val_acc', acc, prog_bar=True, batch_size=batch[1].shape[0])
        return preds
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if not hasattr(self, 'test_outputs'):
            self.test_outputs = []
        self.test_outputs.append({'y_true': y, 'y_hat': y_hat})

        return {'y_true': y, 'y_hat': y_hat}

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
        self.log('test_acc', torch.tensor(acc, dtype=torch.float32))
        self.log('test_f1', torch.tensor(f1, dtype=torch.float32))
        self.log('test_precision', torch.tensor(precision, dtype=torch.float32))
        self.log('test_auc', torch.tensor(auc, dtype=torch.float32))
        
        # log ROC curve in wandb
        y_true_np = y_true.numpy()
        y_hat_np = y_pred_prob.numpy()

        self.logger.experiment.log({"roc": wandb.plot.roc_curve(y_true_np, y_hat_np, labels=["Class 0", "Class 1"], classes_to_plot=[1])})

        self.logger.experiment.log({"performance": wandb.Table(columns=["accuracy", "f1", "precision", "auc"],
                    data=[[acc, f1, precision, auc]])})

        del self.test_outputs
        return {'y_true': y_true, 'y_hat': y_hat}

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
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.95 ** epoch)
        monitor = 'val_loss'
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': monitor}
    
    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        loss = self.criterion(logits, y)
        acc = accuracy(preds, y, 'binary')
        return preds, loss, acc
