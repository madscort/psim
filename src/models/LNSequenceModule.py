import torch
import wandb
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import CNN_collection as CNN
import LSTM_collection as LSTM
import Transformer_collection as Transformers

from torchmetrics.functional import accuracy
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score


MODEL_REGISTRY = {
    'BasicCNN': CNN.BasicCNN,
    'BasicInception': CNN.BasicInception,
    'BasicLSTM': LSTM.BasicLSTM,
    'BasicTransformer': Transformers.BasicTransformer
}

class SequenceModule(pl.LightningModule):
    def __init__(self,
            model,
            lr=0.001,
            optimizer='adam',
            activation_fn='ReLU',
            alt_dropout_rate=0.1,
            fc_dropout_rate=0.5,
            batchnorm=False,
            fc_num=2,
            fold_num: int = None,
            kernel_size: tuple=(3,3,3),
            num_inception_layers: int = 1,
            out_channels: int = 16,
            kernel_size_b1: int = 3,
            kernel_size_b2: int = 5,
            keep_b3: bool = True,
            keep_b4: bool = True,
            pad_pack: bool = False,
            model_input_size: int = 25000,
            hidden_size_lstm: int = 64,
            num_layers_lstm: int = 1,
            embedding_dim: int = None,
            vocab_size: int = 5):
        super(SequenceModule, self).__init__()
        self.fold_num = fold_num
        self.model = MODEL_REGISTRY[model](activation_fn=activation_fn,
            alt_dropout_rate=alt_dropout_rate,
            fc_dropout_rate=fc_dropout_rate,
            batchnorm=batchnorm,
            fc_num=fc_num,
            kernel_size=kernel_size,
            num_inception_layers=num_inception_layers,
            out_channels=out_channels,
            kernel_size_b1=kernel_size_b1,
            kernel_size_b2=kernel_size_b2,
            keep_b3=keep_b3,
            keep_b4=keep_b4,
            pad_pack=pad_pack,
            input_size=model_input_size,
            hidden_size_lstm=hidden_size_lstm,
            num_layers_lstm=num_layers_lstm,
            embedding_dim=embedding_dim,
            vocab_size=vocab_size)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer = optimizer
        self.test_y_hat = []
        self.test_y = []

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
