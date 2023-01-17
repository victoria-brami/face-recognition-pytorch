import pytorch_lightning as pl
from . import FaceNet
from ..losses import TripletLoss
import torch
from ..metrics import EvaluationMetric, AccuracyMetric
from torchmetrics import MeanMetric, MaxMetric
from typing import Any, List




class FaceNetLitModule(pl.LightningModule):
    
    def __init__(self, 
                net: FaceNet,
                train_margin: float,
                val_margin: float,
                test_margin: float,
                threshold: float,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler,
                 ) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False)
        self.net = net
        self.train_margin = train_margin
        self.val_margin = val_margin
        self.test_margin = test_margin
        self.threshold = threshold,
        self.criterion = TripletLoss(margin=self.train_margin)
        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.train_worst_loss = MaxMetric()
        self.val_worst_loss = MaxMetric()

        self.train_acc = AccuracyMetric(margin=0)
        self.val_acc = AccuracyMetric(margin=0)
        self.val_acc_best = MaxMetric()
        self.val_low_acc = AccuracyMetric(margin=self.val_margin)
        self.test_acc = EvaluationMetric(threshold=self.threshold)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_low_acc.reset()
        self.train_worst_loss.reset()

    def forward(self, x: torch.Tensor):
        return self.net(x)


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    
    def training_step(self, batch: Any, batch_idx: int):
        a, p, n = batch
        a_out = self.forward(a)
        p_out = self.forward(p)
        n_out = self.forward(n)
        loss = self.criterion(a_out, p_out, n_out)
        self.train_loss(loss)
        self.train_worst_loss(loss)
        self.train_acc(a_out, p_out, n_out)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/worst_loss", self.train_worst_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}


    def validation_step(self, batch: Any, batch_idx: int):
        a, p, n = batch
        a_out = self.forward(a)
        p_out = self.forward(p)
        n_out = self.forward(n)
        loss = self.criterion(a_out, p_out, n_out)
        self.val_loss(loss)
        self.val_worst_loss(loss)
        self.val_acc(a_out, p_out, n_out)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/worst_loss", self.val_worst_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}
    
    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    
    def test_step(self, batch: Any, batch_idx: int):
        a, p, n = batch
        a_out = self.forward(a)
        p_out = self.forward(p)
        n_out = self.forward(n)
        loss = self.criterion(a_out, p_out, n_out)
        self.test_loss(loss)
        self.test_acc(a_out, p_out, n_out)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
       
        return {"loss": loss}