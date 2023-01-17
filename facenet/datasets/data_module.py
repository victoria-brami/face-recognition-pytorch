import pytorch_lightning as pl
from .lfw import LFW
from typing import Optional
from torch.utils.data import DataLoader, random_split
from torch import Generator




class FaceTripletsDataModule(pl.LightningDataModule):
    
    def __init__(self, hparams) -> None:
        super().__init__()
        self.hparams = hparams
        self.data_train: Optional[LFW] = None
        self.data_val: Optional[LFW] = None
        self.data_test: Optional[LFW] = None
        
        
    def setup(self) -> None:
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = LFW(self.hparams.train_path, transform=None)
            testset = LFW(self.hparams.test_path, transform=None)
            lengths = [int(len(trainset)*self.hparams.train_prop)+1, int(len(trainset)*(1-self.hparams.train_prop))]
            self.data_test = testset
            self.data_train, self.data_val = random_split(
                dataset=trainset,
                lengths=lengths,
                generator=Generator().manual_seed(self.hparams.seed),
            )
    
    def train_dataloader(self):
        return DataLoader(self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )
    
    def test_dataloader(self):
        return DataLoader(self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )