import pytorch_lightning as pl
from .lfw import LFW
from typing import Optional
from torch.utils.data import DataLoader, random_split
from torch import Generator
from pathlib import Path
import os




class FaceTripletsDataModule(pl.LightningDataModule):
    
    def __init__(self, 
    data_dir: Path,
    train_prop: float,
    seed: int,
    batch_size: int, 
    num_workers: int, 
    pin_memory: False
) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_path = os.path.join(data_dir, "train")
        self.test_path = os.path.join(data_dir, "test")
        self.train_prop = train_prop
        self.seed = seed
        self.data_train: Optional[LFW] = None
        self.data_val: Optional[LFW] = None
        self.data_test: Optional[LFW] = None
        
        
    def setup(self, stage=None) -> None:
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = LFW(self.train_path, transform=None)
            testset = LFW(self.test_path, transform=None)
            lengths = [int(len(trainset)*self.train_prop)+1, int(len(trainset)*(1-self.train_prop))]
            self.data_test = testset
            self.data_train, self.data_val = random_split(
                dataset=trainset,
                lengths=lengths,
                generator=Generator().manual_seed(self.seed),
            )
            print("Train path", self.train_path, self.test_path)
            print("Train Set length: {}".format(len(self.data_train)))
            print("Val Set length: {}".format(len(self.data_val)))
            print("Test Set length: {}".format(len(self.data_test)))
    
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