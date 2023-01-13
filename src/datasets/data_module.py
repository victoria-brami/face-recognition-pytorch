import pytorch_lightning as pl
from ..configs import DataConfig
from .lfw import LFW




class FaceTripletsDataModule(pl.LightningDataModule):
    
    def __init__(self, data_cfg: DataConfig) -> None:
        super().__init__()
        self.add_state('train_labels', default=[])
        self.add_state('val_labels', default=[])
        self.data_cfg = data_cfg
        
        
    def _split_datasets_into_train_val(self) -> None:
        pass
    
    def train_dataloader(self):
        return LFW(self.data_cfg.train_path)
    
    def test_dataloader(self):
        return LFW(self.data_cfg.test_path)