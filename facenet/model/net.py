import torch.nn as nn
from torch import Tensor

from facenet.utils import instantiate_default_model



class FaceNet(nn.Module):
    
    def __init__(self, embedding_dim: int, modelname: str="resnet18") -> None:
        super().__init__()
        self.modelname = modelname
        self.embedding_dim = embedding_dim
        self.model = self._build_model()
        

    def _build_model(self) -> nn.Module:
        
        model = instantiate_default_model(self.modelname)
        
        if 'resnet' in getattr(self, 'modelname'):
            num_ftrs = model.fc.in_features
            model.fc =  nn.Linear(num_ftrs, self.embedding_dim)

        elif 'vit' in  getattr(self, 'modelname'):
            num_ftrs = model.heads.head.in_features
            model.heads.head = nn.Linear(num_ftrs, self.embedding_dim)

        return model
        
        
    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.normalize(self.model(x))