import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet18, resnet34
from torchvision import models



class FaceNet(nn.Module):
    
    def __init__(self, embedding_dim: int, modelname: str="resnet18") -> None:
        super().__init__()
        self.modelname = modelname
        self.embedding_dim = embedding_dim
        self.model = self._build_model()
        

    def _build_model(self) -> nn.Module:

        if self.modelname == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif self.modelname == 'resnet34':
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif self.modelname == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif self.modelname == 'resnet101':
            model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        elif self.modelname == 'resnet152':
            model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        else:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        num_ftrs = model.fc.in_features
        model.fc =  nn.Linear(num_ftrs, self.embedding_dim)
        
        return model
        
        
    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.normalize(self.model(x))