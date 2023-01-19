from torch.nn import TripletMarginLoss
from torch import Tensor

class TripletLoss:
    
    def __init__(self, margin, distance: str="l2") -> None:
        p=2 if distance == "l2" else 1
        self._loss = TripletMarginLoss(margin, p)
    
    def __repr__(self) -> str:
        return 'TripletLoss'
    
    def __call__(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        return self._loss(anchor, positive, negative)