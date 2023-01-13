from torch.nn import TripletMarginLoss
import torch

class TripletLoss:
    
    def __init__(self, margin, distance: str="l2") -> None:
        if distance == "l2":
            p=2
        self._loss = TripletMarginLoss(margin, p)
    
    def __repr__(self) -> str:
        return 'TripletLoss'
    
    def __call__(self, anchor: torch.tensor, positive: torch.tensor, negative: torch.tensor) -> torch.tensor:
        return self._loss(anchor, positive, negative)