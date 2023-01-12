from torchmetrics import Metric
from torch import nn


class TripletMetric(Metric):
    
    def __init__(self, margin: float) -> None:
        super().__init__()
        self._criterion = nn.TripletMarginLoss(margin=margin)
        self._acc = None
        
    def update(self, a, b):
        pass