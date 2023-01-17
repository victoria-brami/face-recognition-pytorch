from torchmetrics import Metric
from torch import Tensor
from typing import List
import torch.nn.functional as F


class AccuracyMetric(Metric):
    
    def __init__(self, margin: float) -> None:
        super().__init__()
        self.margin = margin
        self.add_state("correct", default=Tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=Tensor(0), dist_reduce_fx="sum")
        
    def compute(self) -> float:
        return self.correct.float() / self.total.float()
    
    def update(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> None:
        pos_dists = F.pairwise_distance(anchor, positive)
        neg_dists = F.pairwise_distance(anchor, negative)
        pred = (neg_dists - pos_dists - self.margin).cpu().data
        self.correct += (pred > 0).sum()*1.0
        self.total += pos_dists.size()[0]
        
        
class PrecisionRecallMetric(Metric):
    """ computes the proportion of true positive 
        and false positive among the triplets
        depending on a threshold value
    """
    
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold
        self.add_state("true_pos", default=Tensor(0), dist_reduce_fx="sum")
        self.add_state("false_pos", default=Tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=Tensor(0), dist_reduce_fx="sum")
        
    def compute(self, true_pos: bool=True) -> float:
        return self.true_pos.float() / self.total.float() if true_pos else self.false_pos.float() / self.total.float()
    
    def update(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> None:
        pos_dists = F.pairwise_distance(anchor, positive)
        neg_dists = F.pairwise_distance(anchor, negative)
        self.true_pos = (pos_dists < self.threshold).sum()*1.0
        self.false_pos = (neg_dists < self.threshold).sum()*1.0
        self.total += pos_dists.size()[0]
        
        
class EvaluationMetric(Metric):
    
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self._accuracy_metric = AccuracyMetric(margin=0.0)
        self._precision_recall_metric = PrecisionRecallMetric(threshold=threshold)
        self.accuracy = 0.
        self.threshold = threshold
        self.VAR = 0.
        self.FAR = 0.
        self.metric_list = ["accuracy", "threshold", "VAL", "FAR"]
        
    def compute(self):
        metrics = {metric: getattr(self, metric) for metric in self.metric_list}
        metrics["accuracy"] = self._accuracy_metric.compute()
        metrics["VAL"] = self._precision_recall_metric.compute(true_pos=True)
        metrics["FAR"] = self._precision_recall_metric.compute(true_pos=False)
        return {**metrics}
    
    def update(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> None:
        self._accuracy_metric.update(anchor, positive, negative)
        self._precision_recall_metric.update(anchor, positive, negative)
        
# Visualiser des jugements de similarité (créer une échelle avec visages très similaires: Jozwik et al. 2022, face dissimilarity judgements...)
    
    