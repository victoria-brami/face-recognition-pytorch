from torchmetrics import Metric
from torch import Tensor
from typing import List
import torch
import torch.nn.functional as F


class AccuracyMetric(Metric):
    
    def __init__(self, margin: float) -> None:
        super().__init__()
        self.margin = margin
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def compute(self) -> float:
        return self.correct.float() / self.total
    
    def update(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> None:
        pos_dists = F.pairwise_distance(anchor, positive)
        neg_dists = F.pairwise_distance(anchor, negative)
        pred = (neg_dists - pos_dists - self.margin*torch.ones((pos_dists.shape))).cpu()
        self.correct += torch.sum(pred > 0)
        self.total += pos_dists.numel()
        
        
class PrecisionRecallMetric(Metric):
    """ computes the proportion of true positive 
        and false positive among the triplets
        depending on a threshold value
    """
    
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold
        self.add_state("true_pos", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_pos", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def compute(self) -> float:
        return {"TP": self.true_pos.float() / self.total,  "FP": self.false_pos.float() / self.total}
    
    def update(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> None:
        pos_dists = F.pairwise_distance(anchor, positive)
        neg_dists = F.pairwise_distance(anchor, negative)
        print(pos_dists)
        print(type(self.threshold))
        print(torch.ones(pos_dists.shape))
        self.true_pos += torch.sum(pos_dists < self.threshold*torch.ones(pos_dists.shape))
        self.false_pos += torch.sum(neg_dists < self.threshold*torch.ones(neg_dists.shape))
        self.total += pos_dists.numel()
        
        
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
        metrics["VAL"] = self._precision_recall_metric.compute()["TP"]
        metrics["FAR"] = self._precision_recall_metric.compute()["FP"]
        return {**metrics}
    
    def update(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> None:
        self._accuracy_metric.update(anchor, positive, negative)
        self._precision_recall_metric.update(anchor, positive, negative)
        
# Visualiser des jugements de similarité (créer une échelle avec visages très similaires: Jozwik et al. 2022, face dissimilarity judgements...)
    
if __name__ == '__main__':    
    import torch 
    import unittest

    class DatasetTests(unittest.TestCase):
        def setUp(self):
            self.acc_metric = AccuracyMetric(margin=0.)
            self.eval_metric = EvaluationMetric(threshold=1.)
            self.prec_rec_metric = PrecisionRecallMetric(threshold=1.)
            
        def test_prec_recall_computation(self):
            self.prec_rec_metric.reset()
            a = torch.tensor([1.])
            b = torch.tensor([1.1])
            c = torch.tensor([4.2])
            self.prec_rec_metric.update(a, b, c)
            print(self.prec_rec_metric.compute())
            self.prec_rec_metric.update(a, b, c)
            print(self.prec_rec_metric.compute())
            self.assertEqual(self.prec_rec_metric.compute()['TP'], 1, "2nd Wrong computation")
            self.assertEqual(self.prec_rec_metric.compute()['FP'], 0, "3rd Wrong computation")
            self.prec_rec_metric.update(a, c, b)
            print(self.prec_rec_metric.compute() )
            self.assertEqual(self.prec_rec_metric.compute()['FP'], 1 / 3, "3rd Wrong computation")
            self.assertEqual(self.prec_rec_metric.compute()['TP'], 2 / 3, "3rd Wrong computation")
            self.prec_rec_metric.update(a, c, b)
            print("Last",self.prec_rec_metric.compute())

        def test_acc_computation(self):
            self.acc_metric.reset()
            a = torch.tensor([1.])
            b = torch.tensor([1.1])
            c = torch.tensor([1.2])
            self.acc_metric.update(a, b, c)
            print(self.acc_metric.compute())
            self.assertEqual(self.acc_metric.compute(), 1, "2nd Wrong computation")
            self.acc_metric.update(a, c, b)
            print(self.acc_metric.compute())
            self.assertEqual(self.acc_metric.compute(), 0.5, "3rd Wrong computation")
           
    unittest.main(argv=[''], exit=False)