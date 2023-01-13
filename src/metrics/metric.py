from torchmetrics import Metric
import torch
import torch.nn.functional as F


def evaluate(self, checkpoint_path, test_loader, thresh, device):
        self.model.eval()
        checkpoint_dict = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint_dict)
        nb_ta = 0
        nb_fa = 0
        nb_same_id = 0
        nb_diff_id = 0
        for i, (anchor, pos, neg) in enumerate(tqdm(test_loader, desc="Evaluating"), 0):
            with torch.no_grad():
                anchor = anchor.to(device)
                pos = pos.to(device)
                neg = neg.to(device)
                anchor_out = self.model(anchor)
                pos_out = self.model(pos)
                neg_out = self.model(neg)
                dista = F.pairwise_distance(anchor_out, pos_out)
                distb = F.pairwise_distance(anchor_out, neg_out)
                if dista < thresh:
                    nb_ta += 1
                if distb < thresh:
                    nb_fa += 1

                nb_same_id += 1
                nb_diff_id += 1
            
        return {"threshold": thresh, "VAL": nb_ta / nb_same_id, "FAR": nb_fa / nb_diff_id}


class AccuracyMetric(Metric):
    
    def __init__(self, margin: float) -> None:
        super().__init__()
        self.margin = margin
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def compute(self) -> None:
        return self.correct.float() / self.total.float()
    
    def update(self, anchor: torch.tensor, positive: torch.tensor, negative: torch.tensor) -> None:
        pos_dists = F.pairwise_distance(anchor, positive)
        neg_dists = F.pairwise_distance(anchor, negative)
        pred = (neg_dists - pos_dists - self.margin).cpu().data
        self.correct += (pred > 0).sum()*1.0
        self.total += pos_dists.size()[0]
    
    