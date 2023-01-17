import numpy as np
import torch.nn as nn
import torch



def get_valid_triplets(pos_dists: torch.tensor, neg_dists: torch.tensor, margin: float, semi_hard_negative: bool=True) -> np.ndarray:
    
    first_condition = (neg_dists - pos_dists < margin).cpu().numpy().flatten()
    
    if semi_hard_negative:
        # Semi-Hard Negative triplet selection
        #  (negative_distance - positive_distance < margin) AND (positive_distance < negative_distance)
        #   Based on: https://github.com/davidsandberg/facenet/blob/master/src/train_tripletloss.py#L295
        
        second_condition = (pos_dists < neg_dists).cpu().numpy().flatten()
        
    else:
        # Use hard negative triplets
        second_condition = True
    all = (np.logical_and(first_condition, second_condition))
    valid_triplets = np.where(all == 1)
    
    return valid_triplets