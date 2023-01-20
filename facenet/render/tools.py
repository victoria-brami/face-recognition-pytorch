import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Union, Dict, Optional, Tuple
from pathlib import Path
from facenet.model import FaceNet
import numpy as np
import torch
from facenet.utils import (extract_name_from_face_image_path, 

                     load_image, 
                     get_embedding_dims_from_checkpoint)


def get_metrics(image_one: Image, image_two: Image, facenet: FaceNet,) -> Dict:
    """Gets the distances between two images vector representations

    Args:
        image_one (Image): first image to be compared
        image_two (Image): second image to be compared
        facenet (FaceNet): network in eval mode for prediction

    Returns:
        Dict: Contains the distance between the 2 faces representations
    """
    image_one_output = facenet(image_one[None, :])
    image_two_output = facenet(image_two[None, :]) 
    
    dist = np.linalg.norm(image_one_output - image_two_output, p=2)
    
    return {"Distance": dist}
    


def show_predictions(image_one: Image, 
                     image_two: Image, 
                     image_one_name: str,
                     image_two_name: str,
                     metric: Dict,
                     save_path: Optional[Path],
                     fig_size: Tuple[int]=(20, 20)
                     ) -> None:
    """_summary_

    Args:
        image_one (Image): _description_
        image_two (Image): _description_
        image_one_name (str): _description_
        image_two_name (str): _description_
        metric (Dict): Contains the distance between the 2 faces representations
        save_path (Optional[Path]): path where the figure will be saved
        fig_size (tuple, optional): Size of the figure. Defaults to (20, 20).
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
    
    ax1.imshow(image_one)
    ax2.imshow(image_two)
    
    ax1.set_title(image_one_name, fontsize=14)
    ax2.set_title(image_two_name, fontsize=14)
    
    fig.suptitle(" | ".join(["{}: {:.3f}".format(metric_name, metric_value) 
                             for (metric_name, metric_value) in metric]), fontsize=18)
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    
def predict(image_one_path: Path,
            image_two_path: Path,
            checkpoint_path: Path,
            device: str,
            save_path: Optional[Path],
            fig_size: Tuple[int]=(20, 20)
            ):
    
    facename_one = extract_name_from_face_image_path(image_one_path)
    facename_two = extract_name_from_face_image_path(image_two_path)
    
    image_one = load_image(image_one_path)
    image_two = load_image(image_two_path)
    
    checkpoint_file = torch.load(checkpoint_path, map_location=device)
    embedding_dim = get_embedding_dims_from_checkpoint(checkpoint_file)
    net = FaceNet(embedding_dim=embedding_dim)
    net.eval()
    
    metrics = get_metrics(image_one, image_two, net)
    
    show_predictions(image_one, image_two, facename_one, facename_two, metrics, save_path, fig_size)
    
    
if __name__ == '__main__':
    image_one = "data/lfw/test/Alec_Baldwin/Alec_Baldwin_0003.jpg"
    image_two = "data/lfw/test/Tom_Ridge/Tom_Ridge_0030.jpg"
    checkpoint_path = "checkpoints/epoch_007.ckpt"
    device="cpu"
    
    predict(image_one, image_two, checkpoint_path, device, save_path="outputs/pred_one.png")