import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Union, Dict, Optional, Tuple
from pathlib import Path
from facenet.model import FaceNet
import numpy as np
import torch
import os
from torchvision import transforms
from facenet.utils import (extract_name_from_face_image_path,
                     load_image, 
                     get_embedding_dims_from_checkpoint)


RGB_MEAN = [ 0.485, 0.456, 0.406 ]
RGB_STD = [ 0.229, 0.224, 0.225 ]

def get_metrics(image_one: Image, image_two: Image, facenet: FaceNet,) -> Dict:
    """Gets the distances between two images vector representations

    Args:
        image_one (Image): first image to be compared
        image_two (Image): second image to be compared
        facenet (FaceNet): network in eval mode for prediction

    Returns:
        Dict: Contains the distance between the 2 faces representations
    """
    transform = transforms.Compose([ transforms.CenterCrop(size=(150, 150)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean = RGB_MEAN,
                                                                                std = RGB_STD),
                                                        ])
    with torch.no_grad():
        image_one_output = facenet(transform(image_one)[None, :]).detach().numpy()
        image_two_output = facenet(transform(image_two)[None, :]).detach().numpy()
    
    dist = np.linalg.norm(image_one_output - image_two_output)
    
    return {"Distance": dist}
    


def show_predictions(image_one: Image, 
                     image_two: Image, 
                     image_one_name: str,
                     image_two_name: str,
                     metric: Dict,
                     save_folder: Optional[Path],
                     fig_size: Tuple[int]=(20, 10)
                     ) -> None:
    """_summary_

    Args:
        image_one (Image): _description_
        image_two (Image): _description_
        image_one_name (str): _description_
        image_two_name (str): _description_
        metric (Dict): Contains the distance between the 2 faces representations
        save_path (Optional[Path]): path where the figure will be saved
        fig_size (tuple, optional): Size of the figure. Defaults to (20, 10).
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
    
    ax1.imshow(image_one)
    ax2.imshow(image_two)
    
    ax1.set_title(image_one_name, fontsize=18)
    ax2.set_title(image_two_name, fontsize=18)
    
   
    fig.suptitle(" | ".join(["{}: {:.3f}".format(metric_name, metric_value) for (metric_name, metric_value) in metric.items()]), fontsize=24)
    if save_folder:
        save_path = os.path.join(save_folder, "_".join([image_one_name, image_two_name]) + ".jpg")
        plt.savefig(save_path)
    
    plt.show()
    
    
def predict(image_one_path: Path,
            image_two_path: Path,
            checkpoint_path: Path,
            device: str,
            save_folder: Optional[Path],
            fig_size: Tuple[int]=(20, 10)
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
    
    show_predictions(image_one, image_two, facename_one, facename_two, metrics, save_folder, fig_size)
    
    
if __name__ == '__main__':
    image_one = "data/lfw/test/Alec_Baldwin/Alec_Baldwin_0003.jpg"
    image_two = "data/lfw/test/Tom_Ridge/Tom_Ridge_0030.jpg"
    checkpoint_path = "checkpoints/epoch_008.ckpt"
    device="cpu"
    
    predict(image_one, image_two, checkpoint_path, device, save_folder="outputs/pred_one.png")