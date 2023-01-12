from pathlib import Path
import random 
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import copy


def load_image(path_image: str) -> Image.Image:
    """Load image from harddrive and return 3-channel PIL image.
    Args:
        path_image (str): image path
    Returns:
        Image.Image: loaded image
    """
    return Image.open(path_image).convert('RGB')


def get_person_image_paths(path_person_set: str) -> dict:
    """Creates mapping from person name to list of images.
    Args:
        path_person_set (str): Path to dataset that contains folder of images.
    Returns:
        Dict[str, List]: Mapping from person name to image paths,
                         For instance {'name': ['/path/image1.jpg', '/path/image2.jpg']}
    """
    path_person_set = Path(path_person_set)
    person_paths = filter(Path.is_dir, path_person_set.glob('*'))
    return {
        path.name: list(path.glob('*.jpg')) for path in person_paths
    }


def get_persons_with_at_least_k_images(person_paths: dict, k: int) -> list:
    """Filter persons and return names of those having at least k images
    Args:
        person_paths (dict): dict of persons, as returned by `get_person_image_paths`
        k (int): number of images to filter for

    Returns:
        list: list of filtered person names
    """
    return [name for name, paths in person_paths.items() if len(paths) >= k]