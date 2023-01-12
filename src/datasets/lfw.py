from torch.utils.data import Dataset
from .tools import get_person_image_paths, get_persons_with_at_least_k_images
from ..utils.utils import load_image
from torchvision import transforms
from pathlib import Path
import numpy as np
from copy import deepcopy


RGB_MEAN = [ 0.485, 0.456, 0.406 ]
RGB_STD = [ 0.229, 0.224, 0.225 ]


class LFW(Dataset):

    def __init__(self, path: Path, transform: transforms.Compose=None) -> None:
        super().__init__()

        self._person_paths = get_person_image_paths(path)
        self._persons = self.person_paths.keys()
        self._persons_positive = get_persons_with_at_least_k_images(self.person_paths, 2)
        self.transform = transform or transforms.Compose([
                                                            transforms.CenterCrop(size=(150, 150)),
                                                            transforms.Resize((224,224)),  # resized to the network's required input size
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean = RGB_MEAN,
                                                                                std = RGB_STD),
                                                        ])

    def get_anchor_positive_negative_paths(self, index: int) -> tuple:
        """Randomly sample a triplet of image paths.

        Args:
            index (int): Index of the anchor / positive person.

        Returns:
            tuple[Path]: A triplet of paths (anchor, positive, negative)
        """
        # TODO Please implement this function
        # get the people name
        pos_name = list(self._persons_positive)[index]

        # select randomly 2 images within the folder
        nb_pos_images = len(getattr(self._person_paths, pos_name))
        anchor_item, positive_item = np.random.choice(nb_pos_images, 2, replace=False)
        anchor, positive = getattr(self._persons_paths, pos_name)[[anchor_item, positive_item]]

        # select randomly the negative id
        neg_names = deepcopy(*self.persons)
        neg_names.remove(pos_name)
        neg_name = np.random.choice(neg_names, 1)[0]
        nb_neg_images = len(getattr(self.person_paths, neg_name))
        neg_id = np.random.choice(nb_neg_images, 1, replace=False)
        negative =  getattr(self._person_paths, neg_name)[neg_id]

        return anchor, positive, negative


    def __getitem__(self, index: int):
        """Randomly sample a triplet of image tensors.

        Args:
            index (int): Index of the anchor / positive person.

        Returns:
            tuple[Path]: A triplet of tensors (anchor, positive, negative)
        """
        a, p, n = self.get_anchor_positive_negative_paths(index)
        return (
            self.transform(load_image(a)),
            self.transform(load_image(p)),
            self.transform(load_image(n))
        )

    def __len__(self):
        return len(self.persons_positive)


if __name__ == "__main__":
    # This file is supposed to be imported, but you can run it do perform some unittests
    # or other investigations on the dataloading.
    import argparse
    import unittest
    parser = argparse.ArgumentParser()
    parser.add_argument('path_data', type=Path)
    args = parser.parse_args()

    class DatasetTests(unittest.TestCase):
        def setUp(self):
            self.dataset = LFW(args.path_data)

        def test_same_shapes(self):
            a, p, n = self.dataset[0]
            self.assertEqual(a.shape, p.shape, 'inconsistent image sizes')
            self.assertEqual(a.shape, n.shape, 'inconsistent image sizes')

        def test_triplet_paths(self):
            a, p, n = self.dataset.get_anchor_positive_negative_paths(0)
            self.assertEqual(a.parent.name, p.parent.name)
            self.assertNotEqual(a.parent.name, n.parent.name)

    unittest.main(argv=[''], exit=False)