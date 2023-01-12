from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from .tools import load_image, get_person_image_paths, get_persons_with_at_least_k_images

class BaseTriplets(Dataset):
    
    def __init__(self, path, transform=None) -> None:
        super().__init__()

        self.person_paths = get_person_image_paths(path)
        self.persons = self.person_paths.keys()
        self.persons_positive = get_persons_with_at_least_k_images(self.person_paths, 2)

        RGB_MEAN = [ 0.485, 0.456, 0.406 ]
        RGB_STD = [ 0.229, 0.224, 0.225 ]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = RGB_MEAN,
                                     std = RGB_STD),
            ])
        else:
            self.transform = transform

    def get_anchor_positive_negative_paths(self, index: int) -> tuple:
        """Randomly sample a triplet of image paths.

        Args:
            index (int): Index of the anchor / positive person.

        Returns:
            tuple[Path]: A triplet of paths (anchor, positive, negative)
        """
        # TODO Please implement this function
        # get the people name
        select_people_name = list(self.persons_positive)[index]

        # select randomly 2 images within the folder
        nb_select_people_images = len(self.person_paths[select_people_name])
        anchor_id, pos_id = random.sample([i for i in range(nb_select_people_images)], 2)
        anchor = self.person_paths[select_people_name][anchor_id]
        pos = self.person_paths[select_people_name][pos_id]

        # select randomly the negative id
        other_people_list = copy.deepcopy(list(self.persons))
        other_people_list.remove(select_people_name)
        neg_people_name = random.sample(other_people_list, 1)[0]
        nb_neg_people_images = len(list(self.person_paths[neg_people_name]))
        neg_id = random.randint(0, nb_neg_people_images-1)
        neg = list(self.person_paths[neg_people_name])[neg_id]

        return anchor, pos, neg

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
            self.dataset = BaseTriplets(args.path_data)

        def test_same_shapes(self):
            a, p, n = self.dataset[0]
            self.assertEqual(a.shape, p.shape, 'inconsistent image sizes')
            self.assertEqual(a.shape, n.shape, 'inconsistent image sizes')

        def test_triplet_paths(self):
            a, p, n = self.dataset.get_anchor_positive_negative_paths(0)
            self.assertEqual(a.parent.name, p.parent.name)
            self.assertNotEqual(a.parent.name, n.parent.name)

    unittest.main(argv=[''], exit=False)

        