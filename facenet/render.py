import hydra
from typing import List
import logging
from omegaconf import DictConfig
from render import predict


def test_face_recognition(predict_cfg: DictConfig) -> None:
    """
    """
    image_one_path = predict_cfg.get('image_one_path')
    image_two_path = predict_cfg.get('image_two_path')

    save_path = predict_cfg.get('save_folder')

    predict(image_one_path, image_two_path, predict_cfg.checkpoint_path, predict_cfg.device, save_path)



@hydra.main(version_base=None, config_path="../configs", config_name="render.yaml")
def main(cfg: DictConfig) -> None:
    test_face_recognition(cfg)

if __name__ == "__main__":
    main()