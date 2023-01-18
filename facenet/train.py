import hydra
from typing import List
from pytorch_lightning import Trainer, seed_everything, Callback, LightningModule, LightningDataModule
from pytorch_lightning.loggers.logger import Logger
import logging
from omegaconf import DictConfig
from utils import instantiate_callbacks
import sys

log = logging.getLogger('lightning')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)8s] %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
log.addHandler(console_handler)


def train(cfg: DictConfig) -> None:

    seed_everything(cfg.seed)


    model: LightningModule = hydra.utils.instantiate(cfg.model, default_logger=log)
    log.info(f"Instantiated model <{cfg.model._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule, model=model.net)
    log.info(f"Instantiated datamodule <{cfg.datamodule._target_}>")
    logger: Logger = hydra.utils.instantiate(cfg.logger)
    log.info(f"Created the logger <{cfg.logger._target_}>")
   
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    log.info(f"Instantiated callbacks >")
    
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, 
                                               callbacks=callbacks, 
                                               logger=logger) #logger
    log.info(f"Instantiated trainer <{cfg.trainer._target_}>")

    if cfg.get("train"):
        log.info("Starting training!")
        ckpt_path = cfg.get("checkpoint_path")
        if ckpt_path is None:
            log.warning(f"No previous checkpoint found, Starting from scratch")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    train(cfg)

if __name__ == "__main__":
    main()