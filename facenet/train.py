import hydra
from typing import List
from pytorch_lightning import Trainer, seed_everything, Callback, LightningModule, LightningDataModule
from pytorch_lightning.loggers import LightningLoggerBase
import logging
from omegaconf import DictConfig
from utils import instantiate_callbacks

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
) # can customize format


def train(cfg: DictConfig) -> None:

    seed_everything(cfg.seed)

    
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    log.info(f"Instantiated datamodule <{cfg.datamodule._target_}>")

    model: LightningModule = hydra.utils.instantiate(cfg.model)
    log.info(f"Instantiated model <{cfg.model._target_}>")

    logger:  LightningLoggerBase = hydra.utils.instantiate(cfg.logger)
    log.info(f"Created the logger <{cfg.logger._target_}>")
   
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    log.info(f"Instantiated callbacks >")
    
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
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