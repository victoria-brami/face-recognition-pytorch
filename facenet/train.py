import hydra
from typing import List
from pytorch_lightning import Trainer, seed_everything, Callback, LightningModule, LightningDataModule
from pytorch_lightning.loggers import LightningLoggerBase
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO) # can customize format


def train(cfg: DictConfig) -> None:

    seed_everything(cfg.seed)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model ... <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating callbacks... <{cfg.callbacks._target_}>")
    callbacks: List[Callback] = hydra.utils.instantiate(cfg.get("callbacks"))

    log.info(f"Creating the logger ...")
    logger:  List[LightningLoggerBase] = hydra.utils.instantiate(cfg.logger)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("checkpoint_path"))

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