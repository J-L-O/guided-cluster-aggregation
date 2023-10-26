from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
import hydra
import submitit
import logging

import pytorch_lightning as pl

from util.evaluators import Evaluator
from lightning_module.gca import GCAModule
from util.debugging import check_and_start_debugger

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("sum", lambda x, y: x + y)


@hydra.main(version_base=None, config_path="../config", config_name="evaluate")
def evaluate(cfg: DictConfig):
    check_and_start_debugger()

    print(OmegaConf.to_yaml(cfg))

    hydra_config = HydraConfig.get()
    submitit_class = (
        "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher"
    )
    is_submitit = hydra_config.launcher._target_ == submitit_class
    if is_submitit:
        env = submitit.JobEnvironment()
        log.info(f"Launching using SlurmLauncher with env {env}")
    else:
        log.info(f"Launching using BasicLauncher")

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)

    log.info(f"Instantiating lightning module <{cfg.lightning_module._target_}>")
    lightning_module: pl.LightningModule = instantiate(cfg.lightning_module)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = instantiate(cfg.trainer)

    log.info(f"Starting prediction")
    predictions = trainer.predict(lightning_module, datamodule=datamodule, ckpt_path=cfg.checkpoint_path)

    train_preds = GCAModule.merge_dl_output(predictions[0])

    for evaluator_cfg in cfg.evaluators:
        log.info(f"Instantiating evaluator <{evaluator_cfg._target_}>")
        evaluator: Evaluator = instantiate(evaluator_cfg)

        evaluator.evaluate(train_preds)


if __name__ == "__main__":
    evaluate()
