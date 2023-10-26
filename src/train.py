import os
import signal
from typing import Optional

from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning_lite.plugins.environments import SLURMEnvironment
from omegaconf import OmegaConf, DictConfig
import hydra
import submitit
import logging

import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger

from util.debugging import check_and_start_debugger

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("sum", lambda x, y: x + y)


@hydra.main(version_base=None, config_path="../config", config_name="train")
def train(cfg: DictConfig):
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
        plugins = SLURMEnvironment(auto_requeue=True, requeue_signal=signal.SIGUSR2)
    else:
        log.info(f"Launching using BasicLauncher")
        plugins = None

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)

    log.info(f"Instantiating lightning module <{cfg.lightning_module._target_}>")
    lightning_module: pl.LightningModule = instantiate(cfg.lightning_module)

    if cfg.trainer.logger:
        log.info(f"Instantiating logger <{cfg.trainer.logger._target_}>")
        wandb_class = "pytorch_lightning.loggers.wandb.WandbLogger"
        is_wandb = cfg.trainer.logger._target_ == wandb_class

        if is_wandb:
            # Set wandb run name to job id
            # This allows us to resume wandb runs after slurm jobs are preempted
            # Somewhat hacky, different jobs may have the same job_id on different clusters
            # Wait for https://github.com/Lightning-AI/lightning/issues/5342 for a better solution

            if "id" in cfg.trainer.logger:
                wandb_run_name = str(cfg.trainer.logger.id)
            elif is_submitit:
                env = submitit.JobEnvironment()
                wandb_run_name = env.job_id
            else:
                wandb_run_name = None

            if wandb_run_name is not None:
                ckpt_path = "last"
                logger: Optional[Logger, None] = instantiate(cfg.trainer.logger, id=wandb_run_name)
            else:
                ckpt_path = None
                logger: Optional[Logger, None] = instantiate(cfg.trainer.logger)
        else:
            ckpt_path = None
            logger: Optional[Logger, None] = instantiate(cfg.trainer.logger)

        config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        logger.log_hyperparams(config)

        if is_wandb and os.environ.get('WANDB_RUN_ID') is None:
            os.environ["WANDB_RUN_ID"] = logger.version

    else:
        ckpt_path = None
        logger = None

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = instantiate(
        cfg.trainer, logger=logger, plugins=plugins
    )

    # Unset submitit signal handler
    # See https://github.com/facebookincubator/submitit/issues/1709#issuecomment-1246758283
    signal.signal(signal.SIGUSR2, signal.SIG_DFL)

    log.info(f"Starting training")
    trainer.fit(model=lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    train()
