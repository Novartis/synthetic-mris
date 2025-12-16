import os
import sys

import hydra
import pytorch_lightning as ptl
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict

# source path for local functions, model architecture
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(SRC_PATH)

from helpers.evaluate_utils import list_checkpoints


@hydra.main(
    config_path=os.path.abspath(os.path.join(SRC_PATH, "..", "conf")),
    config_name="default",
)
@logger.catch
def main_train(cfg: DictConfig):
    assert os.path.isdir(cfg.data_basedir) and os.path.isdir(cfg.runs_basedir), (
        f"Ensure data and runs basedirectories have been passed -- use hydra arg overrides for these: python {__file__} data_basedir=<...> runs_basedir=<...>"
    )
    torch.set_float32_matmul_precision(cfg.train.torch_matmul_precision)
    ptl.seed_everything(cfg.seed)

    # check if we have to continue the training:
    if "continue_train" in cfg.train and cfg.train.continue_train:
        checkpoint_path = os.path.join(
            cfg.train.logger.save_dir, cfg.train.logger.name, cfg.train.logger.version
        )  # TODO: better naming here, this is manual for the moment and needs to be in sync with main_evaluate
        trained_version = cfg.train.logger.version
        temp = OmegaConf.load(os.path.join(checkpoint_path, "hparams.yaml"))
        cfg = temp.cfg
        with open_dict(cfg):
            cfg.train.continue_train = True
            cfg.train.logger.version = trained_version
            cfg.train.trainer.max_epochs = -1
            cfg.train.pretrained_model_weights = None

    logger.info(f"Training data:   {cfg.data.paths.training_data}")
    logger.info(f"Job directory:   {cfg.job_directory}")
    logger.info(f"training using model from: {cfg.model.net}")
    logger.info(f"batch size: {cfg.train.optimizers.batch_size}")

    if "loss_fcn" in cfg.model.architecture:
        logger.info(f"loss function uses: {cfg.model.architecture.loss_fcn}")
        if cfg.model.architecture.loss_fcn == "all":
            logger.info(f"weight for gen loss:  {cfg.model.architecture.g_high_res_loss_weight}")
            logger.info(f"weight for disc loss: {cfg.model.architecture.d_high_res_loss_weight}")

    if "train_generators" in cfg.model.architecture:
        logger.info(f"training generators: {cfg.model.architecture.train_generators}")

    # load model
    model = hydra.utils.instantiate(cfg.model.net, cfg=cfg, _recursive_=False)

    # load pretrained weights if supplied
    if "pretrained_model_weights" in cfg.train and cfg.train.pretrained_model_weights is not None:
        state_dict = torch.load(cfg.train.pretrained_model_weights)["state_dict"]
        model.load_state_dict(state_dict, strict=True)
        logger.info(f"load pretrained model from {cfg.train.pretrained_model_weights}")

    # instiantiate trainer
    # learn more on trainer flags here:
    # https://lightning.ai/docs/pytorch/stable/common/trainer.html#
    trainer = hydra.utils.instantiate(cfg.train.trainer)

    # set trainer to save model for all checkpoints
    trainer.checkpoint_callback.save_top_k = cfg.train.checkpoint_callback_save_top_k

    # configure logger
    if cfg.train.logger:
        trainer.logger = hydra.utils.instantiate(cfg.train.logger)
        logger.info("to open tensorboard use the following command in the terminal:")
        logger.info(f"tensorboard --logdir={cfg.train.logger.save_dir}")

    if cfg.train.continue_train:
        checkpoint_path = os.path.join(cfg.train.logger.save_dir, cfg.train.logger.name, cfg.train.logger.version)
        checkpoints = list_checkpoints(checkpoint_path)
        if not checkpoints:
            raise ValueError(f"No checkpoints found at / under {checkpoint_path}")
        max_epoch = max(
            p[1] for p in checkpoints if isinstance(p[1], int)
        )  # note multiple versions (ie. ../epoch=1490-step=2495934-v1.ckpt) is not handled in list_checkpoints
        path_latest_checkpoint = next(p[0] for p in checkpoints if p[1] == max_epoch)
        logger.info(f"Loading checkpoint from epoch {max_epoch} -> {path_latest_checkpoint}:")
        state_dict = torch.load(path_latest_checkpoint)["state_dict"]
        model.load_state_dict(state_dict, strict=True)
        trainer.fit(model, ckpt_path=path_latest_checkpoint)
    else:
        logger.info("Start training from scratch...............")
        trainer.fit(model)


if __name__ == "__main__":
    main_train()  # pylint: disable=no-value-for-parameter
