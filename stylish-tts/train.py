import time
import torch
import click
import shutil
import logging
import random
from logging import StreamHandler
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from config_loader import load_config_yaml, load_model_config_yaml
from train_context import TrainContext
from text_utils import TextCleaner

import numpy as np

#  warnings.simplefilter("ignore")
from torch.utils.tensorboard.writer import SummaryWriter

from meldataset import build_dataloader, FilePathDataset
from batch_manager import BatchManager
from stage_context import StageContext, is_valid_stage, valid_stage_list

from models.models import build_model, load_defaults
from losses import GeneratorLoss, DiscriminatorLoss, WavLMLoss, MultiResolutionSTFTLoss
from utils import get_data_path_list, save_git_diff
from loss_log import combine_logs
import tqdm
import auraloss

# from models.slmadv import SLMAdversarialLoss
# from models.diffusion.sampler import (
#     DiffusionSampler,
#     ADPM2Sampler,
#     KarrasSchedule,
# )

import os.path as osp
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create a logger for the current module
logger = logging.getLogger(__name__)


def setup_logger(logger, out_dir):
    logger.setLevel(logging.DEBUG)
    # Prevent messages from being passed to the root logger
    logger.propagate = False

    # Always add a stream handler
    err_handler = StreamHandler()
    err_handler.setLevel(logging.DEBUG)
    err_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(err_handler)

    # Always add a file handler
    file_handler = logging.FileHandler(osp.join(out_dir, "train.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)


@click.command()
@click.option("-p", "--config_path", default="configs/new.config.yml", type=str)
@click.option("-cp", "--model_config_path", default="config/model.config.yml", type=str)
@click.option("--out_dir", type=str)
@click.option("--stage", default="first_tma", type=str)
@click.option("--checkpoint", default="", type=str)
def main(config_path, model_config_path, out_dir, stage, checkpoint):
    train = TrainContext()
    np.random.seed(1)
    random.seed(1)
    if osp.exists(config_path):
        train_config = load_config_yaml(config_path)
        train.config = train_config
    else:
        # TODO: we may be able to pull it out of the model if a model is passed in instead
        logger.error(f"Config file not found at {config_path}")
        exit(1)

    if osp.exists(model_config_path):
        train.model_config = load_model_config_yaml(model_config_path)
    else:
        # TODO: we may be able to pull it out of the model if a model is passed in instead
        logger.error(f"Config file not found at {model_config_path}")
        exit(1)

    train.base_output_dir = out_dir
    train.out_dir = osp.join(out_dir, stage)

    if not osp.exists(train.out_dir):
        os.makedirs(train.out_dir, exist_ok=True)

    if not osp.exists(train.out_dir):
        exit(f"Failed to create or find log directory at {train.out_dir}.")
    shutil.copy(config_path, osp.join(train.out_dir, osp.basename(config_path)))
    shutil.copy(
        model_config_path, osp.join(train.out_dir, osp.basename(model_config_path))
    )
    save_git_diff(train.out_dir)

    train.logger = logging.getLogger(__name__)
    setup_logger(train.logger, train.out_dir)

    ddp_kwargs = DistributedDataParallelKwargs(
        broadcast_buffers=False, find_unused_parameters=True
    )
    train.accelerator = Accelerator(
        project_dir=train.out_dir,
        split_batches=True,
        kwargs_handlers=[ddp_kwargs],
        mixed_precision=train.config.training.mixed_precision,
        step_scheduler_with_optimizer=False,
    )
    train.accelerator.even_batches = False

    if train.accelerator.is_main_process:
        train.writer = SummaryWriter(train.out_dir + "/tensorboard")

    train.accelerator.register_for_checkpointing(train.config)
    train.accelerator.register_for_checkpointing(train.model_config)
    train.accelerator.register_for_checkpointing(train.manifest)

    # Set up data loaders and batch manager
    if not osp.exists(train.config.dataset.train_data):
        exit(f"Train data not found at {train.config.dataset.train_data}")
    if not osp.exists(train.config.dataset.val_data):
        exit(f"Validation data not found at {train.config.dataset.val_data}")
    if not osp.exists(train.config.dataset.wav_path):
        exit(f"Root path not found at {train.config.dataset.wav_path}")

    text_cleaner = TextCleaner(train.model_config.symbol)
    val_list = get_data_path_list(train.config.dataset.val_data)
    val_dataset = FilePathDataset(
        val_list,
        train.config.dataset.wav_path,
        OOD_data=train.config.dataset.OOD_data,
        min_length=train.config.dataset.min_length,
        validation=True,
        multispeaker=train.model_config.model.multispeaker,
        text_cleaner=text_cleaner,
        pitch_path=train.config.dataset.pitch_path,
    )
    train.val_dataloader = build_dataloader(
        val_dataset,
        val_dataset.time_bins(),
        validation=True,
        num_workers=4,
        device=train.config.training.device,
        multispeaker=train.model_config.model.multispeaker,
        train=train,
    )
    train.val_dataloader = train.accelerator.prepare(train.val_dataloader)

    train.batch_manager = BatchManager(
        train.config.dataset,
        train.out_dir,
        probe_batch_max=train.config.training.probe_batch_max,
        device=train.config.training.device,
        accelerator=train.accelerator,
        multispeaker=train.model_config.model.multispeaker,
        text_cleaner=text_cleaner,
        stage=stage,
        epoch=train.manifest.current_epoch,
        train=train,
    )

    # build model
    train.model = build_model(train.model_config)
    for key in train.model:
        train.model[key] = train.accelerator.prepare(train.model[key])
        train.model[key].to(train.config.training.device)

    train.generator_loss = GeneratorLoss(train.model.mpd, train.model.msd).to(
        train.config.training.device
    )
    train.discriminator_loss = DiscriminatorLoss(train.model.mpd, train.model.msd).to(
        train.config.training.device
    )
    train.wavlm_loss = WavLMLoss(
        train.model_config.slm.model,
        None,
        # train.model.wd,
        train.model_config.preprocess.sample_rate,
        train.model_config.slm.sr,
    ).to(train.config.training.device)

    if not is_valid_stage(stage):
        exit(f"{stage} is not a valid stage. Must be one of {valid_stage_list()}")
    train.stage = StageContext()
    train.stage.begin_stage(stage, train)

    train.manifest.current_epoch = 1
    train.manifest.current_total_step = 0
    should_fast_forward = False

    if checkpoint:
        train.accelerator.load_state(checkpoint)
        train.config = train_config
        # if we are not loading on a epoch boundary we need to resume the loader and skip to the correct step
        if train.manifest.stage == stage:
            if train.manifest.current_step != 0:
                should_fast_forward = True
        else:
            train.manifest.current_epoch = 1
            train.manifest.current_step = 0
            train.stage.begin_stage(stage, train)
        # if train.manifest.stage == "acoustic" and stage == "textual":
        #     logger.info("Cloning style encoder into prosodic style encoder...")
        #     train.model.prosodic_style_encoder.load_state_dict(
        #         train.model.style_encoder.state_dict()
        #     )
        logger.info(f"Loaded last checkpoint at {checkpoint} ...")
    else:
        load_defaults(train, train.model)

    train.manifest.stage = stage

    # TODO: How to access model diffusion?
    # train.diffusion_sampler = DiffusionSampler(
    #     kdiffusion,
    #     sampler=ADPM2Sampler(),
    #     sigma_schedule=KarrasSchedule(
    #         sigma_min=0.0001, sigma_max=3.0, rho=9.0
    #     ),  # empirical parameters
    #     clamp=False,
    # )

    train.n_down = 1  # TODO: Use train.model.text_aligner.n_down

    train.manifest.best_loss = float("inf")  # best test loss

    torch.cuda.empty_cache()

    # train.stft_loss = MultiResolutionSTFTLoss().to(train.config.training.device)
    train.stft_loss = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 60],
        win_lengths=[600, 1200, 300],
        scale="mel",
        n_bins=120,
        sample_rate=24000,
        perceptual_weighting=True,
        w_lin_mag=0.1,
        w_phs=0.1,
    )

    # TODO: This value is calculated inconsistently based on whether checkpoints are loaded/saved
    train.manifest.running_std = []

    # train.slm_adversarial_loss = SLMAdversarialLoss(
    #     train.model,
    #     train.wavlm_loss,
    #     train.diffusion_sampler,
    #     train.model_config.slmadv_params.min_len,
    #     train.model_config.slmadv_params.max_len,
    #     batch_percentage=train.model_config.slmadv_params.batch_percentage,
    #     skip_update=train.model_config.slmadv_params.iter,
    #     sig=train.model_config.slmadv_params.sig,
    # )

    if not train.stage.batch_sizes_exist():
        train.batch_manager.probe_loop(train)
        should_fast_forward = False

    train_val_loop(train, should_fast_forward=should_fast_forward)
    train.accelerator.end_training()


def train_val_loop(train: TrainContext, should_fast_forward=False):
    logs = []
    while train.manifest.current_epoch <= train.stage.max_epoch:
        train.batch_manager.init_epoch(train, should_fast_forward=should_fast_forward)
        train.stage.steps_per_epoch = train.batch_manager.get_step_count()
        train.running_loss = 0
        train.start_time = time.time()

        # TODO: This line should be obsolete soon
        _ = [train.model[key].train() for key in train.model]
        progress_bar = None
        if train.accelerator.is_main_process:
            iterator = tqdm.tqdm(
                iterable=enumerate(train.batch_manager.loader),
                desc=f"Train [{train.manifest.current_epoch}/{train.stage.max_epoch}]",
                total=train.stage.steps_per_epoch,
                unit="steps",
                initial=train.manifest.current_step,
                bar_format="{desc}{bar}| {n_fmt}/{total_fmt} {remaining}{postfix} ",
                colour="GREEN",
                delay=5,
                leave=False,
                dynamic_ncols=True,
            )
            progress_bar = iterator
        else:
            iterator = enumerate(train.batch_manager.loader)
        loss = None
        for _, batch in iterator:
            next_log = train.batch_manager.train_iterate(
                batch, train, progress_bar=progress_bar
            )
            train.manifest.current_total_step += 1
            train.manifest.current_step += 1
            train.manifest.total_trained_audio_seconds += (
                float(len(batch[0][0]) * len(batch[0]))
                / train.model_config.preprocess.sample_rate
            )
            if train.accelerator.is_main_process:
                if next_log is not None:
                    logs.append(next_log)
                    if loss is None:
                        loss = next_log.metrics["mel"]
                    loss = loss * 0.9 + next_log.metrics["mel"] * 0.1
                if len(logs) >= train.config.training.log_interval:
                    progress_bar.clear() if progress_bar is not None else None
                    combine_logs(logs).broadcast(train.manifest, train.stage)
                    progress_bar.display() if progress_bar is not None else None
                    logs = []
            num = train.manifest.current_total_step
            val_step = train.config.training.val_interval
            save_step = train.config.training.save_interval
            do_val = num % val_step == 0
            do_save = num % save_step == 0
            next_val = val_step - num % val_step - 1
            next_save = save_step - num % save_step - 1
            postfix = {"mel_loss": f"{loss:.3f}"}
            if next_val < next_save:
                postfix["val"] = str(next_val)
            else:
                postfix["save"] = str(next_save)
            progress_bar.set_postfix(postfix) if progress_bar is not None else None
            if do_val or do_save:
                progress_bar.clear() if progress_bar is not None else None
                train.stage.validate(train)
                progress_bar.display() if progress_bar is not None else None
            if do_save:
                progress_bar.clear() if progress_bar is not None else None
                save_checkpoint(train, prefix="checkpoint")
                progress_bar.display() if progress_bar is not None else None
        if len(logs) > 0:
            combine_logs(logs).broadcast(train.manifest, train.stage)
            logs = []
        train.manifest.current_epoch += 1
        train.manifest.current_step = 0
        train.manifest.training_log.append(
            f"Completed 1 epoch of {train.manifest.stage} training"
        )
        progress_bar.close() if progress_bar is not None else None
    train.stage.validate(train)
    save_checkpoint(train, prefix="checkpoint_final", long=False)


def save_checkpoint(
    train: TrainContext,
    prefix: str = "checkpoint",
    long: bool = True,
) -> None:
    """
    Saves checkpoint using a checkpoint.
    """
    logger.info("Saving...")
    checkpoint_dir = osp.join(train.out_dir, f"{prefix}")
    if long:
        checkpoint_dir += f"_{train.manifest.current_epoch:05d}_step_{train.manifest.current_total_step:09d}"

    # Let the accelerator save all model/optimizer/LR scheduler/rng states
    train.accelerator.save_state(checkpoint_dir, safe_serialization=False)

    logger.info(f"Saved checkpoint to {checkpoint_dir}")


if __name__ == "__main__":
    main()
