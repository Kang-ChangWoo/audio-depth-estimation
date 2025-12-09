from dataloader.BatvisionV1_Dataset import BatvisionV1Dataset
from dataloader.BatvisionV2_Dataset import BatvisionV2Dataset

from models.utils_models import *
from models.unet_cvae_model import define_G_cvae

from utils_criterion import compute_errors
from utils_visualization import save_batch_visualization
from utils_loss import SIlogLoss

import time
import os
import numpy as np
import math
import pickle

import torch
from torch.utils.data import DataLoader
import argparse

from config_loader import load_config
import wandb

WANDB_AVAILABLE = True


def main():
    parser = argparse.ArgumentParser(description="Train U-Net+cVAE model on Batvision dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="batvisionv2",
        choices=["batvisionv1", "batvisionv2"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="cvae",
        help="Name of the experiment (suffix, will be combined with cfg)",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        default=None,
        help="Checkpoint epoch to resume from (None to start from scratch)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config, paper: 256)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config, paper: 0.002 for BV2, 0.001 for BV1)",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", default=False, help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="batvision-depth-estimation",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="W&B entity/team name (optional)"
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B logging mode",
    )
    # Loss / optimizer
    parser.add_argument(
        "--criterion",
        type=str,
        default=None,
        choices=["L1", "SIlog", "Combined"],
        help="Loss function (overrides config)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["Adam", "AdamW", "SGD"],
        help="Optimizer (overrides config)",
    )
    parser.add_argument(
        "--silog_lambda",
        type=float,
        default=None,
        help="SIlog lambda parameter (overrides config)",
    )
    parser.add_argument(
        "--l1_weight",
        type=float,
        default=None,
        help="L1 weight for combined loss (overrides config)",
    )
    parser.add_argument(
        "--silog_weight",
        type=float,
        default=None,
        help="SIlog weight for combined loss (overrides config)",
    )
    parser.add_argument(
        "--audio_format",
        type=str,
        default=None,
        choices=["spectrogram", "mel_spectrogram", "waveform"],
        help="Audio format: spectrogram, mel_spectrogram, or waveform (overrides config)",
    )
    parser.add_argument(
        "--validation",
        type=lambda x: (str(x).lower() == "true"),
        default=None,
        help="Enable validation (True/False, overrides config)",
    )
    parser.add_argument(
        "--validation_iter",
        type=int,
        default=None,
        help="Validation frequency: evaluate every N epochs (overrides config)",
    )
    # cVAE-specific
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=1e-4,
        help="Weight for KL divergence term in total loss",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=128,
        help="Latent dimension for VAE bottleneck",
    )

    args = parser.parse_args()

    # We do not support wandb sweeps in this simplified cVAE script;
    # reuse the same flags but skip sweep-specific logic for clarity.

    # Load configuration
    cfg = load_config(dataset_name=args.dataset, mode="train", experiment_name=args.experiment_name)

    # Adapt experiment name to indicate cVAE usage
    cfg.mode.experiment_name = cfg.mode.experiment_name + "_cvae"

    # Override checkpoint if provided
    if args.checkpoints is not None:
        cfg.mode.checkpoints = args.checkpoints

    # Override batch size and learning rate based on arguments
    if args.batch_size is not None:
        cfg.mode.batch_size = args.batch_size
    elif cfg.mode.batch_size != 256:
        print(f"Note: Paper uses batch_size=256, current config has {cfg.mode.batch_size}")

    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {args.learning_rate}")
        if args.learning_rate > 0.1:
            raise ValueError(
                f"ERROR: Learning rate {args.learning_rate} exceeds safe maximum (0.1). "
                f"This will cause training instability and model divergence."
            )
        cfg.mode.learning_rate = args.learning_rate
        print(f"Learning rate set to: {cfg.mode.learning_rate}")
    else:
        if args.dataset == "batvisionv2" and cfg.mode.learning_rate != 0.002:
            print(
                f"Note: Paper uses learning_rate=0.002 for BV2, current config has {cfg.mode.learning_rate}"
            )
        elif args.dataset == "batvisionv1" and cfg.mode.learning_rate != 0.001:
            print(
                f"Note: Paper uses learning_rate=0.001 for BV1, current config has {cfg.mode.learning_rate}"
            )

    working_dir = os.getcwd()
    print(f"The current working directory is {working_dir}")

    if cfg.mode.mode != "train":
        raise Exception("This script is for training only. Please run test.py for evaluation")
    if cfg.model.name != "unet_baseline":
        raise Exception("This script is for training on unet model only (cVAE variant)")

    # ------------ GPU config ------------
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        print(f"CUDA_VISIBLE_DEVICES is set to: {cuda_visible}")

    if torch.cuda.is_available():
        n_GPU = torch.cuda.device_count()
        max_gpus = min(n_GPU, 4)
        gpu_ids = list(range(max_gpus))
        device = torch.device(f"cuda:{gpu_ids[0]}")
        print(f"{n_GPU} GPU(s) available, using {len(gpu_ids)} GPU(s): {gpu_ids}")
        print(f"Using device: {device}")
        if n_GPU > 0:
            print(f"GPU 0 name: {torch.cuda.get_device_name(0)}")
        if n_GPU > 4:
            print(
                "Note: Limited to 4 GPUs to avoid peer mapping issues. "
                "Use CUDA_VISIBLE_DEVICES to select specific GPUs."
            )
    else:
        n_GPU = 0
        gpu_ids = []
        device = torch.device("cpu")
        print("WARNING: CUDA not available, using CPU")

    batch_size = cfg.mode.batch_size

    # ------------ Create experiment name -----------
    experiment_name = (
        cfg.model.generator
        + "_"
        + cfg.dataset.name
        + "_"
        + "BS"
        + str(cfg.mode.batch_size)
        + "_"
        + "Lr"
        + str(cfg.mode.learning_rate)
        + "_"
        + cfg.mode.optimizer
        + "_"
        + cfg.mode.experiment_name
        + "_cvae"
    )

    # ------------ Create dataset -----------
    if cfg.dataset.name == "batvisionv1":
        train_set = BatvisionV1Dataset(cfg, cfg.dataset.annotation_file_train)
        if cfg.mode.validation:
            val_set = BatvisionV1Dataset(cfg, cfg.dataset.annotation_file_val)
    elif cfg.dataset.name == "batvisionv2":
        train_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_train)
        if cfg.mode.validation:
            val_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_val)
    else:
        raise Exception("Training can be done only on BV1 and BV2")

    print(f"Train Dataset of {len(train_set)} instances")
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=cfg.mode.shuffle,
        num_workers=cfg.mode.num_threads,
    )

    if cfg.mode.validation:
        print(f"Validation Dataset of {len(val_set)} instances")
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=cfg.mode.shuffle,
            num_workers=cfg.mode.num_threads,
        )

    # ---------- Load cVAE Model ----------
    model = define_G_cvae(
        cfg,
        input_nc=2,
        output_nc=1,
        ngf=64,
        netG=cfg.model.generator,
        norm="batch",
        use_dropout=False,
        init_type="normal",
        init_gain=0.02,
        gpu_ids=gpu_ids,
        latent_dim=args.latent_dim,
    )
    print("Model used (cVAE):", cfg.model.generator)
    if len(gpu_ids) > 1:
        print(f"Using DataParallel on {len(gpu_ids)} GPUs: {gpu_ids}")

    # ---------- Criterion & Optimizers ----------
    max_depth = cfg.dataset.max_depth if cfg.dataset.max_depth else 30.0

    # Override config with command line arguments
    if args.criterion is not None:
        cfg.mode.criterion = args.criterion
    if args.optimizer is not None:
        cfg.mode.optimizer = args.optimizer
    if args.silog_lambda is not None:
        cfg.mode.silog_lambda = args.silog_lambda
    if args.l1_weight is not None:
        cfg.mode.l1_weight = args.l1_weight
    if args.silog_weight is not None:
        cfg.mode.silog_weight = args.silog_weight
    if args.audio_format is not None:
        if args.dataset == "batvisionv1" and args.audio_format == "mel_spectrogram":
            raise ValueError(
                "mel_spectrogram is not supported for batvisionv1. "
                "Use 'spectrogram' or 'waveform' instead."
            )
        cfg.dataset.audio_format = args.audio_format
        print(f"Audio format overridden to: {args.audio_format}")

    # Loss function setup
    if cfg.mode.criterion == "L1":
        criterion = nn.L1Loss().to(device)
        l1_criterion = None
        silog_criterion = None
        l1_weight = 0.0
        silog_weight = 0.0
        print("Using loss function: L1")
    elif cfg.mode.criterion == "SIlog":
        lambda_scale = getattr(cfg.mode, "silog_lambda", 0.5)
        criterion = SIlogLoss(lambda_scale=lambda_scale).to(device)
        l1_criterion = None
        silog_criterion = None
        l1_weight = 0.0
        silog_weight = 0.0
        print(f"Using loss function: SIlog (lambda={lambda_scale})")
    elif cfg.mode.criterion == "Combined":
        l1_weight = getattr(cfg.mode, "l1_weight", 0.5)
        silog_weight = getattr(cfg.mode, "silog_weight", 0.5)
        silog_lambda = getattr(cfg.mode, "silog_lambda", 0.5)
        l1_criterion = nn.L1Loss().to(device)
        silog_criterion = SIlogLoss(lambda_scale=silog_lambda).to(device)
        criterion = None
        print(
            f"Using loss function: Combined (L1={l1_weight}, "
            f"SIlog={silog_weight}, lambda={silog_lambda})"
        )
    else:
        raise ValueError(
            f"Unknown criterion: {cfg.mode.criterion}. " f"Available: L1, SIlog, Combined"
        )

    learning_rate = cfg.mode.learning_rate
    if cfg.mode.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif cfg.mode.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif cfg.mode.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.mode.optimizer}")

    # ---------- Experiment Setup ----------
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=experiment_name,
            mode=args.wandb_mode,
            config={
                "model": cfg.model.generator,
                "dataset": cfg.dataset.name,
                "max_depth": cfg.dataset.max_depth,
                "depth_norm": cfg.dataset.depth_norm,
                "images_size": cfg.dataset.images_size,
                "audio_format": cfg.dataset.audio_format,
                "batch_size": cfg.mode.batch_size,
                "learning_rate": cfg.mode.learning_rate,
                "optimizer": cfg.mode.optimizer,
                "criterion": cfg.mode.criterion,
                "epochs": cfg.mode.epochs,
                "num_threads": cfg.mode.num_threads,
                "shuffle": cfg.mode.shuffle,
                "silog_lambda": getattr(cfg.mode, "silog_lambda", 0.5)
                if cfg.mode.criterion in ["SIlog", "Combined"]
                else None,
                "l1_weight": getattr(cfg.mode, "l1_weight", None)
                if cfg.mode.criterion == "Combined"
                else None,
                "silog_weight": getattr(cfg.mode, "silog_weight", None)
                if cfg.mode.criterion == "Combined"
                else None,
                "validation": cfg.mode.validation,
                "validation_iter": cfg.mode.validation_iter if cfg.mode.validation else None,
                "num_gpus": len(gpu_ids),
                "gpu_ids": gpu_ids,
                "device": str(device),
                "kl_weight": args.kl_weight,
                "latent_dim": args.latent_dim,
            },
            tags=[cfg.dataset.name, cfg.model.generator, cfg.mode.criterion, cfg.mode.optimizer, "cVAE"],
        )
        print(f"W&B initialized: project={args.wandb_project}, run={experiment_name}")

    # Create logs and results directories
    log_dir = "./logs/" + experiment_name + "/"
    results_dir = "./results/" + experiment_name + "/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    file = open(os.path.join(log_dir, "architecture.txt"), "w")

    file.write("Dataset name: {}\n".format(cfg.dataset.name))
    file.write("Batch size: {}\n".format(batch_size))
    file.write("Image processing: {}\n".format(cfg.dataset.preprocess))
    file.write("Image resize: {}\n".format(cfg.dataset.images_size))
    file.write("Depth norm: {}\n".format(cfg.dataset.depth_norm))
    file.write("Audio type used for training: {}\n".format(cfg.dataset.audio_format))
    file.write("Learning rate: {}\n".format(cfg.mode.learning_rate))
    file.write("Optimizer used : {}\n".format(cfg.mode.optimizer))
    file.write("Generator: {}\n".format(cfg.model.generator))
    file.write("Latent dim (VAE): {}\n".format(args.latent_dim))
    file.write("KL weight: {}\n".format(args.kl_weight))
    file.write(str(model))
    file.close()

    # ---------- Checkpoints ----------
    if cfg.mode.checkpoints is None:
        checkpoint_epoch = 1
    else:
        load_epoch = cfg.mode.checkpoints
        checkpoint = torch.load(
            "./checkpoints/" + experiment_name + "/checkpoint_" + str(load_epoch) + ".pth"
        )
        model.load_state_dict(checkpoint["state_dict"])
        checkpoint_epoch = checkpoint["epoch"] + 1

    nb_epochs = cfg.mode.epochs
    for param_group in optimizer.param_groups:
        print("Learning rate used: {}".format(param_group["lr"]))

    kl_weight = args.kl_weight
    train_iter = 0
    for epoch in range(checkpoint_epoch, nb_epochs + 1):
        t0 = time.time()
        batch_loss = []
        batch_loss_val = []

        # ------ Training ---------
        model.train()
        for i, (audio, gtdepth) in enumerate(train_loader):
            audio = audio.to(device)
            gtdepth = gtdepth.to(device)

            optimizer.zero_grad()

            # forward (cVAE returns depth and KL)
            depth_pred, kl = model(audio)
            # When using DataParallel, each replica returns a scalar KL;
            # DataParallel gathers them into a vector. Reduce to a scalar.
            if isinstance(kl, torch.Tensor) and kl.dim() > 0:
                kl = kl.mean()

            valid_mask = gtdepth > 0

            if cfg.dataset.depth_norm:
                depth_pred_denorm = depth_pred[valid_mask] * cfg.dataset.max_depth
                gtdepth_denorm = gtdepth[valid_mask] * cfg.dataset.max_depth

                if cfg.mode.criterion == "Combined":
                    depth_loss = l1_weight * l1_criterion(
                        depth_pred_denorm, gtdepth_denorm
                    ) + silog_weight * silog_criterion(depth_pred_denorm, gtdepth_denorm)
                else:
                    depth_loss = criterion(depth_pred_denorm, gtdepth_denorm)
            else:
                if cfg.mode.criterion == "Combined":
                    depth_loss = l1_weight * l1_criterion(
                        depth_pred[valid_mask], gtdepth[valid_mask]
                    ) + silog_weight * silog_criterion(
                        depth_pred[valid_mask], gtdepth[valid_mask]
                    )
                else:
                    depth_loss = criterion(depth_pred[valid_mask], gtdepth[valid_mask])

            loss = depth_loss + kl_weight * kl
            batch_loss.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_iter += 1

        epoch_time = time.time() - t0
        if len(batch_loss) > 0:
            train_loss = np.mean(batch_loss)
            print(
                f"Epoch {epoch}: Train Loss: {train_loss:.6f}, "
                f"Time: {epoch_time:.1f}s, KL weight: {kl_weight}"
            )
            if WANDB_AVAILABLE and args.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/epoch_time": epoch_time,
                    },
                    step=epoch,
                )
        else:
            print(f"Epoch {epoch}: No valid training data, Time: {epoch_time:.1f}s")

        # ------- Validation ------------
        if cfg.mode.validation and epoch % cfg.mode.validation_iter == 0:
            model.eval()
            errors = []
            val_preds = []
            val_gts = []
            with torch.no_grad():
                for batch_idx, (audio_val, gtdepth_val) in enumerate(val_loader):
                    audio_val = audio_val.to(device)
                    gtdepth_val = gtdepth_val.to(device)

                    depth_pred_val, kl_val = model(audio_val)
                    if isinstance(kl_val, torch.Tensor) and kl_val.dim() > 0:
                        kl_val = kl_val.mean()

                    valid_mask_val = gtdepth_val > 0

                    if cfg.dataset.depth_norm:
                        depth_pred_val_denorm = (
                            depth_pred_val[valid_mask_val] * cfg.dataset.max_depth
                        )
                        gtdepth_val_denorm = gtdepth_val[valid_mask_val] * cfg.dataset.max_depth

                        if cfg.mode.criterion == "Combined":
                            loss_val = l1_weight * l1_criterion(
                                depth_pred_val_denorm, gtdepth_val_denorm
                            ) + silog_weight * silog_criterion(
                                depth_pred_val_denorm, gtdepth_val_denorm
                            )
                        else:
                            loss_val = criterion(depth_pred_val_denorm, gtdepth_val_denorm)
                    else:
                        if cfg.mode.criterion == "Combined":
                            loss_val = l1_weight * l1_criterion(
                                depth_pred_val[valid_mask_val], gtdepth_val[valid_mask_val]
                            ) + silog_weight * silog_criterion(
                                depth_pred_val[valid_mask_val], gtdepth_val[valid_mask_val]
                            )
                        else:
                            loss_val = criterion(
                                depth_pred_val[valid_mask_val], gtdepth_val[valid_mask_val]
                            )
                    batch_loss_val.append(loss_val.item())

                    # Store first batch for visualization
                    if batch_idx == 0:
                        if cfg.dataset.depth_norm:
                            val_preds.append(depth_pred_val * cfg.dataset.max_depth)
                            val_gts.append(gtdepth_val * cfg.dataset.max_depth)
                        else:
                            val_preds.append(depth_pred_val)
                            val_gts.append(gtdepth_val)

                    for idx in range(depth_pred_val.shape[0]):
                        gt_map = gtdepth_val[idx].cpu().numpy()
                        pred_map = depth_pred_val[idx].cpu().numpy()

                        if gt_map.ndim == 3:
                            gt_map = gt_map[0]
                        if pred_map.ndim == 3:
                            pred_map = pred_map[0]

                        if cfg.dataset.depth_norm:
                            gt_map = gt_map * cfg.dataset.max_depth
                            pred_map = pred_map * cfg.dataset.max_depth

                        epsilon = 1e-3 if cfg.dataset.depth_norm else 1e-6
                        pred_map = np.clip(pred_map, epsilon, cfg.dataset.max_depth)
                        gt_map = np.maximum(gt_map, 0.0)

                        error_metrics = compute_errors(gt_map, pred_map, min_depth_threshold=0.0)
                        errors.append(error_metrics)

                mean_errors = np.array(errors).mean(0)
                val_loss = np.mean(batch_loss_val)
                (
                    abs_rel,
                    rmse,
                    delta1,
                    delta2,
                    delta3,
                    log10,
                    mae,
                ) = (
                    mean_errors[0],
                    mean_errors[1],
                    mean_errors[2],
                    mean_errors[3],
                    mean_errors[4],
                    mean_errors[5],
                    mean_errors[6],
                )

                print(
                    f"Val - Loss: {val_loss:.6f}, RMSE: {rmse:.3f}, ABS_REL: {abs_rel:.3f}, "
                    f"Log10: {log10:.3f}, Delta1: {delta1:.3f}, Delta2: {delta2:.3f}, "
                    f"Delta3: {delta3:.3f}"
                )

                log_dict = {}
                if WANDB_AVAILABLE and args.use_wandb:
                    log_dict = {
                        "epoch": epoch,
                        "val/loss": val_loss,
                        "val/abs_rel": abs_rel,
                        "val/rmse": rmse,
                        "val/log10": log10,
                        "val/delta1": delta1,
                        "val/delta2": delta2,
                        "val/delta3": delta3,
                        "val/mae": mae,
                    }

                if len(val_preds) > 0:
                    pred_batch = torch.cat(val_preds, dim=0)
                    gt_batch = torch.cat(val_gts, dim=0)
                    vis_path = os.path.join(
                        results_dir, f"epoch_{epoch:04d}_validation.png"
                    )
                    save_batch_visualization(
                        pred_batch, gt_batch, vis_path, epoch, num_samples=min(4, pred_batch.shape[0])
                    )
                    print(f"Validation visualization saved to: {vis_path}")

                    if WANDB_AVAILABLE and args.use_wandb:
                        log_dict["val/visualization"] = wandb.Image(
                            vis_path, caption=f"Epoch {epoch}"
                        )

                if WANDB_AVAILABLE and args.use_wandb and log_dict:
                    wandb.log(log_dict, step=epoch)

        # ------- Save ------------
        if epoch % cfg.mode.saving_checkpoints == 0:
            print("Save network")
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            path_check = "./checkpoints/" + experiment_name + "/"
            isExist = os.path.exists(path_check)
            if not isExist:
                os.makedirs(path_check)
            torch.save(
                state,
                "./checkpoints/" + experiment_name + "/checkpoint_" + str(epoch) + ".pth",
            )

            if WANDB_AVAILABLE and args.use_wandb:
                wandb.log({"checkpoint_saved": epoch}, step=epoch)

    if WANDB_AVAILABLE and args.use_wandb:
        wandb.finish()
        print("W&B run finished")


if __name__ == "__main__":
    main()


