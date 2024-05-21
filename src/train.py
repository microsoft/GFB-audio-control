import torch
from torch.utils.data import DataLoader
import time
from pathlib import Path
import os
import hydra
import utils.utils as utils
from test import do_test
import numpy as np
import random


# used to initialize the workers with random seeds
def worker_init_fn(worker_id):
    seed = (int(torch.randint(0, 2**30, (1,))) + worker_id) % (2**32 - 1)
    np.random.seed(seed)  # worker_id
    random.seed(seed)


def train(args):

    save_path = Path(args.save_path)

    # create save path if it does not exist
    save_path.mkdir(parents=True, exist_ok=True)

    gpu_id = 0

    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(gpu_id)
        print("Training on GPU IDs", gpu_id)
    except:
        device = "cpu"
        print("cuda not found, using cpu as the device")

    num_workers = int(args.exp.num_workers)

    #######################
    # Load Backbone       #
    #######################
    model = hydra.utils.instantiate(args.backbone.dnn)
    model.to(device)

    print(
        "Total number of parameters:", sum(p.numel() for p in model.parameters()) / 1e6
    )

    ####################
    # Setup optimizer  #
    ####################
    optim = hydra.utils.instantiate(args.exp.optimizer, params=model.parameters())

    ####################
    # load checkpoint  #
    ####################

    checkpoint_file = utils.get_checkpoint_file(args.checkpoint, save_path)
    # search for any existing checkpoint that ends with .pt
    checkpoint_files = list(save_path.glob("*.pt"))
    # get timestamp of the checkpoint list
    checkpoint_files = sorted(checkpoint_files, key=lambda x: x.stat().st_mtime)
    # check for the latest checkpoint
    checkpoint_file = checkpoint_files[-1] if checkpoint_files else None

    if checkpoint_file is not None and checkpoint_file.exists():
        # resume from existing checkpoint in save_path, e.g. if training was interrupted
        print("Loading checkpoint from {}".format(checkpoint_file.as_posix()))
        checkpoint = torch.load(checkpoint_file, map_location=device)

        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded model from checkpoint")
        except:
            # load the parameters of the model which have the same name, skip the rest
            print(
                "Could not load the model from the checkpoint, loading only the parameters with the same name"
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        optim.load_state_dict(checkpoint["optimizer_state_dict"])

        steps = checkpoint["steps"]

        print("Loaded checkpoint from {}".format(checkpoint_file.as_posix()))
    else:
        # train from scratch
        steps = 0

    #######################
    # Create data loaders #
    #######################
    train_set = hydra.utils.instantiate(
        args.data.dataset, segment_length=args.exp.segment_length
    )

    batch_size = args.exp.batch_size
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        # sampler=train_sampler
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )
    train_loader = iter(train_loader)

    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = args.exp.cudnn_autotuner
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = False

    running_loss = torch.zeros(1).to(device)

    model.train()
    diff = hydra.utils.instantiate(args.diff)

    def get_batch():
        if args.exp.task == "reverb":
            (
                x,
                T60,
                C50,
            ) = next(train_loader)
            # Loading:
            #   x: (B, 1, T) dry audio
            #   T60: (B,) T60 values
            #   C50: (B,) C50 values
            #   delay: (B,) delay values

            x_dry = x.to(device, non_blocking=True)
            params = {"T60": T60, "C50": C50}
            return x_dry, params

        elif args.exp.task == "declipping":
            x, SDR = next(train_loader)
            # Loading:
            #   x: (B, 1, T) clipped audio
            #   SDR: (B,) SDR values

            x = x.to(device, non_blocking=True)
            params = {"SDR": SDR}
            return x, params

    #################
    # Training loop #
    #################

    start = time.time()
    first_iter = True
    while steps < args.exp.max_steps:

        # load the batch, contains the input and the parameters
        batch = get_batch()

        if first_iter:
            first_iter = False

        # data pairing + model forward evaluation + loss
        if args.exp.task == "reverb":
            x, params = batch
            total_loss, loss_dict, t = diff.compute_loss(
                x, model, cond=params, task="reverb"
            )

        elif args.exp.task == "declipping":
            x, params = batch
            total_loss, loss_dict, t = diff.compute_loss(
                x, model, cond=params, task="declipping"
            )

        else:
            raise NotImplementedError(
                "the task {} is not implemented".format(args.exp.task)
            )

        error = loss_dict["error"]
        # log the loss
        loss = error.mean()
        running_loss += loss.item()

        model.zero_grad(set_to_none=True)
        total_loss.backward()  # backward pass

        optim.step()  # update parameters

        # learning rate ramp up
        if steps <= args.exp.lr_rampup_it:
            for g in optim.param_groups:
                g["lr"] = args.exp.optimizer.lr * min(
                    steps / max(args.exp.lr_rampup_it, 1e-8), 1
                )

        steps += 1

        ### saving and validation
        if (steps + 1) % args.log.save_interval == 0:
            # latest checkpoint
            if args.log.save_model:
                checkpoint_file = os.path.join(
                    save_path, "checkpoint_" + str(steps) + ".pt"
                )
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "steps": steps,
                    },
                    checkpoint_file,
                )

        if (steps + 1) % args.log.heavy_log_interval == 0:
            # testing script
            do_test(
                    args,
                    diff=diff,
                    model=model,
                    save_path=save_path,
                    device=device,
                    iteration=steps,
            )

        if (steps + 1) % args.log.log_interval == 0:
            # gather loss from all GPUs
            log_str = "Iters {} | s/batch {:5.2f}".format(
                steps,
                (time.time() - start) / args.log.log_interval,
            )

            running_loss = (
                running_loss / args.log.log_interval
            )  # average over GPUs and steps
                
            log_str += " | loss {:.4f}".format(running_loss.item())

            print(log_str, " | exp: {}".format(args.run_id))
            start = time.time()


@hydra.main(config_path="conf", config_name="conf_speech_reverb", version_base="1.3.2")
def main(args):
    train(args)


if __name__ == "__main__":
    main()
