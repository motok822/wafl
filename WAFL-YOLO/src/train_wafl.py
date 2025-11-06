import math
import os
import random
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val_dual as validate  # for end-of-epoch mAP
from functions.aggregation import model_aggregate
from models.yolo import Model
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_img_size,
    check_yaml,
    colorstr,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    one_cycle,
    one_flat_cycle,
    print_args,
    yaml_save,
)
from utils.loss_tal_dual import ComputeLoss
from utils.metrics import fitness
from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_optimizer,
)

# Single GPU training only (RANK=-1)
RANK = -1
LOCAL_RANK = -1
WORLD_SIZE = 1
GIT_INFO = None


def train(
    hyp, opt, device, callbacks, preself=False
):  # hyp is path/to/hyp.yaml or hyp dictionary
    (
        save_dir,
        epochs,
        preself_epochs,
        batch_size,
        weights,
        single_cls,
        data,
        cfg,
        noval,
        nosave,
        workers,
        num_clients,
        iid_setting,
        noniid_ratio,
        topology,
        dhe,
        hle,
        detail_log,
    ) = (
        Path(opt.save_dir),
        opt.epochs,
        opt.preself_epochs,
        opt.batch_size,
        opt.weights,
        opt.single_cls,
        opt.data,
        opt.cfg,
        opt.noval,
        opt.nosave,
        opt.workers,
        opt.num_clients,
        opt.iid_setting,
        opt.noniid_ratio,
        opt.topology,
        opt.dhe,
        opt.hle,
        opt.detail_log,
    )
    callbacks.run("on_pretrain_routine_start")

    if preself:
        save_dir = save_dir / "preself"
        epochs = preself_epochs

    # Directories
    w = save_dir / "weights"  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir

    lasts = []
    bests = []
    for i in range(num_clients):
        os.makedirs(w / f"node{i}", exist_ok=True)
        lasts.append(w / f"node{i}" / "last.pt")
        bests.append(w / f"node{i}" / "best.pt")

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    hyp["anchor_t"] = 5.0
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings
    yaml_save(save_dir / "hyp.yaml", hyp)
    yaml_save(save_dir / "opt.yaml", vars(opt))

    # Config
    cuda = device.type != "cpu"
    init_seeds(opt.seed + 1, deterministic=True)
    data_dict = None  # Initialize data_dict
    data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict["train"], data_dict["val"]
    print(f"train path : {train_path}, val path : {val_path}")
    nc = 1 if single_cls else int(data_dict["nc"])  # number of classes
    names = (
        {0: "item"}
        if single_cls and len(data_dict["names"]) != 1
        else data_dict["names"]
    )  # class names

    # Model
    models = []
    ckpts = []
    amps = []
    for i in range(num_clients):
        if weights:
            if preself:
                weights_i = weights
            else:
                weights_i = f"{save_dir}/preself/weights/node{i}/last.pt"
            ckpt = torch.load(
                weights_i, weights_only=False, map_location="cpu"
            )  # load checkpoint to CPU to avoid CUDA memory leak
            model = Model(
                cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")
            ).to(device)  # create
            exclude = ["anchor"] if (cfg or hyp.get("anchors")) else []  # exclude keys
            csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
            csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
            model.load_state_dict(csd, strict=False)  # load
        else:
            model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(
                device
            )  # create
        amp = check_amp(model)  # check AMP
        models.append(model)
        ckpts.append(ckpt)
        amps.append(amp)

    private_params = []

    if dhe:
        for i in range(num_clients):
            for k, v in models[i].named_parameters():
                if ("dfl" in k) or ("38" in k):
                    v.requires_grad = True
                else:
                    private_params.append(k)
                    v.requires_grad = False

    if hle:
        for i in range(num_clients):
            for k, v in models[i].named_parameters():
                if (
                    ("dfl" in k)
                    or ("38.cv2.2.2." in k)
                    or ("38.cv3.2.2." in k)
                    or ("38.cv4.2.2." in k)
                    or ("38.cv5.2.2." in k)
                ):
                    v.requires_grad = True
                else:
                    private_params.append(k)
                    v.requires_grad = False

    # Image size
    gs = max(int(models[0].stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(models[0], imgsz, amps[0])

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizers = [
        smart_optimizer(
            models[i], opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"]
        )
        for i in range(num_clients)
    ]

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
    elif opt.flat_cos_lr:
        lf = one_flat_cycle(1, hyp["lrf"], epochs)  # flat cosine 1->hyp['lrf']
    elif opt.fixed_lr:
        lf = lambda x: 1.0
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear

    schedulers = [
        lr_scheduler.LambdaLR(optimizers[i], lr_lambda=lf) for i in range(num_clients)
    ]

    # EMA
    emas = [ModelEMA(models[i]) for i in range(num_clients)]

    # Initialize training state
    best_fitness, start_epoch = 0.0, 0
    if weights:
        del ckpt, csd

    # Trainloader
    train_loaders, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,
        cache=None if opt.cache == "val" else opt.cache,
        rect=opt.rect,
        rank=-1,
        workers=workers,
        image_weights=opt.image_weights,
        close_mosaic=opt.close_mosaic != 0,
        quad=opt.quad,
        prefix=colorstr("train: "),
        shuffle=True,
        min_items=opt.min_items,
        num_clients=num_clients,
        num_classes=nc,
        noniid_ratio=noniid_ratio,
        iid_setting=iid_setting,
        mode="train",
    )
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, (
        f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"
    )

    val_loader = create_dataloader(
        val_path,
        imgsz,
        batch_size * 2,
        gs,
        single_cls,
        hyp=hyp,
        cache=None if noval else opt.cache,
        rect=True,
        rank=-1,
        workers=workers * 2,
        pad=0.5,
        prefix=colorstr("val: "),
        num_clients=num_clients,
        iid_setting=iid_setting,
        mode="val",
    )[0]

    for i in range(num_clients):
        models[i].half().float()  # pre-reduce anchor precision

    callbacks.run("on_pretrain_routine_end", labels, names)

    # Model attributes
    for i in range(num_clients):
        hyp["label_smoothing"] = opt.label_smoothing
        models[i].nc = nc  # attach number of classes to model
        models[i].hyp = hyp  # attach hyperparameters to model
        models[i].class_weights = (
            labels_to_class_weights(dataset.labels, nc).to(device) * nc
        )  # attach class weights
        models[i].names = names

    # Start training
    nb = [len(train_loaders[i]) for i in range(num_clients)]  # number of batches
    nw = [
        max(round(hyp["warmup_epochs"] * nb[i]), 100) for i in range(num_clients)
    ]  # number of warmup iterations, max(3 epochs, 100 iterations)
    maps = [np.zeros(nc) for _ in range(num_clients)]  # mAP per class
    results = [
        (0, 0, 0, 0, 0, 0, 0) for _ in range(num_clients)
    ]  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    for i in range(num_clients):
        schedulers[i].last_epoch = start_epoch - 1  # do not move
    scalers = [torch.cuda.amp.GradScaler(enabled=amps[i]) for i in range(num_clients)]
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_losses = [
        ComputeLoss(models[i]) for i in range(num_clients)
    ]  # init loss class
    callbacks.run("on_train_start")

    for epoch in range(
        start_epoch, epochs
    ):  # epoch ------------------------------------------------------------------
        callbacks.run("on_train_epoch_start")
        if not preself:
            models = model_aggregate(
                models, topology=topology, private_param=private_params
            )
        for node_i in range(num_clients):
            models[node_i].train()
            print(f"epoch: [{epoch}/{epochs}] node: {node_i} epoch start")

            # Update image weights (optional, single-GPU only)
            if opt.image_weights:
                cw = (
                    models[node_i].class_weights.cpu().numpy()
                    * (1 - maps[node_i]) ** 2
                    / nc
                )  # class weights
                iw = labels_to_image_weights(
                    dataset.labels, nc=nc, class_weights=cw
                )  # image weights
                dataset.indices = random.choices(
                    range(dataset.n), weights=iw, k=dataset.n
                )  # rand weighted idx
            if epoch == (epochs - opt.close_mosaic):
                dataset.mosaic = False

            mlosses = [
                torch.zeros(3, device=device) for _ in range(num_clients)
            ]  # mean losses
            loader = enumerate(train_loaders[node_i])
            if detail_log:
                loader = tqdm(
                    loader, total=nb[node_i], bar_format=TQDM_BAR_FORMAT
                )  # progress bar

            optimizers[node_i].zero_grad()
            for (
                i,
                (imgs, targets, paths, _),
            ) in (
                loader
            ):  # batch -------------------------------------------------------------
                callbacks.run("on_train_batch_start")
                ni = (
                    i + nb[node_i] * epoch
                )  # number integrated batches (since train start)
                imgs = (
                    imgs.to(device, non_blocking=True).float() / 255
                )  # uint8 to float32, 0-255 to 0.0-1.0

                # Warmup
                if ni <= nw[node_i]:
                    xi = [0, nw[node_i]]  # x interp
                    accumulate = max(
                        1, np.interp(ni, xi, [1, nbs / batch_size]).round()
                    )
                    for j, x in enumerate(optimizers[node_i].param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                hyp["warmup_bias_lr"] if j == 0 else 0.0,
                                x["initial_lr"] * lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(
                                ni, xi, [hyp["warmup_momentum"], hyp["momentum"]]
                            )

                # Multi-scale
                if opt.multi_scale:
                    sz = (
                        random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
                    )  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [
                            math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]
                        ]  # new shape (stretched to gs-multiple)
                        imgs = nn.functional.interpolate(
                            imgs, size=ns, mode="bilinear", align_corners=False
                        )

                # Forward
                with torch.cuda.amp.autocast(amps[node_i]):
                    pred = models[node_i](imgs)  # forward
                    loss, loss_items = compute_losses[node_i](
                        pred, targets.to(device)
                    )  # loss scaled by batch_size
                    if opt.quad:
                        loss *= 4.0

                # Backward
                scalers[node_i].scale(loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                scalers[node_i].unscale_(optimizers[node_i])  # unscale gradients
                torch.nn.utils.clip_grad_norm_(
                    models[node_i].parameters(), max_norm=10.0
                )  # clip gradients
                scalers[node_i].step(optimizers[node_i])  # optimizer.step
                scalers[node_i].update()
                optimizers[node_i].zero_grad()
                if emas[node_i]:
                    emas[node_i].update(models[node_i])

                # Log
                mlosses[node_i] = (mlosses[node_i] * i + loss_items) / (
                    i + 1
                )  # update mean losses
                mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                if detail_log:
                    loader.set_description(
                        ("%11s" * 2 + "%11.4g" * 5)
                        % (
                            f"{epoch}/{epochs - 1}",
                            mem,
                            *mlosses[node_i],
                            targets.shape[0],
                            imgs.shape[-1],
                        )
                    )
                callbacks.run(
                    "on_train_batch_end",
                    models[node_i],
                    ni,
                    imgs,
                    targets,
                    paths,
                    list(mlosses[node_i]),
                )
                if callbacks.stop_training:
                    return
                # end batch ------------------------------------------------------------------------------------------------

            # Scheduler
            schedulers[node_i].step()

            # mAP
            callbacks.run("on_train_epoch_end", epoch=epoch)
            emas[node_i].update_attr(
                models[node_i],
                include=["yaml", "nc", "hyp", "names", "stride", "class_weights"],
            )
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results[node_i], maps[node_i], _ = validate.run(
                    data_dict,
                    batch_size=batch_size * 2,
                    imgsz=imgsz,
                    half=amps[node_i],
                    model=emas[node_i].ema,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_losses[node_i],
                    node_i=node_i,
                    detail_log=detail_log,
                )

            # Update best mAP
            fi = fitness(
                np.array(results[node_i]).reshape(1, -1)
            )  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mlosses[node_i]) + list(results[node_i])
            callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi, node_i)

            # Save model
            if (not nosave) or final_epoch:  # if save
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(de_parallel(models[node_i])).half(),
                    "ema": deepcopy(emas[node_i].ema).half(),
                    "updates": emas[node_i].updates,
                    "optimizer": optimizers[node_i].state_dict(),
                    "opt": vars(opt),
                    "git": GIT_INFO,  # {remote, branch, commit} if a git repo
                    "date": datetime.now().isoformat(),
                }

                # Save last, best and delete
                torch.save(ckpt, lasts[node_i])
                if best_fitness == fi:
                    torch.save(ckpt, bests[node_i])
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"node{node_i}" / f"epoch{epoch}.pt")
                del ckpt
                callbacks.run(
                    "on_model_save",
                    lasts[node_i],
                    epoch,
                    final_epoch,
                    best_fitness,
                    fi,
                )

            if stop:
                break  # early stopping

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    del models
    del ckpts
    del emas
    del scalers
    del optimizers
    torch.cuda.empty_cache()
    return results


def load_config(config_path="../config.yaml"):
    """Load configuration from YAML file."""
    # Convert relative path to absolute if needed
    if not Path(config_path).is_absolute():
        config_path = Path(__file__).parent / config_path

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract train configuration
    train_config = config["train"]

    # Create namespace object similar to argparse
    class ConfigNamespace:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    opt = ConfigNamespace(train_config)

    # Handle special key transformations (hyphen to underscore)
    opt.preself_epochs = train_config["preself_epochs"]
    opt.batch_size = train_config["batch_size"]
    opt.image_weights = train_config["image_weights"]
    opt.multi_scale = train_config["multi_scale"]
    opt.single_cls = train_config["single_cls"]
    opt.sync_bn = train_config["sync_bn"]
    opt.exist_ok = train_config["exist_ok"]
    opt.cos_lr = train_config["cos_lr"]
    opt.flat_cos_lr = train_config["flat_cos_lr"]
    opt.fixed_lr = train_config["fixed_lr"]
    opt.label_smoothing = train_config["label_smoothing"]
    opt.save_period = train_config["save_period"]
    opt.local_rank = train_config["local_rank"]
    opt.min_items = train_config["min_items"]
    opt.close_mosaic = train_config["close_mosaic"]
    opt.num_clients = train_config["num_clients"]
    opt.iid_setting = train_config["iid_setting"]
    opt.noniid_ratio = train_config["noniid_ratio"]

    return opt


def main(opt=None, callbacks=Callbacks()):
    # Load config if not provided
    if opt is None:
        opt = load_config()

    # Checks
    print_args(vars(opt))

    # Validate paths
    opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
        check_file(opt.data),
        check_yaml(opt.cfg),
        check_yaml(opt.hyp),
        str(opt.weights),
        str(opt.project),
    )  # checks
    assert len(opt.cfg) or len(opt.weights), (
        "either --cfg or --weights must be specified"
    )
    opt.save_dir = str(increment_path(Path(opt.project), exist_ok=opt.exist_ok))

    # Single GPU mode
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Train
    train(opt.hyp, opt, device, callbacks, preself=True)  # preself train
    train(opt.hyp, opt, device, callbacks)  # WAFL


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolo.pt')
    opt = load_config()
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = load_config()
    main(opt)
