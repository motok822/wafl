import json
import random
from pathlib import Path

import torch
import torch.optim as optim
import torchvision
import wandb
from config_loader import ConfigLoader
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Subset
from transformers import DetrForObjectDetection, DetrImageProcessor
from utils import create_subsets, update_nets

ROOT_DIR = Path(__file__).resolve().parent.parent
config_loader = ConfigLoader()
config = config_loader.load_config(ROOT_DIR / "config" / "config_detr.yaml")


# Custom dataset class for COCO format object detection
class CustomDetection(torchvision.datasets.CocoDetection):
    """Custom Dataset in COCO format for DETR."""

    def __init__(self, img_folder, ann_file, image_processor):
        super().__init__(img_folder, ann_file)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]

        # Filter out annotations without bounding boxes
        target = [obj for obj in target if "bbox" in obj]

        # Convert to DETR format
        boxes = []
        labels = []
        for obj in target:
            # COCO format: [x, y, width, height]
            x, y, w, h = obj["bbox"]
            # Convert to [x_min, y_min, x_max, y_max]
            boxes.append([x, y, x + w, y + h])
            labels.append(obj["category_id"])

        # Prepare annotations in DETR format
        annotations = {"image_id": image_id, "annotations": target}

        # Process image and annotations
        encoding = self.image_processor(
            images=img, annotations=annotations, return_tensors="pt"
        )

        # Remove batch dimension
        pixel_values = encoding["pixel_values"].squeeze(0)
        target = encoding["labels"][0] if "labels" in encoding else {}

        return pixel_values, target


def collate_fn(batch):
    """Custom collate function for DETR."""
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return encoding["pixel_values"], labels


def evaluate_detr(model, dataloader, device):
    """Evaluate DETR model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # For mAP calculation
    results = []
    image_ids = []

    with torch.no_grad():
        for pixel_values, targets in dataloader:
            pixel_values = pixel_values.to(device)

            # Move targets to device
            targets_device = []
            for target in targets:
                target_device = {}
                for k, v in target.items():
                    if isinstance(v, torch.Tensor):
                        target_device[k] = v.to(device)
                    else:
                        target_device[k] = v
                targets_device.append(target_device)
                image_ids.append(target["image_id"].item())

            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=targets_device)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

            # Post-process predictions
            orig_target_sizes = torch.stack(
                [t["orig_size"] for t in targets], dim=0
            ).to(device)
            results_batch = image_processor.post_process_object_detection(
                outputs, target_sizes=orig_target_sizes, threshold=0.0
            )

            for i, result in enumerate(results_batch):
                image_id = targets[i]["image_id"].item()
                for score, label, box in zip(
                    result["scores"], result["labels"], result["boxes"]
                ):
                    box = box.tolist()
                    # Convert to [x, y, w, h]
                    box[2] -= box[0]
                    box[3] -= box[1]

                    results.append(
                        {
                            "image_id": image_id,
                            "category_id": label.item(),
                            "bbox": box,
                            "score": score.item(),
                        }
                    )

    avg_loss = total_loss / max(1, num_batches)

    # Calculate mAP
    map50 = 0.0
    if len(results) > 0:
        coco_gt = dataloader.dataset.coco
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # mAP @ IoU=0.50 is at index 1
        map50 = coco_eval.stats[1]

    return avg_loss, map50


# Initialize WandB with proper configuration
if config.wandb.enabled:
    # Set WandB API key if provided
    if config.wandb.key:
        import os

        os.environ["WANDB_API_KEY"] = config.wandb.key

    wandb.login(key=config.wandb.key)
    run_name = f"{config.wandb.project}_nodes{config.federated.num_devices}_iid{config.dataset.non_iid}_lr{config.training.lr}_bs{config.dataset.batch_size}"

    runs = []
    for i in range(config.federated.num_devices):
        wandb_run = wandb.init(
            project=config.wandb.project,
            name=run_name + f"_run{i}",
            group=config.wandb.group,
            config={
                "num_devices": config.federated.num_devices,
                "non_iid": config.dataset.non_iid,
                "lr": config.training.lr,
                "epochs": config.training.epochs,
                "batch_size": config.dataset.batch_size,
            },
            reinit="create_new",
        )
        # Define metrics
        wandb_run.define_metric("train_step")
        wandb_run.define_metric("wafl_epoch")

        # Associate metrics with axes
        wandb_run.define_metric("train_loss", step_metric="train_step")
        wandb_run.define_metric("fedprox_reg", step_metric="train_step")
        wandb_run.define_metric("train_avg_loss", step_metric="wafl_epoch")
        wandb_run.define_metric("val_loss", step_metric="wafl_epoch")
        wandb_run.define_metric("val_map50", step_metric="wafl_epoch")

        runs.append(wandb_run)
else:
    print("WandB logging disabled")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.random.manual_seed(config.training.seed)
random.seed(config.training.seed)

# Initialize DETR image processor
image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Load Custom Dataset (COCO format)
data_root = ROOT_DIR / config.dataset.data_root
train_img_folder = data_root / config.dataset.train_images
train_ann_file = data_root / config.dataset.train_annotations
val_img_folder = data_root / config.dataset.val_images
val_ann_file = data_root / config.dataset.val_annotations

train_dataset = CustomDetection(
    img_folder=train_img_folder,
    ann_file=train_ann_file,
    image_processor=image_processor,
)
train_dataloaders = []
print("Data loaded")

# Use config values from the namespace
node_num = config.federated.num_devices
class_num = config.dataset.class_num
batch_size = config.dataset.batch_size
noniid = config.dataset.non_iid
pre_epoch = config.training.local_train_epochs
local_training_steps = config.federated.local_training_steps
max_epoch = config.training.epochs
save_dir = getattr(config.training, "save_dir", "output_detr")
lr_list = [config.training.lr for _ in range(node_num)]

contact_list = []
filename = ROOT_DIR / "src" / "contact_pattern" / config.federated.contact_pattern
print(f"Loading ... {filename}")
with open(filename, "r") as f:
    contact_list = json.load(f)

# Create data subsets for each device
if noniid == 0:  # iid
    trainset_size = int(len(train_dataset) / node_num)
    for n in range(node_num):
        indices = list(range(n * trainset_size, (n + 1) * trainset_size))
        traindataset = Subset(train_dataset, indices)
        train_dataloaders.append(
            torch.utils.data.DataLoader(
                traindataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
            )
        )
else:
    # For object detection, we need to handle non-iid differently
    # Simple approach: distribute images based on dominant class
    class_to_node = {cls: random.randint(0, node_num - 1) for cls in range(class_num)}
    subsets = create_subsets(
        train_dataset,
        num_nodes=node_num,
        class_to_node=class_to_node,
        bias_ratio=noniid,
        label_extractor=lambda item: item[1]["class_labels"][0].item()
        if "class_labels" in item[1] and len(item[1]["class_labels"]) > 0
        else 0,
    )
    for n in range(node_num):
        train_dataloaders.append(
            torch.utils.data.DataLoader(
                subsets[n], batch_size=batch_size, shuffle=True, collate_fn=collate_fn
            )
        )

validate_dataset = CustomDetection(
    img_folder=val_img_folder, ann_file=val_ann_file, image_processor=image_processor
)
validate_dataloader = torch.utils.data.DataLoader(
    validate_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)
train_steps = [0 for _ in range(node_num)]

del train_dataset
del validate_dataset

# Initialize DETR models for each device
nets = [
    DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", num_labels=class_num, ignore_mismatched_sizes=True
    )
    for _ in range(node_num)
]

for net in nets:
    net.to(device)

print("Models loaded")

# Initialize optimizers
optimizers = [optim.AdamW(nets[i].parameters(), lr=lr_list[i]) for i in range(node_num)]

# Pre-self training
print("Starting Pre-Self Training...")
for epoch in range(pre_epoch):
    epoch_metrics = {}

    for n in range(node_num):
        net = nets[n]
        optimizer = optimizers[n]
        trainloader = train_dataloaders[n]

        net.train()
        running_loss = 0.0
        batch_count = 0

        for batch_index, (pixel_values, labels) in enumerate(trainloader):
            if batch_index == local_training_steps:
                break

            pixel_values = pixel_values.to(device)

            # Move labels to device
            labels_device = []
            for label in labels:
                label_device = {}
                for k, v in label.items():
                    if isinstance(v, torch.Tensor):
                        label_device[k] = v.to(device)
                    else:
                        label_device[k] = v
                labels_device.append(label_device)

            optimizer.zero_grad()
            outputs = net(pixel_values=pixel_values, labels=labels_device)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

            runs[n].log(
                {
                    "train_loss": loss.item(),
                    "train_step": train_steps[n],
                },
            )
            train_steps[n] += 1

        avg_loss = running_loss / max(1, batch_count)

        print(f"Pre-Epoch [{epoch + 1}/{pre_epoch}] Node {n} Loss {avg_loss:.4f}")
        runs[n].log(
            {
                "train_avg_loss": avg_loss,
                "pre_self_epoch": epoch + 1,
                "phase": "pre_self_training",
                "learning_rate": lr_list[n],
            }
        )

print("Pre-Self Training completed")

# save (after pre self training)
Path(save_dir).mkdir(parents=True, exist_ok=True)
for n in range(node_num):
    torch.save(nets[n].state_dict(), f"{save_dir}/prenet_e{pre_epoch}_n{n}.pth")

last_eval_loss = [float("inf") for _ in range(node_num)]
for n in range(node_num):
    loss, map50 = evaluate_detr(nets[n], validate_dataloader, device)
    last_eval_loss[n] = loss
    print(f"Node {n} Validation Loss {loss:.4f} mAP50 {map50:.4f}")

print("Start WAFL")

for epoch in range(max_epoch):
    epoch_metrics = {}
    contact = contact_list[epoch]

    # Model aggregation (WAFL)
    update_nets(
        nets,
        contact,
        fl_coefficient=1,
        device=device,
    )

    training_losses = []

    for n in range(node_num):
        nbr = contact[str(n)]
        if len(nbr) == 0:
            continue
        print(f"Epoch [{epoch + 1}/{max_epoch}] Node {n} training with neighbors {nbr}")

        net = nets[n]
        optimizer = optimizers[n]
        trainloader = train_dataloaders[n]

        net.train()
        running_loss = 0.0
        batch_count = 0

        for batch_index, (pixel_values, labels) in enumerate(trainloader):
            if batch_index == local_training_steps:
                break

            pixel_values = pixel_values.to(device)

            # Move labels to device
            labels_device = []
            for label in labels:
                label_device = {}
                for k, v in label.items():
                    if isinstance(v, torch.Tensor):
                        label_device[k] = v.to(device)
                    else:
                        label_device[k] = v
                labels_device.append(label_device)

            optimizer.zero_grad()
            outputs = net(pixel_values=pixel_values, labels=labels_device)
            loss = outputs.loss

            # FedProx regularization if enabled
            if config.training.fedprox.enabled:
                fedprox_reg = 0.0
                for param_name, param in net.named_parameters():
                    prox_param = torch.zeros_like(param)
                    for neighbor in nbr:
                        neighbor_param = dict(nets[neighbor].named_parameters())[
                            param_name
                        ]
                        prox_param += neighbor_param
                    prox_param /= len(nbr)
                    fedprox_reg += torch.norm(param - prox_param, p=2) ** 2
                loss += (config.training.fedprox.alpha / 2) * fedprox_reg
            else:
                fedprox_reg = 0.0

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

            runs[n].log(
                {
                    "train_loss": loss.item(),
                    "fedprox_reg": fedprox_reg
                    if config.training.fedprox.enabled
                    else 0,
                    "train_step": train_steps[n],
                },
            )
            train_steps[n] += 1

        avg_loss = running_loss / max(1, batch_count)

        runs[n].log(
            {
                "train_avg_loss": avg_loss,
                "wafl_epoch": epoch + 1,
                "phase": "wafl_training",
                "learning_rate": lr_list[n],
            },
        )

    # Validation evaluation
    validation_losses = []
    for n in range(node_num):
        # skip evaluation if no contacts
        if len(contact[str(n)]) == 0:
            runs[n].log(
                {
                    "val_loss": last_eval_loss[n],
                    "wafl_epoch": epoch + 1,
                    "phase": "wafl_training",
                },
            )
            continue

        loss, map50 = evaluate_detr(nets[n], validate_dataloader, device)
        validation_losses.append(loss)
        print(f"Node {n} Validation Loss {loss:.4f} mAP50 {map50:.4f}")
        last_eval_loss[n] = loss
        runs[n].log(
            {
                "val_loss": loss,
                "val_map50": map50,
                "wafl_epoch": epoch + 1,
                "phase": "wafl_training",
            },
        )

    # save
    if (epoch + 1) == max_epoch:
        for n in range(node_num):
            torch.save(nets[n].state_dict(), f"{save_dir}/net_e{epoch + 1}_n{n}.pth")

print("Training completed!")
