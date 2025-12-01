import json
import random
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb
from config_loader import ConfigLoader
from torch.utils.data import Subset
from torchvision import transforms
from utils import create_subsets, evaluate, update_nets
from utils.lr_scheduler import calculate_next_lr, set_optimizer_lr

ROOT_DIR = Path(__file__).resolve().parent.parent
config_loader = ConfigLoader()
config = config_loader.load_config(ROOT_DIR / "config" / "config.yaml")


# Initialize WandB with proper configuration
if config.wandb.enabled:
    # Generate run name based on configuration

    # Set WandB API key if provided
    if config.wandb.key:
        import os

        os.environ["WANDB_API_KEY"] = config.wandb.key

    wandb.login(key=config.wandb.key)
    run_name = f"{config.wandb.project}_nodes{config.federated.num_devices}_iid{config.dataset.non_iid}_lr{config.training.lr}_bs{config.dataset.batch_size}"
    if config.training.dynamic_lr.enabled:
        run_name += f"_DynamicLRb{config.training.dynamic_lr.beta}t{config.training.dynamic_lr.target_ratio}"
    if config.training.fedprox.enabled:
        run_name += f"_FedProx{config.training.fedprox.alpha}"

    runs = []
    for i, _ in enumerate(range(config.federated.num_devices)):
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
        runs.append(wandb_run)

else:
    print("WandB logging disabled")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.random.manual_seed(config.training.seed)
random.seed(config.training.seed)
train_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
    ]
)
val_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
    ]
)

train_dataset = torchvision.datasets.CIFAR100(
    root=ROOT_DIR / "data", train=True, download=True, transform=train_transform
)
train_dataloaders = []
print("Data loaded")

# Use config values from the namespace
node_num = config.federated.num_devices  # Fixed: should be num_devices, not num_clients
class_num = config.dataset.class_num  # Fixed: should be class_num, not dataset_seed
batch_size = config.dataset.batch_size
noniid = config.dataset.non_iid
pre_epoch = config.training.local_train_epochs
local_training_steps = config.federated.local_training_steps
max_epoch = config.training.epochs
save_dir = getattr(config.training, "save_dir", "output")  # Add fallback
lr_list = [config.training.lr for _ in range(node_num)]

contact_list = []
filename = ROOT_DIR / "src" / "contact_pattern" / config.federated.contact_pattern
print(f"Loading ... {filename}")
with open(filename, "r") as f:
    contact_list = json.load(f)


if noniid == 0:  # iid
    trainset_size = int(len(train_dataset) / node_num)
    for n in range(node_num):
        indices = list(range(n * trainset_size, (n + 1) * trainset_size))
        traindataset = Subset(train_dataset, indices)
        train_dataloaders.append(
            torch.utils.data.DataLoader(
                traindataset, batch_size=batch_size, shuffle=True
            )
        )
else:
    class_to_node = {cls: random.randint(0, node_num - 1) for cls in range(class_num)}
    subsets = create_subsets(
        train_dataset,
        num_nodes=node_num,
        class_to_node=class_to_node,
        bias_ratio=noniid,
    )
    for n in range(node_num):
        train_dataloaders.append(
            torch.utils.data.DataLoader(subsets[n], batch_size=batch_size, shuffle=True)
        )

validate_dataset = torchvision.datasets.CIFAR100(
    root=ROOT_DIR / "data", train=False, download=True, transform=val_transform
)
validate_dataloader = torch.utils.data.DataLoader(
    validate_dataset, batch_size=128, shuffle=False
)
train_steps = [0 for _ in range(node_num)]

del train_dataset
del validate_dataset

nets = [
    timm.create_model("vit_base_patch16_224", pretrained=True) for _ in range(node_num)
]
for n in range(node_num):
    nets[n].head = nn.Linear(nets[n].head.in_features, 100)

prev_net = [[None] * node_num for _ in range(node_num)]
init_net = None
for net in nets:
    net.to(device)

print("Models loaded")

optimizers = [
    optim.AdamW(
        [
            {"params": [nets[i].cls_token], "lr": lr_list[i]},
            {"params": [nets[i].pos_embed], "lr": lr_list[i]},
            {"params": nets[i].patch_embed.parameters(), "lr": lr_list[i]},
            {"params": nets[i].blocks.parameters(), "lr": lr_list[i]},
            {"params": nets[i].norm.parameters(), "lr": lr_list[i]},
            {"params": nets[i].head.parameters(), "lr": lr_list[i]},
        ]
    )
    for i in range(node_num)
]
# optimizers = [optim.AdamW(nets[i].parameters(), lr=lr_list[i]) for i in range(node_num)]
criterion = nn.CrossEntropyLoss()


# pre_self training
print("Starting Pre-Self Training...")
for epoch in range(pre_epoch):
    epoch_metrics = {}

    for n in range(node_num):
        net = nets[n]
        optimizer = optimizers[n]
        trainloader = train_dataloaders[n]

        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_index, (inputs, labels) in enumerate(trainloader):
            if batch_index == local_training_steps:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            runs[n].log(
                {
                    "train_loss": loss.item(),
                },
                step=train_steps[n],
            )
            train_steps[n] += 1

        avg_loss = running_loss / max(1, batch_index + 1)
        accuracy = 100 * correct / total if total > 0 else 0

        print(
            f"Pre-Epoch [{epoch + 1}/{pre_epoch}] Node {n} Loss {avg_loss:.4f} Accuracy {accuracy:.2f}%"
        )
        runs[n].log(
            {
                "train_accuracy": accuracy,
                "pre_self_epoch": epoch + 1,
                "phase": "pre_self_training",
                "learning_rate": lr_list[n],
            }
        )


print("Pre-Self Training completed")

# save (after pre self training)
for n in range(node_num):
    torch.save(nets[n].state_dict(), f"{save_dir}/prenet_e{pre_epoch}_n{n}.pth")

last_eval_acc = [0.0 for _ in range(node_num)]
for n in range(node_num):
    acc = evaluate(nets[n], validate_dataloader, device)
    last_eval_acc[n] = acc
    print(f"Node {n} Validation Accuracy {acc:.2f}%")

print("Start WAFL")

for epoch in range(max_epoch):
    epoch_metrics = {}
    contact = contact_list[epoch]
    update_nets(
        nets,
        contact,
        fl_coefficient=1,
        device=device,
    )

    training_losses = []
    training_accuracies = []

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
        correct = 0
        total = 0

        for batch_index, (inputs, labels) in enumerate(trainloader):
            if batch_index == local_training_steps:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            if config.training.fedprox.enabled:
                fedprox_reg = 0.0
                for param_name, param in net.named_parameters():
                    prox_param = torch.zeros_like(param)
                    for neighbor in nbr:
                        neighbor_param = nets[neighbor].state_dict()[param_name]
                        prox_param += neighbor_param
                    prox_param /= len(nbr)
                    fedprox_reg += torch.norm(param - prox_param, p=2) ** 2
                loss += (config.training.fedprox.alpha / 2) * fedprox_reg
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            runs[n].log(
                {
                    "train_loss": loss.item(),
                    "fedprox_reg": fedprox_reg
                    if config.training.fedprox.enabled
                    else 0,
                },
                step=train_steps[n],
            )
            train_steps[n] += 1

        avg_loss = running_loss / max(1, batch_index + 1)
        accuracy = 100 * correct / total if total > 0 else 0

        runs[n].log(
            {
                "train_accuracy": accuracy,
                "wafl_epoch": epoch + 1,
                "phase": "wafl_training",
                "learning_rate": lr_list[n],
            },
        )

        if config.training.dynamic_lr.enabled:
            avg_model_state_dict = {}
            for key in net.state_dict().keys():
                avg_param = torch.zeros_like(net.state_dict()[key])
                for neighbor in nbr:
                    neighbor_param = nets[neighbor].state_dict()[key]
                    avg_param += neighbor_param
                avg_param /= len(nbr)
                avg_model_state_dict[key] = avg_param
            next_lr, ratio = calculate_next_lr(
                device_id=n,
                last_loss=avg_loss,
                model_state_dict=net.state_dict(),
                avg_model_state_dict=avg_model_state_dict,
                optimizer=optimizer,
                current_lr=optimizer.param_groups[0]["lr"],
                beta=config.training.dynamic_lr.beta,
                device=device,
                target_ratio=config.training.dynamic_lr.target_ratio,
                wandb_run=runs[n] if config.wandb.enabled else None,
            )
            lr_list[n] = next_lr
            set_optimizer_lr(optimizer, next_lr)

            runs[n].log({"dynamic_lr_ratio": ratio})

    # Validation evaluation
    validation_accuracies = []
    for n in range(node_num):
        # skip evaluation if no contacts
        if len(contact[str(n)]) == 0:
            runs[n].log(
                {
                    "val_accuracy": last_eval_acc[n],
                    "wafl_epoch": epoch + 1,
                    "phase": "wafl_training",
                },
            )
            continue

        acc = evaluate(nets[n], validate_dataloader, device)
        validation_accuracies.append(acc)
        print(f"Node {n} Validation Accuracy {acc:.2f}%")
        last_eval_acc[n] = acc
        runs[n].log(
            {
                "val_accuracy": acc,
                "wafl_epoch": epoch + 1,
                "phase": "wafl_training",
            },
        )

    # save
    if (epoch + 1) == max_epoch:
        for n in range(node_num):
            torch.save(nets[n].state_dict(), f"{save_dir}/net_e{epoch + 1}_n{n}.pth")
