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

ROOT_DIR = Path(__file__).resolve().parent.parent
config_loader = ConfigLoader()
config = config_loader.load_config(ROOT_DIR / "config" / "config.yaml")

# Use Pydantic config directly instead of argparse namespace
wandb_config = config.wandb
training_config = config.training
dataset_config = config.dataset
federated_config = config.federated

# Initialize WandB with proper configuration
if wandb_config.enabled:
    # Generate run name based on configuration
    run_name = f"nodes{federated_config.num_devices}_noniid{int(dataset_config.non_iid * 100)}_lr{training_config.lr}_epochs{training_config.epochs}"

    # Set WandB API key if provided
    if wandb_config.key:
        import os

        os.environ["WANDB_API_KEY"] = wandb_config.key

    runs = []
    for i in range(config.federated.num_devices):
        if config.wandb.enabled:
            run = wandb.init(
                project=config.wandb.project,
                name=run_name + f"_node{i}",
                reinit="create_new",
                group=config.wandb.group,
            )
            runs.append(run)
        else:
            runs.append(None)

else:
    print("WandB logging disabled")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.random.manual_seed(training_config.seed)
random.seed(training_config.seed)
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

# Use config values directly
node_num = federated_config.num_devices
class_num = dataset_config.class_num
batch_size = dataset_config.batch_size
noniid = dataset_config.non_iid
pre_epoch = training_config.local_train_epochs
local_training_steps = federated_config.local_training_steps
max_epoch = training_config.epochs
save_dir = training_config.save_dir
lr = training_config.lr

contact_list = []
filename = ROOT_DIR / "src" / "contact_pattern" / "rwp_n10_a0500_r100_p10_s01.json"
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
            {"params": [nets[i].cls_token], "lr": lr},
            {"params": [nets[i].pos_embed], "lr": lr},
            {"params": nets[i].patch_embed.parameters(), "lr": lr},
            {"params": nets[i].blocks.parameters(), "lr": lr},
            {"params": nets[i].norm.parameters(), "lr": lr},
            {"params": nets[i].head.parameters(), "lr": 1e-3},
        ]
    )
    for i in range(node_num)
]
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

        avg_loss = running_loss / max(1, batch_index + 1)
        accuracy = 100 * correct / total if total > 0 else 0

        print(
            f"Pre-Epoch [{epoch + 1}/{pre_epoch}] Node {n} Loss {avg_loss:.4f} Accuracy {accuracy:.2f}%"
        )
        runs[n].log(
            {
                f"train_loss_{n}": avg_loss,
                f"train_accuracy_{n}": accuracy,
            },
            step=epoch,
        )
    for n in range(node_num):
        acc = evaluate(nets[n], validate_dataloader, device)
        runs[n].log(
            {
                f"eval_accuracy_{n}": acc,
            },
            step=epoch,
        )


print("Pre-Self Training completed")

# save (after pre self training)
for n in range(node_num):
    torch.save(nets[n].state_dict(), f"{save_dir}/prenet_e{pre_epoch}_n{n}.pth")

for n in range(node_num):
    acc = evaluate(nets[n], validate_dataloader, device)
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

        avg_loss = running_loss / max(1, batch_index + 1)
        accuracy = 100 * correct / total if total > 0 else 0

        runs[n].log(
            {
                f"train_loss_{n}": avg_loss,
                f"train_accuracy_{n}": accuracy,
            },
            step=pre_epoch + epoch,
        )

    # Validation evaluation
    validation_accuracies = []
    for n in range(node_num):
        acc = evaluate(nets[n], validate_dataloader, device)
        validation_accuracies.append(acc)
        runs[n].log(
            {
                f"eval_accuracy_{n}": acc,
            },
            step=pre_epoch + epoch,
        )

    # save
    if (epoch + 1) == max_epoch:
        for n in range(node_num):
            torch.save(nets[n].state_dict(), f"{save_dir}/net_e{epoch + 1}_n{n}.pth")
