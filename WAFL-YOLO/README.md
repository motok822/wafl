# WAFL-YOLO

Wireless Ad Hoc Federated Learning with YOLOv9 (WAFL-YOLO). These codes are edited versions of [the original YOLOv9 codes](https://github.com/WongKinYiu/yolov9) to make them compatible with WAFL.

## Model exchange and aggregation

In this study, we verified three learning and model exchange methods. The first is Full Parameter Exchange (FPE), in which all parameters are learned and exchanged. The second is Detection Head Exchange (DHE), in which only the paramterers in Detection Head are learned and exchanged. The third is Head Last Exchange (HLE), in which only the last convolution layers in Detection Head are learned and exchanged.

Also, in this scenario, all the devices are assumed to be fixed and exchange their model with neighboring devices based on topology. We verified three types of topology : line, tree, and ringstar.

## Data preparation

We created the target dataset by selecting 10 categories from the Open Image Dataset and used it. Please download the custom dataset from [here](https://drive.google.com/file/d/1dFmatagowqRz7zAf0sZNxREe0sp5kX4Z/view?usp=sharing) and set it in the `data` directory. We expect the directory strcture to be the following.
```
WAFL-YOLO/data/custom_yolo/
  train # train images and labels
  val # val images and labels
```

## Usage

### Module installation

This code has been tested and verified to work with Python 3.12.3 and CUDA 12.4. The specific versions of key dependencies used in our test environment are listed in the `requirements.txt` file.

However, please note that you may need to adjust the versions, especially for `torch` and `torchvision`, to match your specific environment and CUDA version.

After ensuring versions of required dependencies, install them by following commands:

```
pip install -r requirements.txt
```

If you encounter any issues, you may need to modify the versions in `requirements.txt` to suit your specific setup. In particular, ensure that the `torch` and `torchvision` versions are compatible with your CUDA installation if you're using GPU acceleration.

### Change directory
We expect you to execute codes in the `src` directory, so please move there like:

```
cd src
```

### Pretrained-model preparation
Please download pretrained YOLOv9-c model from [here](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt) and set it in `src` directory.

For more information about the model, please refer to [here](https://github.com/WongKinYiu/yolov9).

### Training

All training parameters are configured in the `config.yaml` file located in the project root directory. To train the models, edit the configuration file and then execute `train_wafl.py`:

```bash
# First, edit config.yaml to set your desired parameters
# Then run:
cd src
python train_wafl.py
```

After training, `outputs` directory is created and it includes `results.csv` and checkpoints (If the `outputs` directory already exists, a new directory named `outputs1` will be created. If `outputs1` exists, `outputs2` will be created, and so on).

#### Configuration Parameters

All parameters are defined in `config.yaml` under the `train` section. Key parameters include:

- **weights**: The path to the file of model parameters. The parameters set here will be used as the initial values when training starts. (default: `"./src/yolov9-c.pt"`)
- **num_clients**: The number of clients which participates in the collaborative learning. (default: `10`)
- **noniid_ratio**: The percentage of each client's class in the noniid scenario. (default: `90`)
- **topology**: Select from `"line"`, `"tree"` and `"ringstar"`. (default: `"line"`)
- **iid_setting**: Set to `true` for IID scenario, `false` for non-IID. (default: `false`)
- **dhe**: Set to `true` to use Detection Head Exchange method, `false` for Full Parameter Exchange. (default: `false`)
- **hle**: Set to `true` to use Head Last Exchange method, `false` for Full Parameter Exchange. (default: `false`)
- **detail_log**: Set to `true` to display training logs in more detail. (default: `false`)
- **batch_size**: Total batch size for all GPUs. (default: `16`)
- **epochs**: Total training epochs. (default: `100`)
- **preself_epochs**: Number of preself epochs. (default: `30`)

For a complete list of parameters, please refer to the `config.yaml` file.

### Weights & Biases (WandB) Integration

WAFL-YOLO supports Weights & Biases for experiment tracking and visualization. WandB allows you to monitor training metrics, losses, and validation scores for each device in real-time.

#### Setup

1. Install WandB (already included in `requirements.txt`):
```bash
pip install wandb>=0.12.0
```

2. Login to WandB:
```bash
wandb login
```

3. Configure WandB in `config.yaml`:
```yaml
wandb:
  enabled: true  # Enable/disable WandB logging
  project: "WAFL-YOLO"  # WandB project name
  entity: null  # Your WandB username or team name (null for default)
  name: null  # Run name (null for auto-generated)
  tags: ["federated-learning", "yolov9"]  # Optional tags
  notes: "Training with 10 clients"  # Optional notes
  mode: "online"  # online, offline, or disabled
  log_model: false  # Log model checkpoints to WandB
  log_interval: 1  # Log metrics every N epochs
```

#### Logged Metrics

WandB logs the following metrics for each device (node):
- **Training losses**: `node{i}/train_box_loss`, `node{i}/train_cls_loss`, `node{i}/train_dfl_loss`
- **Validation metrics**: `node{i}/metrics_precision`, `node{i}/metrics_recall`, `node{i}/metrics_mAP_0.5`, `node{i}/metrics_mAP_0.5:0.95`
- **Fitness scores**: `node{i}/fitness`, `node{i}/best_fitness`

Where `{i}` is the device/node number (0 to num_clients-1).

#### Viewing Results

After starting training, WandB will provide a URL to view your experiment dashboard in real-time. You can:
- Compare metrics across different devices
- Track training and validation losses over time
- Monitor mAP scores for each device
- Visualize model performance

### Visualization

The `outputs` directory includes the results of training. All visualization parameters are configured in the `config.yaml` file.

#### mAP Visualization

To visualize the trends in mAP, configure the parameters in `config.yaml` under `visualization.map_plot` section and run:

```bash
cd src
python ./utils/bin/mAP_plot.py
```

Configuration parameters:
- **dirname**: Directory name for reading data and saving plot results (default: `"outputs"`)
- **preself**: Set to `true` to visualize preself training results (default: `false`)

#### Bounding Box Visualization

To check the image with inferred bounding boxes, configure the parameters in `config.yaml` under `visualization.box_vis` section and run:

```bash
cd src
python ./utils/bin/box_visualization.py
```

Configuration parameters:
- **source**: Path to the source image (default: `"../data/custom_yolo/val/images/0957f84aecdf874d.jpg"`)
- **weights**: Path to the model weights (default: `"outputs/weights/node1/last.pt"`)
- **conf_thres**: Confidence score threshold (default: `0.25`)
- **project**: Directory to save results (default: `"outputs"`)

For all available visualization parameters, please refer to the `visualization` section in `config.yaml`.
