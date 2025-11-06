import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import yaml
from pathlib import Path

def load_config(config_path='../../../config.yaml'):
    """Load configuration from YAML file."""
    if not Path(config_path).is_absolute():
        config_path = Path(__file__).parent / config_path

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract visualization configuration
    vis_config = config['visualization']['map_plot']

    class ConfigNamespace:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    return ConfigNamespace(vis_config)


def read_mAP_data_from_csv(filename):
    mAPs = []

    with open(filename, 'r') as f:
        reader = csv.reader(f)

        for i, row in enumerate(reader):
            if i == 0:
                continue
            mAPs.append(float(row[6]))

    return mAPs


if __name__ == '__main__':
    args = load_config()
    plt.figure(figsize=(10, 6))
    all_node_mAPs = []
    for i in range(6):
        filepath = f'{args.dirname}/node{i}/results.csv'
        if args.preself:
            filepath = f'{args.dirname}/preself/node{i}/results.csv'
        mAPs = read_mAP_data_from_csv(filepath)
        epochs = range(1, len(mAPs) + 1)
        plt.plot(epochs, mAPs, label=f'node{i}')
        all_node_mAPs.append(mAPs)

    plt.title('mAP@IoU=0.5')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./{args.dirname}/mAP.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    avg_mAPs = np.average(np.array(all_node_mAPs), axis=0).tolist()
    epochs = range(len(avg_mAPs))
    plt.plot(epochs, avg_mAPs)
    plt.title('mAP@IoU=0.5')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.grid(True)
    plt.savefig(f"./{args.dirname}/mAP_avg.png")
    plt.close()