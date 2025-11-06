import os
import warnings
from pathlib import Path

import pkg_resources as pkg
import torch
import yaml

from utils.general import LOGGER, colorstr, cv2
from utils.plots import plot_images, plot_labels, plot_results
from utils.torch_utils import de_parallel

LOGGERS = ('csv', 'tb', 'wandb', 'clearml', 'comet')  # *.csv, TensorBoard, Weights & Biases, ClearML
RANK = int(os.getenv('RANK', -1))

# Try to import wandb
try:
    import wandb
    assert hasattr(wandb, '__version__')
except (ImportError, AssertionError):
    wandb = None


class Loggers():
    # YOLO Loggers class
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.plots = not opt.noplots  # plot results
        self.logger = logger  # for printing results to console
        self.include = include
        self.keys = [
            'train/box_loss',
            'train/cls_loss',
            'train/dfl_loss',  # train loss
            'metrics/precision',
            'metrics/recall',
            'metrics/mAP_0.5',
            'metrics/mAP_0.5:0.95',  # metrics
        ]  # params
        self.best_keys = ['best/epoch', 'best/precision', 'best/recall', 'best/mAP_0.5', 'best/mAP_0.5:0.95']
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv

        # Load WandB config
        self.wandb_config = self._load_wandb_config()

        # Initialize WandB
        if self.wandb_config.get('enabled', False) and wandb and 'wandb' in self.include:
            self._init_wandb()

    def _load_wandb_config(self):
        """Load WandB configuration from config.yaml"""
        try:
            config_path = Path(__file__).parents[2] / '../config.yaml'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    return config.get('wandb', {})
        except Exception as e:
            self.logger.warning(f'Could not load wandb config: {e}')
        return {}

    def _init_wandb(self):
        """Initialize WandB logging"""
        try:
            # WandB run configuration
            run_name = self.wandb_config.get('name') or f"run_{self.save_dir.name}"

            # Initialize WandB
            self.wandb = wandb.init(
                project=self.wandb_config.get('project', 'WAFL-YOLO'),
                entity=self.wandb_config.get('entity'),
                name=run_name,
                tags=self.wandb_config.get('tags', []),
                notes=self.wandb_config.get('notes', ''),
                mode=self.wandb_config.get('mode', 'online'),
                config={
                    **vars(self.opt),
                    **self.hyp,
                },
                resume='allow',
                id=self.wandb_config.get('id'),
            )

            self.logger.info(f'WandB: initialized successfully (Project: {self.wandb.project_name()}, Run: {self.wandb.name})')
        except Exception as e:
            self.logger.warning(f'WandB: initialization failed: {e}')
            self.wandb = None

    @property
    def remote_dataset(self):
        # Get data_dict if custom dataset artifact link is provided
        data_dict = None

        return data_dict

    def on_train_start(self):
        pass

    def on_pretrain_routine_start(self):
        pass

    def on_pretrain_routine_end(self, labels, names):
        # Callback runs on pre-train routine end
        if self.plots:
            plot_labels(labels, names, self.save_dir)
            paths = self.save_dir.glob('*labels*.jpg')  # training labels

    def on_train_batch_end(self, model, ni, imgs, targets, paths, vals):
        log_dict = dict(zip(self.keys[0:3], vals))
        # Callback runs on train batch end
        # ni: number integrated batches (since train start)
        if self.plots:
            if ni < 3:
                f = self.save_dir / f'train_batch{ni}.jpg'  # filename
                plot_images(imgs, targets, paths, f)
                if ni == 0 and self.tb and not self.opt.sync_bn:
                    log_tensorboard_graph(self.tb, model, imgsz=(self.opt.imgsz, self.opt.imgsz))
            if ni == 10:
                files = sorted(self.save_dir.glob('train*.jpg'))

    def on_train_epoch_end(self, epoch):
        # Callback runs on train epoch end
        pass

    def on_val_start(self):
        pass

    def on_val_image_end(self, pred, predn, path, names, im):
        # Callback runs on val image end
        pass

    def on_val_batch_end(self, batch_i, im, targets, paths, shapes, out):
        pass

    def on_val_end(self, nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix):
        # Callback runs on val end
        pass

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi, node_i):
        # Callback runs at the end of each fit (train+val) epoch
        x = dict(zip(self.keys, vals))
        dir = self.save_dir / f'node{node_i}'
        os.makedirs(dir, exist_ok=True)
        if self.csv:
            file = self.save_dir / f'node{node_i}' / 'results.csv'
            n = len(x) + 1  # number of cols
            s = '' if file.exists() else (('%20s,' * n % tuple(['epoch'] + self.keys)).rstrip(',') + '\n')  # add header
            with open(file, 'a') as f:
                f.write(s + ('%20.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

        # Log to WandB
        if self.wandb and epoch % self.wandb_config.get('log_interval', 1) == 0:
            try:
                # Create log dictionary with node-specific metrics
                log_dict = {
                    'epoch': epoch,
                    f'node{node_i}/fitness': fi,
                    f'node{node_i}/best_fitness': best_fitness,
                }

                # Add all metrics with node prefix
                for key, val in x.items():
                    # Replace forward slash with underscore for better WandB grouping
                    metric_name = key.replace('/', '_')
                    log_dict[f'node{node_i}/{metric_name}'] = val

                # Also log average metrics across all nodes (if tracking multiple nodes)
                # This is handled separately but we can log individual node metrics here

                self.wandb.log(log_dict, step=epoch)
            except Exception as e:
                self.logger.warning(f'WandB: logging failed: {e}')

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        # Callback runs on model save event
        # Log model checkpoint to WandB if enabled
        if self.wandb and self.wandb_config.get('log_model', False) and final_epoch:
            try:
                # Log the best model
                artifact = wandb.Artifact(
                    name=f'model-{self.wandb.id}',
                    type='model',
                    description=f'Best model from epoch {epoch}'
                )
                artifact.add_file(str(last))
                self.wandb.log_artifact(artifact)
                self.logger.info(f'WandB: logged model checkpoint')
            except Exception as e:
                self.logger.warning(f'WandB: model logging failed: {e}')

    def on_train_end(self, last, best, epoch, results):
        # Callback runs on training end, i.e. saving best model
        if self.plots:
            plot_results(file=self.save_dir / 'results.csv')  # save results.png
        files = ['results.png', 'confusion_matrix.png', *(f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R'))]
        files = [(self.save_dir / f) for f in files if (self.save_dir / f).exists()]  # filter
        self.logger.info(f"Results saved to {colorstr('bold', self.save_dir)}")

        # Log final results and plots to WandB
        if self.wandb:
            try:
                # Log result plots
                for f in files:
                    if f.exists():
                        self.wandb.log({f.stem: wandb.Image(str(f))})

                # Mark run as finished
                self.wandb.finish()
                self.logger.info('WandB: run finished successfully')
            except Exception as e:
                self.logger.warning(f'WandB: finalization failed: {e}')

    def on_params_update(self, params: dict):
        # Update hyperparams or configs of the experiment
        pass
