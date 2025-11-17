import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator, validator

logger = logging.getLogger(__name__)


class FedProxConfig(BaseModel):
    """FedProx algorithm configuration."""

    enabled: bool = Field(default=False, description="Enable FedProx algorithm")
    alpha: float = Field(
        default=0.01, ge=0.0, le=1.0, description="FedProx regularization parameter"
    )


class DynamicLRConfig(BaseModel):
    """Dynamic learning rate configuration."""

    enabled: bool = Field(
        default=False, description="Enable dynamic learning rate adjustment"
    )
    beta: float = Field(default=1e-3, gt=0.0, description="Dynamic LR beta parameter")
    target_ratio: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Target ratio for dynamic LR"
    )
    multiplier: float = Field(default=1.0, gt=0.0, description="LR multiplier factor")


class TrainingConfig(BaseModel):
    """Training configuration."""

    epochs: int = Field(
        default=100, ge=1, description="Total number of training epochs"
    )
    local_train_epochs: int = Field(
        default=10, ge=1, description="Number of local training epochs per round"
    )
    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")
    lr: float = Field(default=1e-5, gt=0.0, description="Learning rate")
    fedprox: FedProxConfig = Field(
        default_factory=FedProxConfig, description="FedProx configuration"
    )
    dynamic_lr: DynamicLRConfig = Field(
        default_factory=DynamicLRConfig, description="Dynamic LR configuration"
    )
    save_dir: Optional[str] = Field(
        default="output", description="Directory to save trained models"
    )

    @validator("lr")
    def validate_lr(cls, v):
        if v <= 0 or v > 1.0:
            raise ValueError("Learning rate must be between 0 and 1")
        return v


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    batch_size: int = Field(default=64, ge=1, description="Batch size for training")
    class_num: int = Field(
        default=100, ge=1, description="Number of classes in the dataset"
    )
    non_iid: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Non-IID ratio (0=IID, 1=completely non-IID)",
    )
    seed: int = Field(default=42, ge=0, description="Seed for dataset partitioning")

    @validator("batch_size")
    def validate_batch_size(cls, v):
        if v & (v - 1) != 0:  # Check if power of 2
            logger.warning(
                f"Batch size {v} is not a power of 2, which may affect performance"
            )
        return v


class FederatedConfig(BaseModel):
    """Federated learning configuration."""

    enabled: bool = Field(default=True, description="Enable federated learning")
    num_devices: int = Field(
        default=10, ge=1, description="Number of federated clients"
    )
    max_mult: float = Field(
        default=1.0, gt=0.0, description="Maximum multiplication factor"
    )
    contact_pattern: str = Field(
        default="rwp_n10_a0500_r100_p10_s01.json",
        description="Path to contact pattern JSON file",
    )
    local_training_steps: Optional[int] = Field(
        default=10, ge=1, description="Number of local training steps per epoch"
    )

    @validator("num_devices")
    def validate_num_devices(cls, v):
        if v > 1000:
            logger.warning(f"Large number of clients ({v}) may impact performance")
        return v


class WandBConfig(BaseModel):
    """Weights & Biases configuration."""

    enabled: bool = Field(default=True, description="Enable WandB logging")
    project: str = Field(default="wafl-dial", description="WandB project name")
    group: Optional[str] = Field(default=None, description="WandB group name")
    key: Optional[str] = Field(default=None, description="WandB API key")

    @validator("project")
    def validate_project_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("WandB project name cannot be empty")
        return v.strip()

    @validator("key")
    def validate_api_key(cls, v):
        if v and len(v) < 20:
            logger.warning("WandB API key seems too short, please verify")
        return v


class WAFLDIALConfig(BaseModel):
    """Complete WAFL-DIAL configuration."""

    training: TrainingConfig = Field(
        default_factory=TrainingConfig, description="Training configuration"
    )
    dataset: DatasetConfig = Field(
        default_factory=DatasetConfig, description="Dataset configuration"
    )
    federated: FederatedConfig = Field(
        default_factory=FederatedConfig, description="Federated learning configuration"
    )
    wandb: WandBConfig = Field(
        default_factory=WandBConfig, description="WandB configuration"
    )

    @model_validator(mode="after")
    def validate_config_consistency(self):
        """Validate consistency across configuration sections."""
        # Ensure local epochs don't exceed total epochs
        if self.training.local_train_epochs > self.training.epochs:
            raise ValueError("Local train epochs cannot exceed total epochs")

        # Warn if batch size is very large relative to expected data per client
        if self.dataset.batch_size > 1000 and self.federated.num_devices > 50:
            logger.warning("Large batch size with many clients may cause memory issues")

        return self

    def generate_run_name(self) -> str:
        """Generate a descriptive run name based on config parameters."""
        name_parts = [
            f"nodes{self.federated.num_devices}",
            f"nonIID{int(self.dataset.non_iid * 100)}",
            f"lr{self.training.lr}",
            f"epochs{self.training.epochs}",
        ]
        if self.training.fedprox.enabled:
            name_parts.append(f"FedProx{self.training.fedprox.alpha}")
        if self.training.dynamic_lr.enabled:
            name_parts.append(
                f"DynamicLRb{self.training.dynamic_lr.beta}t{self.training.dynamic_lr.target_ratio}"
            )
        return "_".join(name_parts)

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"  # Forbid extra fields
        use_enum_values = True


class ConfigLoader:
    """Pydantic-based configuration loader for WAFL-DIAL project."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the config loader.

        Args:
            config_path: Path to the YAML config file
        """
        self.config: Optional[WAFLDIALConfig] = None
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: Path) -> WAFLDIALConfig:
        """Load configuration from YAML file.

        Args:
            config_path: Path to the YAML config file

        Returns:
            Validated configuration object

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValidationError: If configuration validation fails
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)

            # Validate and parse with Pydantic
            self.config = WAFLDIALConfig.model_validate(raw_config)
            logger.info(f"Configuration loaded and validated from {config_path}")
            return self.config

        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")

    def get_training_config(self) -> TrainingConfig:
        """Get training configuration section."""
        if self.config is None:
            return TrainingConfig()
        return self.config.training

    def get_dataset_config(self) -> DatasetConfig:
        """Get dataset configuration section."""
        if self.config is None:
            return DatasetConfig()
        return self.config.dataset

    def get_federated_config(self) -> FederatedConfig:
        """Get federated learning configuration section."""
        if self.config is None:
            return FederatedConfig()
        return self.config.federated

    def get_wandb_config(self) -> WandBConfig:
        """Get WandB configuration section."""
        if self.config is None:
            return WandBConfig()
        return self.config.wandb

    def get_fedprox_config(self) -> FedProxConfig:
        """Get FedProx configuration section."""
        training_config = self.get_training_config()
        return training_config.fedprox

    def get_dynamic_lr_config(self) -> DynamicLRConfig:
        """Get dynamic learning rate configuration section."""
        training_config = self.get_training_config()
        return training_config.dynamic_lr
