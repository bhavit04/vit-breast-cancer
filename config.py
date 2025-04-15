from dataclasses import dataclass


@dataclass
class Config:
    name: str = "gpu_training"
    device: str = "cuda"  # Changed from cuda to cpu
    log_path: str = "logs"
    data_path: str = "raw"
    save_path: str = "checkpoints"
    model: str = "cait_xxs24_224"  # Smallest CaiT model for efficiency
    # google/vit-base-patch16-224-in21k
    # microsoft/beit-base-patch16-224-pt22k-ft22k
    # microsoft/swin-base-patch4-window7-224-in22k
    # "cait"
    freeze: bool = True  # Keep backbone frozen initially
    epochs: int = 15     # Reduced epochs to prevent overheating
    lr: float = 1e-10
    classifier_lr: float = 1e-4
    split: float = 0.95
    threshold: int = 5
    batch_size: int = 16  # Reduced batch size to prevent memory overload
    num_workers: int = 2  # Reduced workers to prevent CPU overload
    test_mode: bool = False
    quick_mode: bool = False
    # Memory efficient settings
    mixed_precision: bool = True  # Enable mixed precision training
    gradient_accumulation_steps: int = 2  # Accumulate gradients to simulate larger batch size
    max_grad_norm: float = 1.0  # Gradient clipping to prevent exploding gradients
