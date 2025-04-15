import argparse
from dataclasses import fields
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
from config import Config
from data import BreastCancerDataset
from models.cait import cait_xxs24_224
from utils import set_seed


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def get_args():
    parser = argparse.ArgumentParser()
    for field in fields(Config):
        name = field.name
        default = getattr(Config, name)
        parser.add_argument(f"--{name}", default=default, type=type(default))
    args = parser.parse_args()
    return args


def train(config: Config):
    # Initialize wandb
    wandb.init(project="breast-cancer", name=config.name, config=config.__dict__)
    
    # Set random seed
    set_seed(42)
    
    # Create datasets
    train_dataset = BreastCancerDataset(config.data_path, "train", config.split, config.test_mode, config.quick_mode)
    val_dataset = BreastCancerDataset(config.data_path, "valid", config.split, config.test_mode, config.quick_mode)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Get model
    model = cait_xxs24_224(pretrained=True, num_classes=1)
    if config.freeze:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the classifier
        for param in model.head.parameters():
            param.requires_grad = True
    model = model.to(config.device)
    
    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_dataset.pos_weight]).to(config.device))
    optimizer = Adam([
        {"params": [p for n, p in model.named_parameters() if "head" not in n], "lr": config.lr},
        {"params": model.head.parameters(), "lr": config.classifier_lr}
    ])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")):
            images = images.to(config.device)
            labels = labels.to(config.device).float().view(-1, 1)
            
            # Mixed precision training
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / config.gradient_accumulation_steps
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            
            # Step optimizer every gradient_accumulation_steps
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Clip gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                # Update weights
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * config.gradient_accumulation_steps
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(config.device)
                labels = labels.to(config.device).float().view(-1, 1)
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Check early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
        
        # Log metrics
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{config.save_path}/best_model.pth")
            print(f"New best model saved with validation loss: {val_loss:.4f}")
    
    wandb.finish()


def main():
    config = Config()
    train(config)


if __name__ == "__main__":
    main()
