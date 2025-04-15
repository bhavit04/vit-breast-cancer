import glob
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class BreastCancerDataset(Dataset):
    def __init__(self, root, mode, split=0.95, test_mode=False, quick_mode=False):
        if mode not in {"train", "valid"}:
            raise ValueError
        patient_ids = sorted(os.listdir(root))
        split = int(split * len(patient_ids))
        # train validation split
        patient_ids = patient_ids[:split] if mode == "train" else patient_ids[split:]
        
        # If in quick mode, use only 10% of patients for faster training
        if quick_mode:
            num_patients = len(patient_ids) // 10  # 10% of patients
            patient_ids = patient_ids[:num_patients]
            
        self.positives = []
        self.negatives = []
        for patient_id in patient_ids:
            for image_path in glob.glob(os.path.join(root, patient_id, "*/*.png")):
                if image_path.endswith("1.png"):
                    self.positives.append(image_path)
                else:
                    self.negatives.append(image_path)
        
        # If in test mode, use only first few patients
        if test_mode:
            self.positives = self.positives[:2]  # Use only 2 patients for testing
            self.negatives = self.negatives[:2]  # Use only 2 negative images

        # transforms
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match model input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if mode == "train":
            # data augmentation during training
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize to match model input
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    @property
    def pos_weight(self):
        return len(self.negatives) / len(self.positives)

    def __len__(self):
        return len(self.positives) + len(self.negatives)

    def __getitem__(self, i):
        label = None
        image_path = None
        if i < len(self.positives):
            label = 1.0
            image_path = self.positives[i]
        else:
            label = 0.0
            image_path = self.negatives[i - len(self.positives)]
        image = Image.open(image_path)
        image = self.transforms(image)
        return image, label


def collate_fn(batch):
    """custom collate_fn to support PIL batch"""
    inputs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return inputs, labels


def make_loaders(config):
    train_dataset = BreastCancerDataset(config.data_path, "train", config.split, config.test_mode, config.quick_mode)
    valid_dataset = BreastCancerDataset(config.data_path, "valid", config.split, config.test_mode, config.quick_mode)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        shuffle=False,
    )
    return train_loader, valid_loader
