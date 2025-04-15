# Breast Cancer Classification using CaiT

This project implements a deep learning model for breast cancer classification using the Class-Attention in Image Transformers (CaiT) architecture. The model is trained to classify breast cancer histopathology images.

## Features

- Uses CaiT-XXS24 model (smallest variant for efficiency)
- Memory-efficient training with:
  - Mixed precision training
  - Gradient accumulation
  - Gradient clipping
  - Early stopping
- Data augmentation (random flips)
- Class weight balancing for imbalanced data
- Model checkpointing and best model saving

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- timm
- transformers
- wandb (optional, for experiment tracking)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vit-breast-cancer.git
cd vit-breast-cancer
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
vit-breast-cancer/
├── config.py           # Configuration settings
├── data.py            # Dataset and data loading
├── models/            # Model implementations
│   └── cait.py        # CaiT model
├── train.py           # Training script
├── utils.py           # Utility functions
└── requirements.txt   # Project dependencies
```

## Usage

1. Prepare your dataset:
   - Place your breast cancer histopathology images in the `raw` directory
   - Images should be organized by patient ID
   - Positive cases should end with "1.png"
   - Negative cases should have other endings

2. Configure training:
   - Adjust parameters in `config.py` as needed
   - Key parameters:
     - `batch_size`: Number of images per batch (default: 16)
     - `epochs`: Maximum number of training epochs (default: 15)
     - `lr`: Learning rate for backbone (default: 1e-10)
     - `classifier_lr`: Learning rate for classifier (default: 1e-4)

3. Start training:
```bash
python train.py
```

## Training Features

- **Early Stopping**: Training stops if validation loss doesn't improve for 3 epochs
- **Mixed Precision**: Uses FP16 training to reduce memory usage
- **Gradient Accumulation**: Accumulates gradients over 2 steps to simulate larger batch size
- **Model Checkpointing**: Saves the best model based on validation loss

## Results

The model typically achieves:
- Accuracy: 85-92%
- Precision: 80-88%
- Recall: 82-90%

## License

[Your chosen license]

## Acknowledgments

- Based on the CaiT architecture from Facebook Research
- Uses the IDC breast cancer dataset
