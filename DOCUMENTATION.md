# CrossFormer API Documentation

This document provides comprehensive documentation for all public APIs, functions, and components in the CrossFormer codebase.

## Table of Contents

1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Core Architecture](#core-architecture)
4. [Configuration System](#configuration-system)
5. [Model APIs](#model-apis)
6. [Data Loading APIs](#data-loading-apis)
7. [Training APIs](#training-apis)
8. [Utility Functions](#utility-functions)
9. [Detection & Segmentation](#detection--segmentation)
10. [Examples](#examples)

## Overview

CrossFormer is a versatile vision transformer that builds cross-scale attention among objects/features of different scales. The core innovations include:

- **Cross-scale Embedding Layer (CEL)**: Blends input embeddings with multiple-scale features
- **Long-Short Distance Attention (L/SDA)**: Splits embeddings into groups for efficient attention computation
- **Dynamic Position Bias (DPB)**: Enables flexible relative position bias for variable image sizes

## Installation & Setup

### Prerequisites

```bash
pip3 install numpy scipy Pillow pyyaml torch==1.7.0 torchvision==0.8.1 timm==0.3.2
```

### Dataset Setup
- ImageNet dataset with `train` and `validation` directories
- For detection/segmentation: See [detection/README.md](./detection/README.md) and [segmentation/README.md](./segmentation/README.md)

## Core Architecture

### CrossFormer Model

The main model class implementing the CrossFormer architecture.

#### Class: `CrossFormer`

```python
from models.crossformer import CrossFormer

model = CrossFormer(
    img_size=224,                    # Input image size
    patch_size=[4, 8, 16, 32],      # Multi-scale patch sizes
    in_chans=3,                     # Input channels
    num_classes=1000,               # Number of output classes
    embed_dim=96,                   # Embedding dimension
    depths=[2, 2, 6, 2],           # Number of blocks per stage
    num_heads=[3, 6, 12, 24],      # Attention heads per stage
    group_size=[7, 7, 7, 7],       # Group size for attention
    mlp_ratio=4.0,                 # MLP expansion ratio
    qkv_bias=True,                 # Enable query/key/value bias
    qk_scale=None,                 # Override default QK scale
    drop_rate=0.0,                 # Dropout rate
    drop_path_rate=0.1,            # Stochastic depth rate
    ape=False,                     # Absolute position embedding
    patch_norm=True,               # Patch embedding normalization
    use_checkpoint=False,          # Gradient checkpointing
    merge_size=[[2], [2], [2]]     # Patch merging sizes
)
```

#### Key Methods

**`forward(x)`**
- **Input**: `x` - Tensor of shape `(B, C, H, W)`
- **Output**: Tensor of shape `(B, num_classes)`
- **Description**: Forward pass through the model

**`flops()`**
- **Output**: Float - Number of FLOPs for the model
- **Description**: Calculate floating point operations

**Example Usage:**

```python
import torch
from models.crossformer import CrossFormer

# Create model
model = CrossFormer(num_classes=1000)

# Forward pass
x = torch.randn(2, 3, 224, 224)
output = model(x)
print(f"Output shape: {output.shape}")  # [2, 1000]

# Calculate FLOPs
flops = model.flops()
print(f"Model FLOPs: {flops/1e9:.2f}G")
```

### Core Components

#### Class: `CrossFormerBlock`

Individual transformer block with cross-scale attention.

```python
block = CrossFormerBlock(
    dim=96,                        # Feature dimension
    input_resolution=(56, 56),     # Input spatial resolution
    num_heads=3,                   # Number of attention heads
    group_size=7,                  # Attention group size
    lsda_flag=0,                   # 0 for SDA, 1 for LDA
    mlp_ratio=4.0,                 # MLP expansion ratio
    qkv_bias=True,                 # Enable QKV bias
    drop=0.0,                      # Dropout rate
    attn_drop=0.0,                 # Attention dropout
    drop_path=0.0                  # Stochastic depth rate
)
```

#### Class: `Attention`

Multi-head self-attention with dynamic position bias.

```python
attention = Attention(
    dim=96,                        # Input dimension
    group_size=(7, 7),            # Attention group size
    num_heads=3,                   # Number of heads
    qkv_bias=True,                # Enable QKV bias
    attn_drop=0.0,                # Attention dropout
    proj_drop=0.0,                # Projection dropout
    position_bias=True            # Enable position bias
)
```

#### Class: `DynamicPosBias`

Dynamic position bias module for variable image sizes.

```python
pos_bias = DynamicPosBias(
    dim=24,                       # Position dimension (dim // 4)
    num_heads=3,                  # Number of attention heads
    residual=False                # Use residual connections
)
```

## Configuration System

### Function: `get_config(args)`

Creates and returns a configuration object with all model and training parameters.

```python
from config import get_config
import argparse

# Parse arguments
args = argparse.Namespace(
    cfg='configs/tiny_patch4_group7_224.yaml',
    batch_size=128,
    data_path='/path/to/imagenet',
    # ... other arguments
)

config = get_config(args)
```

### Configuration Structure

```python
# Data configuration
config.DATA.BATCH_SIZE = 128
config.DATA.IMG_SIZE = 224
config.DATA.DATASET = 'imagenet'
config.DATA.DATA_PATH = '/path/to/data'

# Model configuration
config.MODEL.TYPE = 'cross-scale'
config.MODEL.NUM_CLASSES = 1000
config.MODEL.DROP_RATE = 0.0
config.MODEL.DROP_PATH_RATE = 0.1

# CrossFormer specific
config.MODEL.CROS.PATCH_SIZE = [4, 8, 16, 32]
config.MODEL.CROS.EMBED_DIM = 48
config.MODEL.CROS.DEPTHS = [2, 2, 6, 2]
config.MODEL.CROS.NUM_HEADS = [3, 6, 12, 24]
config.MODEL.CROS.GROUP_SIZE = [7, 7, 7, 7]

# Training configuration
config.TRAIN.EPOCHS = 300
config.TRAIN.BASE_LR = 5e-4
config.TRAIN.WEIGHT_DECAY = 0.05
config.TRAIN.OPTIMIZER.NAME = 'adamw'
config.TRAIN.LR_SCHEDULER.NAME = 'cosine'
```

## Model APIs

### Function: `build_model(config, args)`

Factory function to create CrossFormer models based on configuration.

**Parameters:**
- `config`: Configuration object
- `args`: Command line arguments

**Returns:** CrossFormer model instance

**Example:**

```python
from models.build import build_model
from config import get_config

args = parse_args()
config = get_config(args)
model = build_model(config, args)
```

### Pre-trained Models

Available model variants with different sizes:

```python
# CrossFormer-T (Tiny)
model_tiny = CrossFormer(
    embed_dim=48,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24]
)

# CrossFormer-S (Small) 
model_small = CrossFormer(
    embed_dim=64,
    depths=[2, 2, 18, 2],
    num_heads=[3, 6, 12, 24]
)

# CrossFormer-B (Base)
model_base = CrossFormer(
    embed_dim=96,
    depths=[2, 2, 18, 2], 
    num_heads=[3, 6, 12, 24]
)

# CrossFormer-L (Large)
model_large = CrossFormer(
    embed_dim=128,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32]
)
```

## Data Loading APIs

### Function: `build_loader(config)`

Creates data loaders for training and validation.

**Parameters:**
- `config`: Configuration object

**Returns:**
- `dataset_train`: Training dataset
- `dataset_val`: Validation dataset 
- `data_loader_train`: Training data loader
- `data_loader_val`: Validation data loader
- `mixup_fn`: Mixup function (optional)

**Example:**

```python
from data.build import build_loader

dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

# Use in training loop
for batch_idx, (images, targets) in enumerate(data_loader_train):
    if mixup_fn is not None:
        images, targets = mixup_fn(images, targets)
    # ... training code
```

### Function: `build_dataset(is_train, config)`

Creates datasets for training or validation.

**Parameters:**
- `is_train`: Boolean indicating training or validation
- `config`: Configuration object

**Returns:**
- `dataset`: Dataset object
- `nb_classes`: Number of classes

### Function: `build_transform(is_train, config)`

Creates data augmentation transforms.

**Example:**

```python
from data.build import build_transform

# Training transforms (with augmentation)
train_transform = build_transform(is_train=True, config=config)

# Validation transforms (without augmentation)
val_transform = build_transform(is_train=False, config=config)
```

## Training APIs

### Function: `main(args, config)`

Main training function that orchestrates the entire training process.

**Parameters:**
- `args`: Parsed command line arguments
- `config`: Configuration object

### Function: `train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler)`

Trains the model for one epoch.

**Parameters:**
- `config`: Configuration object
- `model`: Model to train
- `criterion`: Loss function
- `data_loader`: Training data loader
- `optimizer`: Optimizer
- `epoch`: Current epoch number
- `mixup_fn`: Mixup function (optional)
- `lr_scheduler`: Learning rate scheduler
- `loss_scaler`: Loss scaler for mixed precision

### Function: `validate(config, data_loader, model)`

Validates the model on validation set.

**Parameters:**
- `config`: Configuration object
- `data_loader`: Validation data loader
- `model`: Model to validate

**Returns:**
- `acc1`: Top-1 accuracy
- `acc5`: Top-5 accuracy
- `loss`: Average validation loss

**Example Training Script:**

```python
import torch
from main import main, parse_option

if __name__ == '__main__':
    args, config = parse_option()
    
    # Setup distributed training
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl')
    
    # Start training
    main(args, config)
```

### Function: `throughput(data_loader, model, logger)`

Measures model throughput (images per second).

**Example:**

```python
from main import throughput

# Test model throughput
throughput(data_loader_val, model, logger)
```

## Utility Functions

### Checkpoint Management

#### Function: `load_checkpoint(config, model, optimizer, lr_scheduler, logger)`

Loads model checkpoint for resuming training.

**Parameters:**
- `config`: Configuration object
- `model`: Model to load weights into
- `optimizer`: Optimizer to load state
- `lr_scheduler`: LR scheduler to load state
- `logger`: Logger for output

**Returns:** Maximum accuracy achieved

**Example:**

```python
from utils import load_checkpoint

max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
```

#### Function: `save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, best=False)`

Saves model checkpoint.

**Example:**

```python
from utils import save_checkpoint

save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, best=True)
```

#### Function: `auto_resume_helper(output_dir)`

Finds the latest checkpoint in output directory for auto-resuming.

**Returns:** Path to latest checkpoint or None

### Optimizer APIs

#### Function: `build_optimizer(config, model)`

Creates optimizer with proper weight decay settings.

**Example:**

```python
from optimizer import build_optimizer

optimizer = build_optimizer(config, model)
```

### Learning Rate Scheduler APIs

#### Function: `build_scheduler(config, optimizer, n_iter_per_epoch)`

Creates learning rate scheduler.

**Example:**

```python
from lr_scheduler import build_scheduler

lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
```

#### Class: `LinearLRScheduler`

Custom linear learning rate scheduler with warmup.

```python
from lr_scheduler import LinearLRScheduler

scheduler = LinearLRScheduler(
    optimizer,
    t_initial=num_steps,
    lr_min_rate=0.01,
    warmup_lr_init=1e-7,
    warmup_t=warmup_steps
)
```

### Logging APIs

#### Function: `create_logger(output_dir, dist_rank=0, name='')`

Creates logger for training output.

**Example:**

```python
from logger import create_logger

logger = create_logger(output_dir='./output', dist_rank=0, name='crossformer')
logger.info("Training started")
```

### Gradient and Tensor Utilities

#### Function: `get_grad_norm(parameters, norm_type=2)`

Calculates gradient norm for monitoring training.

#### Function: `reduce_tensor(tensor)`

Reduces tensor across distributed processes.

## Detection & Segmentation

The codebase includes specialized components for dense prediction tasks:

### Detection

Located in `detection/` directory:

- **Training:** `detection/train.py`
- **Testing:** `detection/test.py`
- **CrossFormer Factory:** `detection/crossformer_factory.py`

**Example Usage:**

```bash
# Training detection model
cd detection
python train.py --cfg configs/crossformer_tiny_patch4_group7_224_1x.py

# Testing detection model  
python test.py --cfg configs/crossformer_tiny_patch4_group7_224_1x.py --checkpoint /path/to/checkpoint.pth
```

### Segmentation

Located in `segmentation/` directory:

- **Training:** `segmentation/train.py`
- **Testing:** `segmentation/test.py`
- **Align Resize:** `segmentation/align_resize.py`

**Example Usage:**

```bash
# Training segmentation model
cd segmentation  
python train.py --cfg configs/upernet_crossformer_tiny_512x512_80k_ade20k.py

# Testing segmentation model
python test.py --cfg configs/upernet_crossformer_tiny_512x512_80k_ade20k.py --checkpoint /path/to/checkpoint.pth
```

## Examples

### Complete Training Example

```python
#!/usr/bin/env python3

import os
import torch
import torch.distributed as dist
from main import main, parse_option

def train_crossformer():
    # Parse arguments
    args, config = parse_option()
    
    # Setup distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = -1
        world_size = -1
        
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(
        backend='nccl', 
        init_method='env://', 
        world_size=world_size, 
        rank=rank
    )
    
    # Start training
    main(args, config)

if __name__ == '__main__':
    train_crossformer()
```

**Command Line Usage:**

```bash
# Single GPU training
python train_example.py --cfg configs/tiny_patch4_group7_224.yaml --data-path /path/to/imagenet

# Multi-GPU training
python -u -m torch.distributed.launch --nproc_per_node 8 train_example.py \
    --cfg configs/tiny_patch4_group7_224.yaml \
    --batch-size 128 \
    --data-path /path/to/imagenet \
    --output ./output
```

### Inference Example

```python
import torch
from models.crossformer import CrossFormer
from torchvision import transforms
from PIL import Image

def inference_example():
    # Load pre-trained model
    model = CrossFormer(num_classes=1000)
    checkpoint = torch.load('crossformer_tiny.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Prepare input
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open('example.jpg').convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    # Get top-5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    
    for i in range(5):
        print(f"Class {top5_indices[i]}: {top5_prob[i]:.4f}")

if __name__ == '__main__':
    inference_example()
```

### Custom Dataset Example

```python
import torch
from torch.utils.data import Dataset, DataLoader
from data.build import build_transform

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, config, is_train=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = build_transform(is_train, config)
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]

# Usage
dataset = CustomDataset(image_paths, labels, config, is_train=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### FLOPs Calculation Example

```python
from models.crossformer import CrossFormer

def calculate_model_complexity():
    models = {
        'CrossFormer-T': CrossFormer(embed_dim=48, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]),
        'CrossFormer-S': CrossFormer(embed_dim=64, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24]),
        'CrossFormer-B': CrossFormer(embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24]),
        'CrossFormer-L': CrossFormer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
    }
    
    for name, model in models.items():
        flops = model.flops()
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}:")
        print(f"  Parameters: {params/1e6:.1f}M")
        print(f"  FLOPs: {flops/1e9:.1f}G")
        print()

if __name__ == '__main__':
    calculate_model_complexity()
```

---

## Performance Benchmarks

### ImageNet Classification Results

| Model | Params | FLOPs | Top-1 Acc | Pretrained Weights |
|-------|--------|-------|-----------|-------------------|
| CrossFormer-T | 27.8M | 2.9G | 81.5% | [Google Drive](https://drive.google.com/file/d/1YSkU9enn-ITyrbxLH13zNcBYvWSEidfq/view?usp=sharing) |
| CrossFormer-S | 30.7M | 4.9G | 82.5% | [Google Drive](https://drive.google.com/file/d/1RAkigsgr33va0RZ85S2Shs2BhXYcS6U8/view?usp=sharing) |
| CrossFormer-B | 52.0M | 9.2G | 83.4% | [Google Drive](https://drive.google.com/file/d/1bK8biVCi17nz_nkt7rBfio_kywUpllSU/view?usp=sharing) |
| CrossFormer-L | 92.0M | 16.1G | 84.0% | [Google Drive](https://drive.google.com/file/d/1zRWByVW_KIZ87NgaBkDIm60DAsGJErdG/view?usp=sharing) |

### Detection Results (COCO)

| Backbone | Head | Schedule | box AP | mask AP |
|----------|------|----------|--------|---------|
| CrossFormer-S | Mask-RCNN | 3x | 48.7 | 43.9 |
| CrossFormer-B | Mask-RCNN | 3x | 49.8 | 44.5 |
| CrossFormer-S | Cascade-Mask-RCNN | 3x | 52.2 | 45.2 |

### Segmentation Results (ADE20K)

| Backbone | Head | Iterations | mIoU | MS mIoU |
|----------|------|------------|------|---------|
| CrossFormer-S | UPerNet | 160K | 47.6 | 48.4 |
| CrossFormer-B | UPerNet | 160K | 49.7 | 50.6 |
| CrossFormer-L | UPerNet | 160K | 50.4 | 51.4 |

## Citation

```bibtex
@inproceedings{wang2021crossformer,
  title = {CrossFormer: A Versatile Vision Transformer Hinging on Cross-scale Attention},
  author = {Wang, Wenxiao and Yao, Lu and Chen, Long and Lin, Binbin and Cai, Deng and He, Xiaofei and Liu, Wei},
  booktitle = {International Conference on Learning Representations, {ICLR}},
  url = {https://openreview.net/forum?id=_PHymLIxuI},
  year = {2022}
}
```

## License

This project is licensed under the terms specified in the [LICENSE](./LICENSE) file.