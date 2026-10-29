#!/usr/bin/env python3
"""
Advanced Training Script for Hybrid Deepfake Detection Model

This script provides multiple training strategies to improve model accuracy:
1. Fine-tuning the existing ResNet18 model
2. Transfer learning with different architectures
3. Data augmentation for better generalization
4. Advanced training techniques (mixup, cutmix, etc.)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import numpy as np
import cv2
import json
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from pathlib import Path
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepfakeDataset(Dataset):
    """Custom dataset for deepfake detection training"""
    
    def __init__(self, data_dir, transform=None, max_samples=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # Load samples (expecting structure: data_dir/real/, data_dir/fake/)
        for label, class_name in enumerate(['real', 'fake']):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), label))
                for img_path in class_dir.glob('*.png'):
                    self.samples.append((str(img_path), label))
        
        # Shuffle and limit samples if specified
        random.shuffle(self.samples)
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Return a black image if loading fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class AdvancedDeepfakeModel(nn.Module):
    """Enhanced model with multiple architectures and techniques"""
    
    def __init__(self, architecture='resnet18', pretrained=True, num_classes=2, dropout=0.5):
        super(AdvancedDeepfakeModel, self).__init__()
        self.architecture = architecture
        
        if architecture == 'resnet18':
            if pretrained:
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet18(weights=None)
            num_features = self.backbone.fc.in_features
            
        elif architecture == 'resnet50':
            if pretrained:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            else:
                self.backbone = models.resnet50(weights=None)
            num_features = self.backbone.fc.in_features
            
        elif architecture == 'efficientnet_b0':
            if pretrained:
                self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.efficientnet_b0(weights=None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif architecture == 'vit_b_16':
            if pretrained:
                self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.vit_b_16(weights=None)
            num_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()
        
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Enhanced classifier with dropout and batch normalization
        if architecture in ['resnet18', 'resnet50']:
            self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        return self.classifier(features)

class MixUp:
    """MixUp data augmentation for better generalization"""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch):
        x, y = batch
        batch_size = x.size(0)
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam

class DeepfakeTrainer:
    """Advanced trainer with multiple optimization strategies"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Data transforms with advanced augmentation
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1)
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Initialize model
        self.model = AdvancedDeepfakeModel(
            architecture=config.architecture,
            pretrained=config.pretrained,
            dropout=config.dropout
        ).to(self.device)
        
        # Load existing weights if available
        if config.resume_from and os.path.exists(config.resume_from):
            self.load_checkpoint(config.resume_from)
        
        # Loss and optimizer
        if config.use_focal_loss:
            self.criterion = self.focal_loss
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        
        # MixUp if enabled
        self.mixup = MixUp(config.mixup_alpha) if config.use_mixup else None
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_accuracy = 0.0
    
    def focal_loss(self, outputs, targets, alpha=1, gamma=2):
        """Focal loss for handling class imbalance"""
        ce_loss = nn.CrossEntropyLoss(reduction='none')(outputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def get_optimizer(self):
        """Get optimizer based on configuration"""
        if self.config.optimizer == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def get_scheduler(self):
        """Get learning rate scheduler"""
        if self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        elif self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.step_size, gamma=0.1)
        elif self.config.scheduler == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5)
        else:
            return None
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            # Try to load as state dict directly
            self.model.load_state_dict(checkpoint, strict=False)
            logger.info(f"Loaded model weights from {checkpoint_path}")
    
    def save_checkpoint(self, epoch, accuracy, filepath):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'config': self.config
        }, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixup and random.random() < 0.5:
                # Apply MixUp
                mixed_data, target_a, target_b, lam = self.mixup((data, target))
                output = self.model(mixed_data)
                loss = lam * self.criterion(output, target_a) + (1 - lam) * self.criterion(output, target_b)
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Store for detailed metrics
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                probs = torch.softmax(output, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake class
        
        accuracy = 100. * correct / total
        
        # Calculate detailed metrics
        precision = precision_score(all_targets, all_preds, average='binary')
        recall = recall_score(all_targets, all_preds, average='binary')
        f1 = f1_score(all_targets, all_preds, average='binary')
        try:
            auc = roc_auc_score(all_targets, all_probs)
        except:
            auc = 0.5
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        return metrics
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.config.epochs):
            logger.info(f"Epoch {epoch+1}/{self.config.epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            logger.info(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
            logger.info(f"Val F1: {val_metrics['f1']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = val_metrics['accuracy']
                self.save_checkpoint(
                    epoch, 
                    val_metrics['accuracy'], 
                    f"models/best_model_{self.config.architecture}.pth"
                )
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    epoch, 
                    val_metrics['accuracy'], 
                    f"models/checkpoint_epoch_{epoch+1}.pth"
                )
        
        # Plot training curves
        self.plot_training_curves()
        
        logger.info(f"Training completed! Best validation accuracy: {self.best_accuracy:.2f}%")
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_accuracies, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Validation Accuracy')
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_config():
    """Create training configuration"""
    class Config:
        def __init__(self):
            # Model configuration
            self.architecture = 'resnet18'  # resnet18, resnet50, efficientnet_b0, vit_b_16
            self.pretrained = True
            self.dropout = 0.5
            
            # Training configuration
            self.epochs = 50
            self.batch_size = 32
            self.lr = 1e-4
            self.weight_decay = 1e-4
            self.optimizer = 'adamw'  # adam, adamw, sgd
            self.scheduler = 'cosine'  # cosine, step, reduce_on_plateau
            self.step_size = 20
            
            # Advanced techniques
            self.use_mixup = True
            self.mixup_alpha = 0.2
            self.use_focal_loss = False
            self.label_smoothing = 0.1
            
            # Data configuration
            self.train_data_dir = 'data/train'
            self.val_data_dir = 'data/val'
            self.max_samples = None  # Limit samples for quick testing
            
            # Checkpointing
            self.resume_from = 'models/resnet18_ffpp.pth'  # Path to existing model
    
    return Config()

def main():
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--architecture', type=str, default='resnet18', 
                       choices=['resnet18', 'resnet50', 'efficientnet_b0', 'vit_b_16'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_config()
    if args.data_dir:
        config.train_data_dir = os.path.join(args.data_dir, 'train')
        config.val_data_dir = os.path.join(args.data_dir, 'val')
    config.architecture = args.architecture
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    if args.resume:
        config.resume_from = args.resume
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize trainer
    trainer = DeepfakeTrainer(config)
    
    # Create datasets
    if os.path.exists(config.train_data_dir):
        train_dataset = DeepfakeDataset(config.train_data_dir, trainer.train_transform, config.max_samples)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    else:
        logger.error(f"Training data directory not found: {config.train_data_dir}")
        return
    
    if os.path.exists(config.val_data_dir):
        val_dataset = DeepfakeDataset(config.val_data_dir, trainer.val_transform)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    else:
        logger.warning(f"Validation data directory not found: {config.val_data_dir}")
        val_loader = None
    
    # Start training
    if val_loader:
        trainer.train(train_loader, val_loader)
    else:
        logger.info("Training without validation data")
        for epoch in range(config.epochs):
            trainer.train_epoch(train_loader)

if __name__ == "__main__":
    main()
