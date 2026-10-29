#!/usr/bin/env python3
"""
Quick Training Script for Improving Your Deepfake Detection Model

This is a simplified version that helps you get started with training immediately.
It will fine-tune your existing ResNet18 model for better accuracy.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import numpy as np
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDeepfakeDataset(Dataset):
    """Simple dataset for quick training"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # Load samples from real and fake directories
        for label, class_name in enumerate(['real', 'fake']):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.rglob('*.jpg'):
                    self.samples.append((str(img_path), label))
                for img_path in class_dir.rglob('*.png'):
                    self.samples.append((str(img_path), label))
        
        logger.info(f"Found {len(self.samples)} samples in {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_model(pretrained_path="models/resnet18_ffpp.pth"):
    """Create and load the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create ResNet18 model
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Binary classification
    
    # Load existing weights if available
    if os.path.exists(pretrained_path):
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device, weights_only=False)
        
        # Filter out classifier weights if needed
        if any(k.startswith("fc.") for k in state_dict.keys()):
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
            model.load_state_dict(filtered_state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
    else:
        logger.warning(f"Pretrained model not found at {pretrained_path}")
    
    return model.to(device)

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4):
    """Simple training loop"""
    device = next(model.parameters()).device
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_losses.append(avg_train_loss)
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            val_acc = 100. * val_correct / val_total
            val_accuracies.append(val_acc)
            
            logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "models/improved_model.pth")
                logger.info(f"New best model saved! Validation accuracy: {val_acc:.2f}%")
        else:
            logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        scheduler.step()
    
    return train_losses, val_accuracies, best_val_acc

def main():
    parser = argparse.ArgumentParser(description="Quick training for deepfake detection model")
    parser.add_argument("--train_data", type=str, default="data/train", help="Training data directory")
    parser.add_argument("--val_data", type=str, default="data/val", help="Validation data directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--pretrained", type=str, default="models/resnet18_ffpp.pth", help="Pretrained model path")
    
    args = parser.parse_args()
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    if not os.path.exists(args.train_data):
        logger.error(f"Training data directory not found: {args.train_data}")
        logger.info("Please prepare your data first using prepare_data.py")
        logger.info("Or create the following structure:")
        logger.info("  data/")
        logger.info("    train/")
        logger.info("      real/")
        logger.info("        image1.jpg")
        logger.info("      fake/")
        logger.info("        image2.jpg")
        return
    
    train_dataset = SimpleDeepfakeDataset(args.train_data, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    val_loader = None
    if os.path.exists(args.val_data):
        val_dataset = SimpleDeepfakeDataset(args.val_data, val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Create model
    model = create_model(args.pretrained)
    
    # Train model
    logger.info("Starting training...")
    train_losses, val_accuracies, best_val_acc = train_model(
        model, train_loader, val_loader, args.epochs, args.lr
    )
    
    # Save final model
    torch.save(model.state_dict(), "models/final_trained_model.pth")
    
    # Save training history
    history = {
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "best_val_accuracy": best_val_acc,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size
    }
    
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    logger.info("Models saved:")
    logger.info("  - models/improved_model.pth (best model)")
    logger.info("  - models/final_trained_model.pth (final model)")
    logger.info("  - training_history.json (training metrics)")

if __name__ == "__main__":
    main()
