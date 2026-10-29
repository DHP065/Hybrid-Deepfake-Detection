#!/usr/bin/env python3
"""
Model Evaluation Script for Deepfake Detection

This script provides comprehensive evaluation tools:
1. Test model performance on validation/test sets
2. Generate confusion matrices and ROC curves
3. Compare different model architectures
4. Analyze model predictions
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import json
from pathlib import Path
import logging
from tqdm import tqdm
from train_model import AdvancedDeepfakeModel, DeepfakeDataset
from hybrid_deepfake_detector import HybridDeepfakeDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation suite"""
    
    def __init__(self, model_path, model_type="advanced", architecture="resnet18", device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model_type = model_type
        
        # Load model
        if model_type == "advanced":
            self.model = AdvancedDeepfakeModel(architecture=architecture).to(self.device)
            self.load_advanced_model()
        else:
            # Load original hybrid detector
            self.detector = HybridDeepfakeDetector(model_path)
            self.model = self.detector.model
        
        self.model.eval()
        
        # Transforms for evaluation
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load_advanced_model(self):
        """Load advanced model checkpoint"""
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            logger.info(f"Loaded model from {self.model_path}")
        else:
            logger.warning(f"Model file not found: {self.model_path}")
    
    def evaluate_dataset(self, data_loader, class_names=["Real", "Fake"]):
        """Evaluate model on a dataset"""
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(data_loader, desc="Evaluating"):
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities), np.array(all_targets)
    
    def calculate_metrics(self, predictions, probabilities, targets):
        """Calculate comprehensive evaluation metrics with safety checks"""
        # Manual sanity accuracy
        total = len(targets)
        num_correct = int((predictions == targets).sum())
        manual_accuracy = num_correct / total if total > 0 else 0.0

        # Core metrics
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'accuracy_manual': manual_accuracy,
            'num_correct': num_correct,
            'total': int(total),
            'precision': precision_score(targets, predictions, average='binary', zero_division=0),
            'recall': recall_score(targets, predictions, average='binary', zero_division=0),
            'f1_score': f1_score(targets, predictions, average='binary', zero_division=0),
            'auc_roc': roc_auc_score(targets, probabilities[:, 1]) if probabilities.shape[1] > 1 else 0.5
        }
        
        # Per-class metrics
        precision_per_class = precision_score(targets, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(targets, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)
        
        metrics['per_class'] = {
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1_score': f1_per_class.tolist()
        }
        
        # Class distribution
        metrics['class_counts'] = {
            'real_0': int((targets == 0).sum()),
            'fake_1': int((targets == 1).sum())
        }
        
        return metrics
    
    def plot_confusion_matrix(self, predictions, targets, class_names=["Real", "Fake"], save_path="confusion_matrix.png"):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, probabilities, targets, save_path="roc_curve.png"):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(targets, probabilities[:, 1])
        auc = roc_auc_score(targets, probabilities[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fpr, tpr, auc
    
    def plot_prediction_distribution(self, probabilities, targets, save_path="prediction_distribution.png"):
        """Plot distribution of prediction probabilities"""
        real_probs = probabilities[targets == 0, 1]  # Fake probability for real samples
        fake_probs = probabilities[targets == 1, 1]  # Fake probability for fake samples
        
        plt.figure(figsize=(10, 6))
        plt.hist(real_probs, bins=50, alpha=0.7, label='Real Videos', color='green')
        plt.hist(fake_probs, bins=50, alpha=0.7, label='Fake Videos', color='red')
        plt.xlabel('Fake Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Probabilities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_misclassifications(self, predictions, probabilities, targets, data_loader, max_examples=10):
        """Analyze misclassified examples"""
        misclassified_indices = np.where(predictions != targets)[0]
        
        logger.info(f"Total misclassifications: {len(misclassified_indices)}")
        
        # Group by type of error
        false_positives = misclassified_indices[targets[misclassified_indices] == 0]  # Real predicted as fake
        false_negatives = misclassified_indices[targets[misclassified_indices] == 1]  # Fake predicted as real
        
        logger.info(f"False Positives (Real → Fake): {len(false_positives)}")
        logger.info(f"False Negatives (Fake → Real): {len(false_negatives)}")
        
        # Analyze confidence of misclassifications
        fp_confidences = probabilities[false_positives, 1] if len(false_positives) > 0 else []
        fn_confidences = probabilities[false_negatives, 0] if len(false_negatives) > 0 else []
        
        if len(fp_confidences) > 0:
            logger.info(f"False Positive avg confidence: {np.mean(fp_confidences):.3f}")
        if len(fn_confidences) > 0:
            logger.info(f"False Negative avg confidence: {np.mean(fn_confidences):.3f}")
        
        return {
            'false_positives': false_positives[:max_examples].tolist(),
            'false_negatives': false_negatives[:max_examples].tolist(),
            'fp_confidences': fp_confidences[:max_examples].tolist() if len(fp_confidences) > 0 else [],
            'fn_confidences': fn_confidences[:max_examples].tolist() if len(fn_confidences) > 0 else []
        }
    
    def compare_thresholds(self, probabilities, targets):
        """Find optimal threshold for classification"""
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_f1 = 0
        
        results = []
        
        for threshold in thresholds:
            pred_thresh = (probabilities[:, 1] > threshold).astype(int)
            
            acc = accuracy_score(targets, pred_thresh)
            prec = precision_score(targets, pred_thresh, average='binary', zero_division=0)
            rec = recall_score(targets, pred_thresh, average='binary', zero_division=0)
            f1 = f1_score(targets, pred_thresh, average='binary', zero_division=0)
            
            results.append({
                'threshold': threshold,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Plot threshold analysis
        thresholds_list = [r['threshold'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        precisions = [r['precision'] for r in results]
        recalls = [r['recall'] for r in results]
        f1_scores = [r['f1_score'] for r in results]
        
        plt.figure(figsize=(12, 8))
        plt.plot(thresholds_list, accuracies, label='Accuracy', marker='o')
        plt.plot(thresholds_list, precisions, label='Precision', marker='s')
        plt.plot(thresholds_list, recalls, label='Recall', marker='^')
        plt.plot(thresholds_list, f1_scores, label='F1-Score', marker='d')
        plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best Threshold ({best_threshold:.2f})')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Performance vs Classification Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Optimal threshold: {best_threshold:.3f} (F1-Score: {best_f1:.3f})")
        
        return best_threshold, results
    
    def generate_report(self, test_data_dir, output_dir="evaluation_results"):
        """Generate comprehensive evaluation report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test data
        test_dataset = DeepfakeDataset(test_data_dir, self.transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        logger.info(f"Evaluating on {len(test_dataset)} test samples")
        
        # Evaluate model
        predictions, probabilities, targets = self.evaluate_dataset(test_loader)
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, probabilities, targets)
        
        # Generate visualizations
        cm = self.plot_confusion_matrix(predictions, targets, 
                                      save_path=os.path.join(output_dir, "confusion_matrix.png"))
        
        fpr, tpr, auc = self.plot_roc_curve(probabilities, targets,
                                          save_path=os.path.join(output_dir, "roc_curve.png"))
        
        self.plot_prediction_distribution(probabilities, targets,
                                        save_path=os.path.join(output_dir, "prediction_distribution.png"))
        
        # Threshold analysis
        best_threshold, threshold_results = self.compare_thresholds(probabilities, targets)
        
        # Misclassification analysis
        misclassifications = self.analyze_misclassifications(predictions, probabilities, targets, test_loader)
        
        # Create comprehensive report
        report = {
            'model_path': self.model_path,
            'test_samples': len(test_dataset),
            'metrics': metrics,
            'best_threshold': best_threshold,
            'confusion_matrix': cm.tolist(),
            'misclassifications': misclassifications,
            'threshold_analysis': threshold_results
        }
        
        # Save report
        with open(os.path.join(output_dir, "evaluation_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Test Samples: {len(test_dataset)}")
        # Display metrics as percentages for readability and show sanity counts
        print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%  (manual: {metrics['accuracy_manual'] * 100:.2f}%)")
        print(f"Correct/Total: {metrics['num_correct']}/{metrics['total']}")
        print(f"Class counts → real(0): {metrics['class_counts']['real_0']}, fake(1): {metrics['class_counts']['fake_1']}")
        print(f"Precision: {metrics['precision'] * 100:.2f}%")
        print(f"Recall: {metrics['recall'] * 100:.2f}%")
        print(f"F1-Score: {metrics['f1_score'] * 100:.2f}%")
        print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
        print(f"Best Threshold: {best_threshold:.3f}")
        print("="*50)
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data directory")
    parser.add_argument("--model_type", type=str, default="advanced", choices=["advanced", "hybrid"],
                       help="Type of model to evaluate")
    parser.add_argument("--architecture", type=str, default="resnet18",
                       choices=["resnet18", "resnet50", "efficientnet_b0", "vit_b_16"])
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path, args.model_type, args.architecture)
    
    # Generate evaluation report
    report = evaluator.generate_report(args.test_data, args.output_dir)
    
    logger.info(f"Evaluation completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
