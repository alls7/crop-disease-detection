"""
Training Pipeline for Crop Disease Detection
CNN vs ViT comparison on agricultural data
Author: Ali SU - Hacettepe University
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import copy
from typing import Dict, Tuple, List, Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class CropDiseaseTrainer:
    """
    Trainer class for crop disease detection models
    Handles training CNN and ViT models on PlantVillage agricultural data
    """
    
    def __init__(self, 
                 model: nn.Module,
                 model_name: str,
                 device: torch.device,
                 config: Dict):
        """
        Initialize crop disease trainer
        
        Args:
            model: CNN or ViT model for crop disease detection
            model_name: Name of the model (ResNet-50, ViT-Base/16, etc.)
            device: Device to train on (cuda/cpu)
            config: Training configuration for crop disease detection
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.config = config
        
        # Initialize optimizer for crop disease training
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler for crop disease fine-tuning
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Monitor validation accuracy
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Loss function for crop disease classification
        self.criterion = nn.CrossEntropyLoss()
        
        # Training tracking
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_weights = None
        self.patience_counter = 0
        
        logger.info(f"Initialized crop disease trainer for {model_name}")
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train the crop disease model for one epoch
        
        Args:
            train_loader: DataLoader with crop disease training data
            
        Returns:
            Tuple of (average_loss, accuracy) for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (crop_images, disease_labels) in enumerate(train_loader):
            # Move crop disease data to device
            crop_images = crop_images.to(self.device)
            disease_labels = disease_labels.to(self.device)
            
            # Forward pass through crop disease model
            self.optimizer.zero_grad()
            disease_predictions = self.model(crop_images)
            loss = self.criterion(disease_predictions, disease_labels)
            
            # Backward pass for crop disease training
            loss.backward()
            self.optimizer.step()
            
            # Track crop disease training metrics
            total_loss += loss.item()
            _, predicted_diseases = torch.max(disease_predictions.data, 1)
            total_samples += disease_labels.size(0)
            correct_predictions += (predicted_diseases == disease_labels).sum().item()
            
            # Print progress for crop disease training
            if (batch_idx + 1) % 50 == 0:
                current_acc = 100. * correct_predictions / total_samples
                logger.info(f"  Batch {batch_idx+1}/{len(train_loader)}: "
                           f"Loss={loss.item():.4f}, Acc={current_acc:.2f}%")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the crop disease model
        
        Args:
            val_loader: DataLoader with crop disease validation data
            
        Returns:
            Tuple of (average_loss, accuracy) for validation
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for crop_images, disease_labels in val_loader:
                # Move crop disease data to device
                crop_images = crop_images.to(self.device)
                disease_labels = disease_labels.to(self.device)
                
                # Forward pass through crop disease model
                disease_predictions = self.model(crop_images)
                loss = self.criterion(disease_predictions, disease_labels)
                
                # Track crop disease validation metrics
                total_loss += loss.item()
                _, predicted_diseases = torch.max(disease_predictions.data, 1)
                total_samples += disease_labels.size(0)
                correct_predictions += (predicted_diseases == disease_labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader) -> Dict[str, List]:
        """
        Complete training loop for crop disease detection
        
        Args:
            train_loader: Training data with crop disease images
            val_loader: Validation data with crop disease images
            
        Returns:
            Training history dictionary
        """
        
        epochs = self.config.get('epochs', 20)
        early_stopping_patience = self.config.get('early_stopping_patience', 5)
        
        logger.info(f"Starting crop disease training for {self.model_name}")
        logger.info(f"Training for {epochs} epochs with early stopping patience {early_stopping_patience}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            logger.info(f"\nEpoch {epoch+1}/{epochs} - {self.model_name}")
            logger.info("-" * 50)
            
            # Training phase on crop disease data
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation phase on crop disease data
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update learning rate based on crop disease validation performance
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record crop disease training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['learning_rates'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # Log crop disease training results
            logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logger.info(f"  Learning Rate: {current_lr:.6f}")
            logger.info(f"  Epoch Time: {epoch_time:.2f}s")
            
            # Save best crop disease model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
                logger.info(f"  New best crop disease validation accuracy: {val_acc:.2f}%")
                
                # Save best model for crop disease detection
                self._save_best_model()
            else:
                self.patience_counter += 1
                
            # Early stopping for crop disease training
            if self.patience_counter >= early_stopping_patience:
                logger.info(f"  Early stopping triggered for {self.model_name} after {epoch+1} epochs")
                break
        
        # Load best weights for crop disease model
        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)
        
        total_time = time.time() - start_time
        logger.info(f"\nCrop disease training completed for {self.model_name}")
        logger.info(f"Total training time: {total_time:.2f}s")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        return self.training_history
    
    def _save_best_model(self):
        """Save the best crop disease model"""
        save_dir = Path("results/models")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / f"best_{self.model_name.lower().replace('-', '_')}_crop_disease.pth"
        
        torch.save({
            'model_name': self.model_name,
            'model_state_dict': self.best_model_weights,
            'best_val_acc': self.best_val_acc,
            'training_config': self.config,
            'training_history': self.training_history
        }, model_path)
        
        logger.info(f"Saved best crop disease model to {model_path}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the trained crop disease model
        
        Args:
            test_loader: Test data with crop disease images
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {self.model_name} on crop disease test data")
        
        self.model.eval()
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for crop_images, disease_labels in test_loader:
                crop_images = crop_images.to(self.device)
                disease_labels = disease_labels.to(self.device)
                
                disease_predictions = self.model(crop_images)
                _, predicted_diseases = torch.max(disease_predictions, 1)
                
                total_samples += disease_labels.size(0)
                correct_predictions += (predicted_diseases == disease_labels).sum().item()
                
                all_predictions.extend(predicted_diseases.cpu().numpy())
                all_labels.extend(disease_labels.cpu().numpy())
        
        test_accuracy = 100. * correct_predictions / total_samples
        
        # Calculate additional metrics for crop disease detection
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        metrics = {
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_samples': total_samples
        }
        
        logger.info(f"Crop Disease Detection Results for {self.model_name}:")
        logger.info(f"  Test Accuracy: {test_accuracy:.2f}%")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall: {recall:.3f}")
        logger.info(f"  F1-Score: {f1:.3f}")
        
        return metrics, all_predictions, all_labels

def train_crop_disease_model(model: nn.Module,
                           model_name: str,
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           config: Dict,
                           device: torch.device) -> Tuple[nn.Module, Dict]:
    """
    Train a single crop disease detection model
    
    Args:
        model: Model to train for crop disease detection
        model_name: Name of the model
        train_loader: Training data with crop disease images
        val_loader: Validation data with crop disease images
        config: Training configuration
        device: Device to train on
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    
    # Create trainer for crop disease detection
    trainer = CropDiseaseTrainer(model, model_name, device, config)
    
    # Train the crop disease model
    history = trainer.train(train_loader, val_loader)
    
    return trainer.model, history, trainer.best_val_acc

def compare_crop_disease_models(models_dict: Dict[str, nn.Module],
                              train_loader: DataLoader,
                              val_loader: DataLoader,
                              test_loader: DataLoader,
                              config: Dict,
                              device: torch.device) -> Dict:
    """
    Compare multiple models on crop disease detection task
    
    Args:
        models_dict: Dictionary of model_name -> model
        train_loader: Training data with crop disease images
        val_loader: Validation data with crop disease images  
        test_loader: Test data with crop disease images
        config: Training configuration
        device: Device to train on
        
    Returns:
        Dictionary with comparison results
    """
    
    logger.info("Starting crop disease detection model comparison")
    logger.info(f"Models to compare: {list(models_dict.keys())}")
    
    results = {}
    
    for model_name, model in models_dict.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING {model_name.upper()} FOR CROP DISEASE DETECTION")
        logger.info(f"{'='*60}")
        
        # Train the crop disease model
        trained_model, history, best_val_acc = train_crop_disease_model(
            model, model_name, train_loader, val_loader, config, device
        )
        
        # Evaluate on crop disease test data
        trainer = CropDiseaseTrainer(trained_model, model_name, device, config)
        test_metrics, predictions, labels = trainer.evaluate(test_loader)
        
        # Store crop disease detection results
        results[model_name] = {
            'model': trained_model,
            'training_history': history,
            'best_val_acc': best_val_acc,
            'test_metrics': test_metrics,
            'predictions': predictions,
            'labels': labels
        }
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
    
    logger.info(f"\n{'='*60}")
    logger.info("CROP DISEASE DETECTION COMPARISON COMPLETED")
    logger.info(f"{'='*60}")
    
    # Print comparison summary
    print(f"\nCrop Disease Detection Results Summary:")
    print(f"{'Model':<20} {'Val Acc (%)':<12} {'Test Acc (%)':<12} {'F1-Score':<10}")
    print("-" * 60)
    
    for model_name, result in results.items():
        val_acc = result['best_val_acc']
        test_acc = result['test_metrics']['test_accuracy']
        f1_score = result['test_metrics']['f1_score']
        print(f"{model_name:<20} {val_acc:<12.2f} {test_acc:<12.2f} {f1_score:<10.3f}")
    
    return results

if __name__ == "__main__":
    # Test the crop disease trainer
    logger.info("Crop Disease Detection Trainer Module Loaded")
