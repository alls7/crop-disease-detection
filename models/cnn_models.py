"""
CNN model implementations for crop disease detection on PlantVillage dataset.
Optimized for agricultural pathology and plant disease classification.
Author: Ali SU - Hacettepe University
"""

import torch
import torch.nn as nn
import timm
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class CropDiseaseCNN(nn.Module):
    """
    CNN model wrapper optimized for crop disease detection
    Designed specifically for PlantVillage agricultural dataset
    """
    
    def __init__(self, 
                 model_name: str, 
                 num_classes: int = 15,  # PlantVillage subset classes
                 pretrained: bool = True,
                 dropout_rate: float = 0.3,
                 fine_tune_strategy: str = 'gradual'):
        """
        Initialize CNN for crop disease detection
        
        Args:
            model_name: CNN architecture name (resnet50, efficientnet_b0)
            num_classes: Number of crop disease classes
            pretrained: Use ImageNet pretrained weights for transfer learning
            dropout_rate: Dropout for agricultural domain adaptation
            fine_tune_strategy: 'gradual' or 'full' fine-tuning
        """
        super(CropDiseaseCNN, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.fine_tune_strategy = fine_tune_strategy
        
        # Load backbone with ImageNet weights for agricultural transfer learning
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0  # Remove classifier for custom crop disease head
        )
        
        # Get feature dimension for crop disease classifier
        self.feature_dim = self.backbone.num_features
        
        # Custom classifier head for crop disease detection
        self.crop_disease_classifier = self._build_crop_disease_head()
        
        # Initialize for agricultural transfer learning
        self._setup_transfer_learning()
        
        logger.info(f"Initialized {model_name} for crop disease detection with {num_classes} disease classes")
        
    def _build_crop_disease_head(self) -> nn.Module:
        """Build classifier head optimized for crop disease detection"""
        
        # Multi-layer classifier for complex agricultural patterns
        return nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(self.dropout_rate / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(self.dropout_rate / 4),
            nn.Linear(256, self.num_classes)
        )
    
    def _setup_transfer_learning(self):
        """Setup transfer learning strategy for crop disease detection"""
        
        if self.fine_tune_strategy == 'gradual':
            # Start with frozen backbone for agricultural adaptation
            self.freeze_backbone()
            logger.info("Starting with frozen backbone for crop disease adaptation")
        else:
            # Full fine-tuning from start
            self.unfreeze_backbone()
            logger.info("Using full fine-tuning for crop disease detection")
    
    def freeze_backbone(self):
        """Freeze backbone for feature extraction only (Stage 1 of transfer learning)"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Keep crop disease classifier trainable
        for param in self.crop_disease_classifier.parameters():
            param.requires_grad = True
            
        logger.info(f"Frozen {self.model_name} backbone for crop disease feature extraction")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for full fine-tuning (Stage 2 of transfer learning)"""
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        logger.info(f"Unfrozen {self.model_name} backbone for crop disease fine-tuning")
    
    def unfreeze_top_layers(self, num_layers: int = 2):
        """Unfreeze only top layers for gradual fine-tuning"""
        
        # This implementation depends on the specific architecture
        # For ResNet: unfreeze last few blocks
        # For EfficientNet: unfreeze last few MB blocks
        
        if 'resnet' in self.model_name.lower():
            # Unfreeze last ResNet blocks for crop disease adaptation
            layers_to_unfreeze = [self.backbone.layer4, self.backbone.layer3][:num_layers]
        elif 'efficientnet' in self.model_name.lower():
            # Unfreeze last EfficientNet blocks for crop disease adaptation  
            all_blocks = list(self.backbone.blocks)
            layers_to_unfreeze = all_blocks[-num_layers:]
        else:
            # Generic approach
            all_params = list(self.backbone.parameters())
            layers_to_unfreeze = all_params[-num_layers*10:]  # Approximate
            
        for layer in layers_to_unfreeze:
            if hasattr(layer, 'parameters'):
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                layer.requires_grad = True
                
        logger.info(f"Unfrozen top {num_layers} layers for crop disease adaptation")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for crop disease classification
        
        Args:
            x: Batch of crop disease images [B, 3, 224, 224]
            
        Returns:
            Crop disease predictions [B, num_classes]
        """
        # Extract features using backbone
        crop_features = self.backbone(x)
        
        # Classify crop diseases
        disease_predictions = self.crop_disease_classifier(crop_features)
        
        return disease_predictions
    
    def get_crop_disease_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for crop disease analysis"""
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(x)
        return features
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information for crop disease detection"""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        # Model size estimation
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        return {
            'model_name': self.model_name,
            'architecture': 'CNN',
            'task': 'crop_disease_detection',
            'dataset': 'PlantVillage',
            'num_disease_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'model_size_mb': round(model_size_mb, 2),
            'feature_dim': self.feature_dim,
            'dropout_rate': self.dropout_rate,
            'fine_tune_strategy': self.fine_tune_strategy,
            'transfer_learning': 'ImageNet -> PlantVillage'
        }
    
    def save_crop_disease_model(self, save_path: str, metadata: Optional[Dict] = None):
        """Save trained crop disease model"""
        
        save_dict = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            'metadata': metadata or {}
        }
        
        torch.save(save_dict, save_path)
        logger.info(f"Saved crop disease model to {save_path}")
    
    def load_crop_disease_model(self, load_path: str):
        """Load trained crop disease model"""
        
        checkpoint = torch.load(load_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded crop disease model from {load_path}")
        
        return checkpoint.get('metadata', {})

def create_resnet50_for_crop_diseases(num_classes: int = 15, 
                                     pretrained: bool = True,
                                     dropout_rate: float = 0.3,
                                     fine_tune_strategy: str = 'gradual') -> CropDiseaseCNN:
    """
    Create ResNet-50 model optimized for crop disease detection
    
    ResNet-50 with deep residual connections is excellent for:
    - Complex crop disease pattern recognition
    - Agricultural image feature extraction
    - Transfer learning from ImageNet to PlantVillage
    
    Args:
        num_classes: Number of crop disease classes in PlantVillage
        pretrained: Use ImageNet pretrained weights for agricultural transfer
        dropout_rate: Dropout for agricultural domain adaptation
        fine_tune_strategy: Transfer learning strategy for crop diseases
        
    Returns:
        CropDiseaseCNN model optimized for agricultural pathology
    """
    
    logger.info("Creating ResNet-50 for crop disease detection...")
    
    model = CropDiseaseCNN(
        model_name='resnet50',
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        fine_tune_strategy=fine_tune_strategy
    )
    
    info = model.get_model_info()
    logger.info(f"ResNet-50 Crop Disease Model: {info['total_parameters']:,} params, "
                f"{info['model_size_mb']} MB")
    
    return model

def create_efficientnet_b0_for_crop_diseases(num_classes: int = 15,
                                            pretrained: bool = True,
                                            dropout_rate: float = 0.3,
                                            fine_tune_strategy: str = 'gradual') -> CropDiseaseCNN:
    """
    Create EfficientNet-B0 model optimized for crop disease detection
    
    EfficientNet-B0 with compound scaling is ideal for:
    - Efficient crop disease classification
    - Mobile deployment in agricultural settings
    - Balanced accuracy vs computational cost for farms
    
    Args:
        num_classes: Number of crop disease classes in PlantVillage
        pretrained: Use ImageNet pretrained weights for agricultural transfer
        dropout_rate: Dropout for agricultural domain adaptation
        fine_tune_strategy: Transfer learning strategy for crop diseases
        
    Returns:
        CropDiseaseCNN model optimized for efficient agricultural deployment
    """
    
    logger.info("Creating EfficientNet-B0 for crop disease detection...")
    
    model = CropDiseaseCNN(
        model_name='efficientnet_b0',
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        fine_tune_strategy=fine_tune_strategy
    )
    
    info = model.get_model_info()
    logger.info(f"EfficientNet-B0 Crop Disease Model: {info['total_parameters']:,} params, "
                f"{info['model_size_mb']} MB")
    
    return model

def compare_cnn_architectures_for_crop_diseases(num_classes: int = 15) -> Dict[str, Dict]:
    """
    Compare CNN architectures for crop disease detection task
    
    Args:
        num_classes: Number of crop disease classes
        
    Returns:
        Comparison dictionary with model statistics
    """
    
    logger.info("Comparing CNN architectures for crop disease detection...")
    
    models = {
        'ResNet-50': create_resnet50_for_crop_diseases(num_classes),
        'EfficientNet-B0': create_efficientnet_b0_for_crop_diseases(num_classes)
    }
    
    comparison = {}
    for name, model in models.items():
        comparison[name] = model.get_model_info()
        
        # Clean up memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return comparison

def test_cnn_models_on_crop_diseases():
    """Test CNN models with dummy crop disease data"""
    
    print("Testing CNN models for crop disease detection...")
    
    # Dummy crop disease images (batch_size=4, channels=3, height=224, width=224)
    dummy_crop_images = torch.randn(4, 3, 224, 224)
    
    models_to_test = [
        ('ResNet-50', create_resnet50_for_crop_diseases),
        ('EfficientNet-B0', create_efficientnet_b0_for_crop_diseases)
    ]
    
    for model_name, model_fn in models_to_test:
        try:
            # Create crop disease model
            model = model_fn(num_classes=15)  # PlantVillage subset
            model.eval()
            
            # Test forward pass with crop disease images
            with torch.no_grad():
                disease_predictions = model(dummy_crop_images)
                
            expected_shape = (4, 15)  # batch_size=4, num_disease_classes=15
            assert disease_predictions.shape == expected_shape, f"Expected {expected_shape}, got {disease_predictions.shape}"
            
            print(f"{model_name}: Crop disease detection successful, output shape: {disease_predictions.shape}")
            
            # Test transfer learning functions
            print(f"  Testing transfer learning for {model_name}...")
            model.freeze_backbone()
            model.unfreeze_top_layers(2)
            model.unfreeze_backbone()
            
            # Test model info
            info = model.get_model_info()
            print(f"  {model_name}: {info['total_parameters']:,} parameters, "
                  f"{info['model_size_mb']} MB")
            print(f"  Transfer learning: {info['transfer_learning']}")
            
            # Clean up
            del model, disease_predictions
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"{model_name} test failed: {e}")
            raise e
    
    print("All CNN crop disease models tested successfully!")

# Backward compatibility functions (for existing imports)
def create_resnet50(num_classes=15, pretrained=True):
    """Backward compatible ResNet-50 creation"""
    return create_resnet50_for_crop_diseases(num_classes, pretrained)

def create_efficientnet_b0(num_classes=15, pretrained=True):
    """Backward compatible EfficientNet-B0 creation"""
    return create_efficientnet_b0_for_crop_diseases(num_classes, pretrained)

def test_cnn_models():
    """Backward compatible test function"""
    return test_cnn_models_on_crop_diseases()

if __name__ == "__main__":
    # Test CNN models for crop disease detection
    test_cnn_models_on_crop_diseases()
    
    # Show architecture comparison for crop diseases
    print("\n" + "="*60)
    print("CNN ARCHITECTURE COMPARISON FOR CROP DISEASE DETECTION")
    print("="*60)
    
    comparison = compare_cnn_architectures_for_crop_diseases()
    for model_name, stats in comparison.items():
        print(f"\n{model_name} (Crop Disease Detection):")
        print(f"  Parameters: {stats['total_parameters']:,}")
        print(f"  Model Size: {stats['model_size_mb']} MB")
        print(f"  Feature Dim: {stats['feature_dim']}")
        print(f"  Task: {stats['task']}")
        print(f"  Transfer Learning: {stats['transfer_learning']}")
