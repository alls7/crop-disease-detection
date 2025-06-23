"""
CNN model implementations for crop disease detection on PlantVillage dataset.
Compatible with Colab notebook implementation.
Author: Ali SU - Hacettepe University
"""

import torch
import torch.nn as nn
import timm
import logging
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CropDiseaseCNN(nn.Module):
    """
    CNN model wrapper optimized for crop disease detection
    Compatible with Colab notebook implementation
    """
    
    def __init__(self, 
                 model_name: str, 
                 num_classes: int = 15,
                 pretrained: bool = True,
                 dropout_rate: float = 0.3,
                 fine_tune_strategy: str = 'gradual'):
        """
        Initialize CNN for crop disease detection
        
        Args:
            model_name: CNN architecture (resnet50, efficientnet_b0)
            num_classes: Number of crop disease classes
            pretrained: Use ImageNet pretrained weights
            dropout_rate: Dropout for agricultural domain adaptation
            fine_tune_strategy: 'gradual' or 'full' fine-tuning
        """
        super(CropDiseaseCNN, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.fine_tune_strategy = fine_tune_strategy
        
        # Load backbone with ImageNet weights
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0  # Remove classifier for custom head
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Custom classifier head for crop disease detection
        self.crop_disease_classifier = self._build_crop_disease_head()
        
        # Setup transfer learning
        self._setup_transfer_learning()
        
        print(f"Initialized {model_name} for crop disease detection with {num_classes} classes")
        
    def _build_crop_disease_head(self) -> nn.Module:
        """Build classifier head optimized for crop disease detection"""
        
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
        """Setup transfer learning strategy"""
        if self.fine_tune_strategy == 'gradual':
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()
    
    def freeze_backbone(self):
        """Freeze backbone for feature extraction only"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        for param in self.crop_disease_classifier.parameters():
            param.requires_grad = True
            
        print(f"Frozen {self.model_name} backbone for crop disease feature extraction")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for full fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        print(f"Unfrozen {self.model_name} backbone for crop disease fine-tuning")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for crop disease classification"""
        crop_features = self.backbone(x)
        disease_predictions = self.crop_disease_classifier(crop_features)
        return disease_predictions
    
    def get_crop_disease_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for crop disease analysis"""
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(x)
        return features
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        return {
            'model_name': self.model_name,
            'architecture': 'CNN',
            'task': 'crop_disease_detection',
            'dataset': 'PlantVillage',
            'num_disease_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
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
        print(f"Saved crop disease model to {save_path}")
    
    def load_crop_disease_model(self, load_path: str):
        """Load trained crop disease model"""
        checkpoint = torch.load(load_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded crop disease model from {load_path}")
        return checkpoint.get('metadata', {})

def create_resnet50_for_crop_diseases(num_classes: int = 15, 
                                     pretrained: bool = True,
                                     dropout_rate: float = 0.3,
                                     fine_tune_strategy: str = 'gradual') -> CropDiseaseCNN:
    """Create ResNet-50 optimized for crop disease detection"""
    
    print("Creating ResNet-50 for crop disease detection...")
    
    model = CropDiseaseCNN(
        model_name='resnet50',
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        fine_tune_strategy=fine_tune_strategy
    )
    
    return model

def create_efficientnet_b0_for_crop_diseases(num_classes: int = 15,
                                            pretrained: bool = True,
                                            dropout_rate: float = 0.3,
                                            fine_tune_strategy: str = 'gradual') -> CropDiseaseCNN:
    """Create EfficientNet-B0 optimized for crop disease detection"""
    
    print("Creating EfficientNet-B0 for crop disease detection...")
    
    model = CropDiseaseCNN(
        model_name='efficientnet_b0',
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        fine_tune_strategy=fine_tune_strategy
    )
    
    return model

# Test function
def test_cnn_models_for_crop_diseases():
    """Test CNN models with dummy data"""
    print("Testing CNN models for crop disease detection...")
    
    dummy_images = torch.randn(2, 3, 224, 224)
    
    for model_name, model_fn in [('ResNet-50', create_resnet50_for_crop_diseases),
                                ('EfficientNet-B0', create_efficientnet_b0_for_crop_diseases)]:
        try:
            model = model_fn(num_classes=15)
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_images)
                
            print(f"{model_name}: {dummy_images.shape} -> {output.shape}")
            
            # Test model info
            info = model.get_model_info()
            print(f"  Parameters: {info['total_parameters']:,}")
            print(f"  Size: {info['model_size_mb']} MB")
            
        except Exception as e:
            print(f"{model_name} test failed: {e}")

if __name__ == "__main__":
    test_cnn_models_for_crop_diseases()
