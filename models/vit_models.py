"""
Vision Transformer model implementations for crop disease detection.
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

class CropDiseaseViT(nn.Module):
    """
    Vision Transformer wrapper optimized for crop disease detection
    Compatible with Colab notebook implementation
    """
    
    def __init__(self, 
                 model_name: str, 
                 num_classes: int = 15,
                 pretrained: bool = True,
                 dropout_rate: float = 0.2,
                 attention_dropout: float = 0.1,
                 fine_tune_strategy: str = 'gradual',
                 img_size: int = 224):
        """
        Initialize ViT for crop disease detection
        
        Args:
            model_name: ViT architecture (vit_base_patch16_224, deit_small_patch16_224)
            num_classes: Number of crop disease classes
            pretrained: Use ImageNet-21k pretrained weights
            dropout_rate: Dropout for agricultural domain adaptation
            attention_dropout: Attention dropout for ViT layers
            fine_tune_strategy: 'gradual' or 'full' fine-tuning
            img_size: Input image size
        """
        super(CropDiseaseViT, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.fine_tune_strategy = fine_tune_strategy
        self.img_size = img_size
        
        # Load backbone with ImageNet-21k weights
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            drop_rate=dropout_rate,
            attn_drop_rate=attention_dropout
        )
        
        # Get ViT configuration
        self.embed_dim = self.backbone.embed_dim
        self.num_patches = getattr(self.backbone.patch_embed, 'num_patches', 196)
        self.patch_size = getattr(self.backbone.patch_embed, 'patch_size', (16, 16))
        
        # Custom classifier head for crop disease detection
        self.crop_disease_classifier = self._build_crop_disease_head()
        
        # Setup transfer learning
        self._setup_transfer_learning()
        
        print(f"Initialized {model_name} for crop disease detection with {num_classes} classes")
        
    def _build_crop_disease_head(self) -> nn.Module:
        """Build classifier head optimized for ViT crop disease detection"""
        
        return nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(self.dropout_rate / 2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
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
        """Freeze ViT backbone for feature extraction only"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        for param in self.crop_disease_classifier.parameters():
            param.requires_grad = True
            
        print(f"Frozen {self.model_name} backbone for crop disease feature extraction")
    
    def unfreeze_backbone(self):
        """Unfreeze ViT backbone for full fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        print(f"Unfrozen {self.model_name} backbone for crop disease fine-tuning")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for crop disease classification with ViT"""
        crop_features = self.backbone(x)
        disease_predictions = self.crop_disease_classifier(crop_features)
        return disease_predictions
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract ViT features for crop disease analysis"""
        return self.backbone.forward_features(x)
    
    def analyze_crop_disease_attention(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze ViT attention for crop disease images"""
        
        self.backbone.eval()
        batch_size = x.shape[0]
        
        with torch.no_grad():
            # Get patch embeddings
            if hasattr(self.backbone, 'patch_embed'):
                patch_embeddings = self.backbone.patch_embed(x)
            else:
                # Fallback for different ViT implementations
                patch_embeddings = x  # Simplified
        
        return {
            'patch_embeddings': patch_embeddings,
            'num_patches': self.num_patches,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive ViT model information"""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        # ViT specific information
        num_layers = len(self.backbone.blocks) if hasattr(self.backbone, 'blocks') else 'Unknown'
        num_heads = self.backbone.blocks[0].attn.num_heads if hasattr(self.backbone, 'blocks') and len(self.backbone.blocks) > 0 else 'Unknown'
        
        return {
            'model_name': self.model_name,
            'architecture': 'ViT',
            'task': 'crop_disease_detection',
            'dataset': 'PlantVillage',
            'num_disease_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': round(model_size_mb, 2),
            'embed_dim': self.embed_dim,
            'num_patches': self.num_patches,
            'patch_size': self.patch_size,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'img_size': self.img_size,
            'dropout_rate': self.dropout_rate,
            'attention_dropout': self.attention_dropout,
            'fine_tune_strategy': self.fine_tune_strategy,
            'transfer_learning': 'ImageNet-21k -> PlantVillage'
        }
    
    def save_crop_disease_model(self, save_path: str, metadata: Optional[Dict] = None):
        """Save trained ViT crop disease model"""
        save_dict = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            'metadata': metadata or {}
        }
        torch.save(save_dict, save_path)
        print(f"Saved ViT crop disease model to {save_path}")
    
    def load_crop_disease_model(self, load_path: str):
        """Load trained ViT crop disease model"""
        checkpoint = torch.load(load_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded ViT crop disease model from {load_path}")
        return checkpoint.get('metadata', {})

def create_vit_base_patch16_for_crop_diseases(num_classes: int = 15,
                                             pretrained: bool = True,
                                             dropout_rate: float = 0.2,
                                             attention_dropout: float = 0.1,
                                             fine_tune_strategy: str = 'gradual',
                                             img_size: int = 224) -> CropDiseaseViT:
    """Create ViT-Base/16 optimized for crop disease detection"""
    
    print("Creating ViT-Base/16 for crop disease detection...")
    
    model = CropDiseaseViT(
        model_name='vit_base_patch16_224',
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        attention_dropout=attention_dropout,
        fine_tune_strategy=fine_tune_strategy,
        img_size=img_size
    )
    
    return model

def create_deit_small_for_crop_diseases(num_classes: int = 15,
                                       pretrained: bool = True,
                                       dropout_rate: float = 0.2,
                                       attention_dropout: float = 0.1,
                                       fine_tune_strategy: str = 'gradual',
                                       img_size: int = 224) -> CropDiseaseViT:
    """Create DeiT-Small optimized for crop disease detection"""
    
    print("Creating DeiT-Small for crop disease detection...")
    
    model = CropDiseaseViT(
        model_name='deit_small_patch16_224',
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        attention_dropout=attention_dropout,
        fine_tune_strategy=fine_tune_strategy,
        img_size=img_size
    )
    
    return model

# Test function
def test_vit_models_for_crop_diseases():
    """Test ViT models with dummy data"""
    print("Testing ViT models for crop disease detection...")
    
    dummy_images = torch.randn(2, 3, 224, 224)
    
    for model_name, model_fn in [('ViT-Base/16', create_vit_base_patch16_for_crop_diseases),
                                ('DeiT-Small', create_deit_small_for_crop_diseases)]:
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
            print(f"  Embed Dim: {info['embed_dim']}")
            
        except Exception as e:
            print(f"{model_name} test failed: {e}")

if __name__ == "__main__":
    test_vit_models_for_crop_diseases()
