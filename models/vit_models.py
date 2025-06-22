"""
Vision Transformer model implementations for crop disease detection on PlantVillage dataset.
Optimized for agricultural pathology with attention mechanisms for plant disease classification.
Author: Ali SU - Hacettepe University
"""

import torch
import torch.nn as nn
import timm
import logging
from typing import Dict, Any, Optional, List, Tuple
import math
from pathlib import Path

logger = logging.getLogger(__name__)

class CropDiseaseViT(nn.Module):
    """
    Vision Transformer model wrapper optimized for crop disease detection
    Designed specifically for PlantVillage agricultural dataset with attention mechanisms
    """
    
    def __init__(self, 
                 model_name: str, 
                 num_classes: int = 15,  # PlantVillage subset classes
                 pretrained: bool = True,
                 dropout_rate: float = 0.2,
                 attention_dropout: float = 0.1,
                 fine_tune_strategy: str = 'gradual',
                 img_size: int = 224):
        """
        Initialize ViT for crop disease detection
        
        Args:
            model_name: ViT architecture name (vit_base_patch16_224, deit_small_patch16_224)
            num_classes: Number of crop disease classes
            pretrained: Use ImageNet-21k pretrained weights for agricultural transfer
            dropout_rate: Dropout for agricultural domain adaptation
            attention_dropout: Attention dropout for ViT layers
            fine_tune_strategy: 'gradual' or 'full' fine-tuning for crop diseases
            img_size: Input image size for crop disease images
        """
        super(CropDiseaseViT, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.fine_tune_strategy = fine_tune_strategy
        self.img_size = img_size
        
        # Load backbone with ImageNet-21k weights for agricultural transfer learning
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier for custom crop disease head
            img_size=img_size,
            drop_rate=dropout_rate,
            attn_drop_rate=attention_dropout
        )
        
        # Get ViT configuration for crop disease adaptation
        self.embed_dim = self.backbone.embed_dim
        self.num_patches = self.backbone.patch_embed.num_patches
        self.patch_size = self.backbone.patch_embed.patch_size
        
        # Custom classifier head for crop disease detection
        self.crop_disease_classifier = self._build_crop_disease_head()
        
        # Initialize for agricultural transfer learning
        self._setup_transfer_learning()
        
        logger.info(f"Initialized {model_name} for crop disease detection with {num_classes} disease classes")
        logger.info(f"ViT config: {self.embed_dim}D embeddings, {self.num_patches} patches, {self.patch_size} patch size")
        
    def _build_crop_disease_head(self) -> nn.Module:
        """Build classifier head optimized for crop disease detection with ViT features"""
        
        # Agricultural-specific classifier with attention pooling
        return nn.Sequential(
            nn.LayerNorm(self.embed_dim),  # ViT standard normalization
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),  # GELU activation for transformers
            nn.LayerNorm(512),
            nn.Dropout(self.dropout_rate / 2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(self.dropout_rate / 4),
            nn.Linear(256, self.num_classes)
        )
    
    def _setup_transfer_learning(self):
        """Setup transfer learning strategy for crop disease detection"""
        
        if self.fine_tune_strategy == 'gradual':
            # Start with frozen transformer blocks for agricultural adaptation
            self.freeze_backbone()
            logger.info("Starting with frozen ViT backbone for crop disease adaptation")
        else:
            # Full fine-tuning from start
            self.unfreeze_backbone()
            logger.info("Using full ViT fine-tuning for crop disease detection")
    
    def freeze_backbone(self):
        """Freeze ViT backbone for feature extraction only (Stage 1 of transfer learning)"""
        
        # Freeze patch embedding
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False
            
        # Freeze positional embeddings
        if hasattr(self.backbone, 'pos_embed'):
            self.backbone.pos_embed.requires_grad = False
        if hasattr(self.backbone, 'cls_token'):
            self.backbone.cls_token.requires_grad = False
            
        # Freeze transformer blocks
        for block in self.backbone.blocks:
            for param in block.parameters():
                param.requires_grad = False
                
        # Freeze normalization
        if hasattr(self.backbone, 'norm'):
            for param in self.backbone.norm.parameters():
                param.requires_grad = False
        
        # Keep crop disease classifier trainable
        for param in self.crop_disease_classifier.parameters():
            param.requires_grad = True
            
        logger.info(f"Frozen {self.model_name} backbone for crop disease feature extraction")
    
    def unfreeze_backbone(self):
        """Unfreeze ViT backbone for full fine-tuning (Stage 2 of transfer learning)"""
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        logger.info(f"Unfrozen {self.model_name} backbone for crop disease fine-tuning")
    
    def unfreeze_top_layers(self, num_layers: int = 3):
        """Unfreeze only top transformer layers for gradual fine-tuning"""
        
        # Unfreeze last few transformer blocks for crop disease adaptation
        total_blocks = len(self.backbone.blocks)
        layers_to_unfreeze = self.backbone.blocks[max(0, total_blocks - num_layers):]
        
        for block in layers_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True
                
        # Also unfreeze final norm layer
        if hasattr(self.backbone, 'norm'):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
                
        logger.info(f"Unfrozen top {num_layers} transformer layers for crop disease adaptation")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for crop disease classification with ViT
        
        Args:
            x: Batch of crop disease images [B, 3, 224, 224]
            
        Returns:
            Crop disease predictions [B, num_classes]
        """
        # Extract ViT features for crop diseases
        crop_features = self.backbone(x)  # [B, embed_dim]
        
        # Classify crop diseases using attention features
        disease_predictions = self.crop_disease_classifier(crop_features)
        
        return disease_predictions
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract ViT features for crop disease analysis without classification"""
        return self.backbone.forward_features(x)
    
    def get_attention_maps(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Extract attention maps from ViT for crop disease region analysis
        
        Args:
            x: Crop disease images [B, 3, 224, 224]
            layer_idx: Transformer layer index (-1 for last layer)
            
        Returns:
            Attention maps showing disease-relevant regions
        """
        # This requires hooks to extract attention weights
        # Simplified implementation for now
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone.forward_features(x)
        return features
    
    def analyze_crop_disease_attention(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze what ViT attention focuses on in crop disease images
        
        Args:
            x: Crop disease images
            
        Returns:
            Dictionary with attention analysis for agricultural interpretation
        """
        
        self.backbone.eval()
        batch_size = x.shape[0]
        
        with torch.no_grad():
            # Get patch embeddings for crop disease images
            patch_embeddings = self.backbone.patch_embed(x)  # [B, num_patches, embed_dim]
            
            # Add positional embeddings
            if hasattr(self.backbone, 'pos_embed'):
                patch_embeddings = patch_embeddings + self.backbone.pos_embed[:, 1:, :]
            
            # Add class token for crop disease classification
            if hasattr(self.backbone, 'cls_token'):
                cls_tokens = self.backbone.cls_token.expand(batch_size, -1, -1)
                x_tokens = torch.cat((cls_tokens, patch_embeddings), dim=1)
            else:
                x_tokens = patch_embeddings
        
        return {
            'patch_embeddings': patch_embeddings,
            'total_tokens': x_tokens,
            'num_patches': self.num_patches,
            'patch_size': self.patch_size
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive ViT model information for crop disease detection"""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        # Model size estimation
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
            'frozen_parameters': frozen_params,
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
        logger.info(f"Saved ViT crop disease model to {save_path}")
    
    def load_crop_disease_model(self, load_path: str):
        """Load trained ViT crop disease model"""
        
        checkpoint = torch.load(load_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded ViT crop disease model from {load_path}")
        
        return checkpoint.get('metadata', {})

def create_vit_base_patch16_for_crop_diseases(num_classes: int = 15,
                                             pretrained: bool = True,
                                             dropout_rate: float = 0.2,
                                             attention_dropout: float = 0.1,
                                             fine_tune_strategy: str = 'gradual',
                                             img_size: int = 224) -> CropDiseaseViT:
    """
    Create ViT-Base/16 model optimized for crop disease detection
    
    ViT-Base with 16x16 patches excels at:
    - Global crop disease pattern recognition via self-attention
    - Complex agricultural pathology understanding
    - Long-range dependencies in plant disease symptoms
    
    Args:
        num_classes: Number of crop disease classes in PlantVillage
        pretrained: Use ImageNet-21k pretrained weights for agricultural transfer
        dropout_rate: Dropout for agricultural domain adaptation
        attention_dropout: Attention dropout for ViT robustness
        fine_tune_strategy: Transfer learning strategy for crop diseases
        img_size: Input image size for crop disease detection
        
    Returns:
        CropDiseaseViT model optimized for agricultural pathology
    """
    
    logger.info("Creating ViT-Base/16 for crop disease detection...")
    
    model = CropDiseaseViT(
        model_name='vit_base_patch16_224',
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        attention_dropout=attention_dropout,
        fine_tune_strategy=fine_tune_strategy,
        img_size=img_size
    )
    
    info = model.get_model_info()
    logger.info(f"ViT-Base/16 Crop Disease Model: {info['total_parameters']:,} params, "
                f"{info['model_size_mb']} MB")
    logger.info(f"ViT Architecture: {info['num_layers']} layers, {info['num_heads']} heads, "
                f"{info['embed_dim']}D embeddings")
    
    return model

def create_deit_small_for_crop_diseases(num_classes: int = 15,
                                       pretrained: bool = True,
                                       dropout_rate: float = 0.2,
                                       attention_dropout: float = 0.1,
                                       fine_tune_strategy: str = 'gradual',
                                       img_size: int = 224) -> CropDiseaseViT:
    """
    Create DeiT-Small model optimized for crop disease detection
    
    DeiT-Small with knowledge distillation is perfect for:
    - Efficient crop disease classification with limited agricultural data
    - Mobile deployment in farming environments
    - Data-efficient learning for new crop disease types
    
    Args:
        num_classes: Number of crop disease classes in PlantVillage
        pretrained: Use ImageNet pretrained weights for agricultural transfer
        dropout_rate: Dropout for agricultural domain adaptation
        attention_dropout: Attention dropout for ViT robustness
        fine_tune_strategy: Transfer learning strategy for crop diseases
        img_size: Input image size for crop disease detection
        
    Returns:
        CropDiseaseViT model optimized for efficient agricultural deployment
    """
    
    logger.info("Creating DeiT-Small for crop disease detection...")
    
    model = CropDiseaseViT(
        model_name='deit_small_patch16_224',
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        attention_dropout=attention_dropout,
        fine_tune_strategy=fine_tune_strategy,
        img_size=img_size
    )
    
    info = model.get_model_info()
    logger.info(f"DeiT-Small Crop Disease Model: {info['total_parameters']:,} params, "
                f"{info['model_size_mb']} MB")
    logger.info(f"DeiT Architecture: {info['num_layers']} layers, {info['num_heads']} heads, "
                f"{info['embed_dim']}D embeddings")
    
    return model

def compare_vit_architectures_for_crop_diseases(num_classes: int = 15) -> Dict[str, Dict]:
    """
    Compare ViT architectures for crop disease detection task
    
    Args:
        num_classes: Number of crop disease classes
        
    Returns:
        Comparison dictionary with ViT model statistics
    """
    
    logger.info("Comparing ViT architectures for crop disease detection...")
    
    models = {
        'ViT-Base/16': create_vit_base_patch16_for_crop_diseases(num_classes),
        'DeiT-Small': create_deit_small_for_crop_diseases(num_classes)
    }
    
    comparison = {}
    for name, model in models.items():
        comparison[name] = model.get_model_info()
        
        # Clean up memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return comparison

def compare_cnn_vs_vit_for_crop_diseases(num_classes: int = 15) -> Dict[str, Dict]:
    """
    Compare CNN vs ViT models for crop disease detection
    
    Args:
        num_classes: Number of crop disease classes
        
    Returns:
        Comprehensive comparison of CNN vs ViT for agricultural tasks
    """
    
    logger.info("Comparing CNN vs ViT for crop disease detection...")
    
    # Import CNN models
    from .cnn_models import create_resnet50_for_crop_diseases, create_efficientnet_b0_for_crop_diseases
    
    models = {
        'ResNet-50 (CNN)': create_resnet50_for_crop_diseases(num_classes),
        'EfficientNet-B0 (CNN)': create_efficientnet_b0_for_crop_diseases(num_classes),
        'ViT-Base/16 (ViT)': create_vit_base_patch16_for_crop_diseases(num_classes),
        'DeiT-Small (ViT)': create_deit_small_for_crop_diseases(num_classes)
    }
    
    comparison = {}
    for name, model in models.items():
        comparison[name] = model.get_model_info()
        
        # Clean up memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return comparison

def test_vit_models_on_crop_diseases():
    """Test ViT models with dummy crop disease data"""
    
    print("Testing ViT models for crop disease detection...")
    
    # Dummy crop disease images (batch_size=4, channels=3, height=224, width=224)
    dummy_crop_images = torch.randn(4, 3, 224, 224)
    
    models_to_test = [
        ('ViT-Base/16', create_vit_base_patch16_for_crop_diseases),
        ('DeiT-Small', create_deit_small_for_crop_diseases)
    ]
    
    for model_name, model_fn in models_to_test:
        try:
            # Create ViT crop disease model
            model = model_fn(num_classes=15)  # PlantVillage subset
            model.eval()
            
            # Test forward pass with crop disease images
            with torch.no_grad():
                disease_predictions = model(dummy_crop_images)
                crop_features = model.forward_features(dummy_crop_images)
                
            expected_pred_shape = (4, 15)  # batch_size=4, num_disease_classes=15
            assert disease_predictions.shape == expected_pred_shape, f"Expected {expected_pred_shape}, got {disease_predictions.shape}"
            
            print(f"{model_name}: Crop disease detection successful, output shape: {disease_predictions.shape}")
            print(f"  Features shape: {crop_features.shape}")
            
            # Test transfer learning functions
            print(f"  Testing ViT transfer learning for {model_name}...")
            model.freeze_backbone()
            model.unfreeze_top_layers(3)
            model.unfreeze_backbone()
            
            # Test attention analysis
            attention_info = model.analyze_crop_disease_attention(dummy_crop_images)
            print(f"  Attention analysis: {attention_info['num_patches']} patches, "
                  f"patch size {attention_info['patch_size']}")
            
            # Test model info
            info = model.get_model_info()
            print(f"  {model_name}: {info['total_parameters']:,} parameters, "
                  f"{info['model_size_mb']} MB")
            print(f"  ViT config: {info['num_layers']} layers, {info['num_heads']} heads, "
                  f"{info['embed_dim']}D embeddings")
            print(f"  Transfer learning: {info['transfer_learning']}")
            
            # Clean up
            del model, disease_predictions, crop_features
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"{model_name} test failed: {e}")
            raise e
    
    print("All ViT crop disease models tested successfully!")

# Backward compatibility functions (for existing imports)
def create_vit_base_patch16(num_classes=15, pretrained=True):
    """Backward compatible ViT-Base/16 creation"""
    return create_vit_base_patch16_for_crop_diseases(num_classes, pretrained)

def create_deit_small(num_classes=15, pretrained=True):
    """Backward compatible DeiT-Small creation"""
    return create_deit_small_for_crop_diseases(num_classes, pretrained)

def test_vit_models():
    """Backward compatible test function"""
    return test_vit_models_on_crop_diseases()

if __name__ == "__main__":
    # Test ViT models for crop disease detection
    test_vit_models_on_crop_diseases()
    
    # Show ViT architecture comparison for crop diseases
    print("\n" + "="*60)
    print("VIT ARCHITECTURE COMPARISON FOR CROP DISEASE DETECTION")
    print("="*60)
    
    comparison = compare_vit_architectures_for_crop_diseases()
    for model_name, stats in comparison.items():
        print(f"\n{model_name} (Crop Disease Detection):")
        print(f"  Parameters: {stats['total_parameters']:,}")
        print(f"  Model Size: {stats['model_size_mb']} MB")
        print(f"  Embed Dim: {stats['embed_dim']}, Layers: {stats['num_layers']}")
        print(f"  Patches: {stats['num_patches']}, Patch Size: {stats['patch_size']}")
        print(f"  Task: {stats['task']}")
        print(f"  Transfer Learning: {stats['transfer_learning']}")
    
    # CNN vs ViT comparison for crop diseases
    print("\n" + "="*60)
    print("CNN vs VIT COMPARISON FOR CROP DISEASE DETECTION")
    print("="*60)
    
    try:
        full_comparison = compare_cnn_vs_vit_for_crop_diseases()
        for model_name, stats in full_comparison.items():
            print(f"\n{model_name}:")
            print(f"  Parameters: {stats['total_parameters']:,}")
            print(f"  Model Size: {stats['model_size_mb']} MB")
            print(f"  Architecture: {stats['architecture']}")
            print(f"  Task: {stats['task']}")
    except ImportError:
        print("CNN models not available for comparison")
