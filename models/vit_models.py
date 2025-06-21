"""
Vision Transformer model implementations for crop disease detection.
Simple implementation using timm library.
"""

import torch
import torch.nn as nn
import timm
import logging

logger = logging.getLogger(__name__)

class ViTModel(nn.Module):
    """Generic ViT model wrapper"""
    
    def __init__(self, model_name, num_classes=38, pretrained=True):
        super(ViTModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    def forward_features(self, x):
        """Extract features without classification head"""
        if hasattr(self.backbone, 'forward_features'):
            return self.backbone.forward_features(x)
        else:
            return self.backbone(x)
    
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': round(model_size_mb, 2),
            'num_classes': self.num_classes
        }

def create_vit_base_patch16(num_classes=38, pretrained=True):
    """Create ViT-Base/16 model"""
    logger.info("Creating ViT-Base/16 model...")
    model = ViTModel('vit_base_patch16_224', num_classes, pretrained)
    return model

def create_deit_small(num_classes=38, pretrained=True):
    """Create DeiT-Small model"""
    logger.info("Creating DeiT-Small model...")
    model = ViTModel('deit_small_patch16_224', num_classes, pretrained)
    return model

def test_vit_models():
    """Test ViT models"""
    print("Testing ViT models...")
    
    dummy_input = torch.randn(2, 3, 224, 224)
    
    models_to_test = [
        ('ViT-Base/16', create_vit_base_patch16),
        ('DeiT-Small', create_deit_small)
    ]
    
    for model_name, model_fn in models_to_test:
        try:
            model = model_fn(num_classes=38)
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_input)
                features = model.forward_features(dummy_input)
                
            print(f"{model_name}: Forward pass successful, output shape: {output.shape}")
            print(f"{model_name}: Features shape: {features.shape}")
            
            info = model.get_model_info()
            print(f"{model_name}: {info['total_parameters']:,} parameters")
            
            del model, output, features
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"{model_name} test failed: {e}")
    
    print("ViT model tests completed")

def compare_cnn_vs_vit(num_classes=38):
    """Compare CNN and ViT models"""
    from .cnn_models import create_resnet50, create_efficientnet_b0
    
    models = {
        'ResNet-50': create_resnet50(num_classes),
        'EfficientNet-B0': create_efficientnet_b0(num_classes),
        'ViT-Base/16': create_vit_base_patch16(num_classes),
        'DeiT-Small': create_deit_small(num_classes)
    }
    
    comparison = {}
    for name, model in models.items():
        comparison[name] = model.get_model_info()
        del model
        torch.cuda.empty_cache()
    
    return comparison

if __name__ == "__main__":
    test_vit_models()
