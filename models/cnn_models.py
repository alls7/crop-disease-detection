"""
CNN model implementations for crop disease detection.
Simple implementation using timm library.
"""

import torch
import torch.nn as nn
import timm
import logging

logger = logging.getLogger(__name__)

class CNNModel(nn.Module):
    """Generic CNN model wrapper"""
    
    def __init__(self, model_name, num_classes=38, pretrained=True):
        super(CNNModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=num_classes
        )
        
    def forward(self, x):
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

def create_resnet50(num_classes=38, pretrained=True):
    """Create ResNet-50 model"""
    logger.info("Creating ResNet-50 model...")
    model = CNNModel('resnet50', num_classes, pretrained)
    return model

def create_efficientnet_b0(num_classes=38, pretrained=True):
    """Create EfficientNet-B0 model"""
    logger.info("Creating EfficientNet-B0 model...")
    model = CNNModel('efficientnet_b0', num_classes, pretrained)
    return model

def test_cnn_models():
    """Test CNN models"""
    print("Testing CNN models...")
    
    dummy_input = torch.randn(2, 3, 224, 224)
    
    models_to_test = [
        ('ResNet-50', create_resnet50),
        ('EfficientNet-B0', create_efficientnet_b0)
    ]
    
    for model_name, model_fn in models_to_test:
        try:
            model = model_fn(num_classes=38)
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_input)
                
            print(f"{model_name}: Forward pass successful, output shape: {output.shape}")
            
            info = model.get_model_info()
            print(f"{model_name}: {info['total_parameters']:,} parameters")
            
            del model, output
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"{model_name} test failed: {e}")
    
    print("CNN model tests completed")

if __name__ == "__main__":
    test_cnn_models()
