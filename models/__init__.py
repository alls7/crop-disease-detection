"""
Crop Disease Detection Models
Author: Ali SU - Hacettepe University
"""

from .cnn_models import (
    CropDiseaseCNN,
    create_resnet50_for_crop_diseases,
    create_efficientnet_b0_for_crop_diseases
)

from .vit_models import (
    CropDiseaseViT,
    create_vit_base_patch16_for_crop_diseases,
    create_deit_small_for_crop_diseases
)

__all__ = [
    'CropDiseaseCNN',
    'CropDiseaseViT',
    'create_resnet50_for_crop_diseases',
    'create_efficientnet_b0_for_crop_diseases',
    'create_vit_base_patch16_for_crop_diseases',
    'create_deit_small_for_crop_diseases'
]
