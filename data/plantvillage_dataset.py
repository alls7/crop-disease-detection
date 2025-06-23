"""
PlantVillage Dataset Pipeline for Crop Disease Detection
Compatible with Colab notebook implementation
Author: Ali SU - Hacettepe University
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantVillageDataset(Dataset):
    """PlantVillage Dataset for crop disease detection - Colab compatible"""
    
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Load crop disease images and labels  
        self.image_paths, self.labels, self.class_names = self._load_dataset()
        self.num_classes = len(self.class_names)
        
        print(f"Loaded PlantVillage: {len(self.image_paths)} images, {self.num_classes} classes")
        
    def _load_dataset(self) -> Tuple[List[Path], List[int], List[str]]:
        """Load PlantVillage dataset structure"""
        
        image_paths = []
        labels = []
        class_names = []
        
        # Find data directory (flexible structure)
        possible_dirs = [
            self.root_dir / "PlantVillage",
            self.root_dir / "plantdisease", 
            self.root_dir / "New Plant Diseases Dataset(Augmented)",
            self.root_dir
        ]
        
        actual_data_dir = None
        for check_dir in possible_dirs:
            if check_dir.exists():
                subdirs = [d for d in check_dir.iterdir() if d.is_dir()]
                if len(subdirs) >= 10:
                    actual_data_dir = check_dir
                    break
        
        if actual_data_dir is None:
            raise ValueError(f"PlantVillage dataset not found in {self.root_dir}")
            
        # Load classes
        class_dirs = sorted([d for d in actual_data_dir.iterdir() if d.is_dir()])
        
        for class_dir in class_dirs:
            class_names.append(class_dir.name)
            
        # Load images
        for class_idx, class_dir in enumerate(class_dirs):
            image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
            class_images = []
            for ext in image_extensions:
                class_images.extend(list(class_dir.glob(ext)))
            
            for img_path in class_images:
                image_paths.append(img_path)
                labels.append(class_idx)
                
        return image_paths, labels, class_names
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get crop disease image and label"""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='white')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_crop_disease_transforms(image_size: int = 224, augment: bool = True):
    """Get transforms for crop disease images - Colab compatible"""
    
    # Base transforms
    base_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    
    if augment:
        # Training transforms with agricultural augmentation
        train_transforms = [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    else:
        train_transforms = base_transforms.copy()
    
    return (
        transforms.Compose(train_transforms),
        transforms.Compose(base_transforms)
    )

def create_plantvillage_loaders(data_dir: str, batch_size: int = 32, 
                               train_split: float = 0.7, val_split: float = 0.15,
                               image_size: int = 224, num_workers: int = 2):
    """Create data loaders - Colab compatible"""
    
    train_transform, val_transform = get_crop_disease_transforms(image_size, augment=True)
    
    # Create dataset
    full_dataset = PlantVillageDataset(root_dir=data_dir, transform=None)
    
    num_classes = full_dataset.num_classes
    class_names = full_dataset.class_names
    total_size = len(full_dataset)
    
    # Calculate splits
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader, num_classes, class_names

# Test function
def test_plantvillage_pipeline(data_dir: str = "data/plantvillage"):
    """Test the pipeline"""
    try:
        train_loader, val_loader, test_loader, num_classes, class_names = create_plantvillage_loaders(data_dir)
        print(f"Success! {num_classes} classes, {len(train_loader)} train batches")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    test_plantvillage_pipeline()
