"""
PlantVillage Dataset Pipeline for Crop Disease Detection
Real agricultural data loading and preprocessing
Author: Ali SU - Hacettepe University
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)

class PlantVillageDataset(Dataset):
    """
    PlantVillage Dataset class for crop disease detection
    
    Dataset contains agricultural images with crop diseases:
    - Multiple plant species (Apple, Corn, Tomato, Pepper, etc.)
    - Various disease conditions + healthy samples
    - Real-world agricultural pathology images
    """
    
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        """
        Args:
            root_dir (str): Root directory containing PlantVillage images
            transform: Optional transforms to apply to crop disease images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Load crop disease images and labels
        self.image_paths, self.labels, self.class_names = self._load_crop_disease_dataset()
        self.num_classes = len(self.class_names)
        
        logger.info(f"Loaded PlantVillage dataset: {len(self.image_paths)} crop disease images from {self.num_classes} classes")
        
    def _load_crop_disease_dataset(self) -> Tuple[List[Path], List[int], List[str]]:
        """Load PlantVillage crop disease dataset structure"""
        
        image_paths = []
        labels = []
        class_names = []
        
        # Find PlantVillage data directory (various possible structures)
        possible_dirs = [
            self.root_dir / "PlantVillage",
            self.root_dir / "plantdisease", 
            self.root_dir / "New Plant Diseases Dataset(Augmented)",
            self.root_dir / "color",
            self.root_dir
        ]
        
        actual_data_dir = None
        for check_dir in possible_dirs:
            if check_dir.exists():
                subdirs = [d for d in check_dir.iterdir() if d.is_dir()]
                if len(subdirs) >= 10:  # Should have many crop disease classes
                    actual_data_dir = check_dir
                    break
        
        if actual_data_dir is None:
            raise ValueError(f"PlantVillage dataset not found in {self.root_dir}")
            
        logger.info(f"Found PlantVillage dataset at: {actual_data_dir}")
        
        # Load crop disease classes
        class_dirs = [d for d in actual_data_dir.iterdir() if d.is_dir()]
        class_dirs.sort()  # Ensure consistent class ordering
        
        for class_dir in class_dirs:
            disease_name = class_dir.name
            class_names.append(disease_name)
            
        # Load crop disease images
        for class_idx, class_dir in enumerate(class_dirs):
            # Find all crop disease images in this class
            image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
            class_images = []
            for ext in image_extensions:
                class_images.extend(list(class_dir.glob(ext)))
            
            for img_path in class_images:
                image_paths.append(img_path)
                labels.append(class_idx)
                
        logger.info(f"PlantVillage classes: {class_names}")
        return image_paths, labels, class_names
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get crop disease image and label at index"""
        img_path = self.image_paths[idx]
        disease_label = self.labels[idx]
        
        try:
            # Load crop disease image
            crop_image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading crop disease image {img_path}: {e}")
            # Return blank image if crop disease image loading fails
            crop_image = Image.new('RGB', (224, 224), color='white')
        
        # Apply transforms to crop disease image
        if self.transform:
            crop_image = self.transform(crop_image)
            
        return crop_image, disease_label
    
    def get_disease_distribution(self) -> Dict[str, int]:
        """Get distribution of crop disease samples per class"""
        distribution = {}
        for label in self.labels:
            disease_name = self.class_names[label]
            distribution[disease_name] = distribution.get(disease_name, 0) + 1
        return distribution
    
    def get_crop_types(self) -> List[str]:
        """Extract unique crop types from disease class names"""
        crop_types = set()
        for disease_class in self.class_names:
            # Extract crop name before first underscore or parenthesis
            if '___' in disease_class:
                crop_name = disease_class.split('___')[0]
            elif '(' in disease_class:
                crop_name = disease_class.split('(')[0].strip()
            else:
                crop_name = disease_class.split('_')[0]
            crop_types.add(crop_name)
        return sorted(list(crop_types))

def get_crop_disease_transforms(image_size: int = 224, 
                               augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get data transforms for crop disease images
    
    Args:
        image_size: Target image size for crop disease detection
        augment: Whether to apply data augmentation for crop disease training
        
    Returns:
        Tuple of (train_transform, val_transform) for crop diseases
    """
    
    # Base transforms for crop disease images
    base_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet statistics (good for crop images)
            std=[0.229, 0.224, 0.225]
        )
    ]
    
    if augment:
        # Training transforms with agricultural data augmentation
        crop_disease_train_transforms = [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),  # Common for crop diseases
            transforms.RandomVerticalFlip(p=0.3),    # Less common but useful
            transforms.RandomRotation(degrees=15),    # Moderate rotation for crops
            transforms.ColorJitter(
                brightness=0.2,  # Lighting conditions in agriculture
                contrast=0.2,    # Different camera conditions
                saturation=0.1,  # Preserve crop disease colors
                hue=0.05        # Minimal hue change for diseases
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    else:
        crop_disease_train_transforms = base_transforms.copy()
    
    # Validation transforms for crop diseases (no augmentation)
    crop_disease_val_transforms = base_transforms.copy()
    
    return (
        transforms.Compose(crop_disease_train_transforms),
        transforms.Compose(crop_disease_val_transforms)
    )

def create_plantvillage_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    image_size: int = 224,
    num_workers: int = 2,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, int, List[str]]:
    """
    Create train, validation, and test data loaders for PlantVillage crop diseases
    
    Args:
        data_dir: Directory containing PlantVillage crop disease dataset
        batch_size: Batch size for crop disease data loaders
        train_split: Proportion for crop disease training set
        val_split: Proportion for crop disease validation set
        test_split: Proportion for crop disease test set
        image_size: Target image size for crop disease detection
        num_workers: Number of workers for crop disease data loading
        augment: Whether to apply data augmentation for crop diseases
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes, class_names)
    """
    
    # Get crop disease transforms
    train_transform, val_transform = get_crop_disease_transforms(image_size, augment)
    
    # Create full PlantVillage crop disease dataset
    full_dataset = PlantVillageDataset(
        root_dir=data_dir,
        transform=None  # We'll set transforms later for each split
    )
    
    num_classes = full_dataset.num_classes
    class_names = full_dataset.class_names
    total_size = len(full_dataset)
    
    # Calculate split sizes for crop disease data
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    logger.info(f"PlantVillage splits: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Split crop disease dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible splits
    )
    
    # Apply transforms to each crop disease split
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Create crop disease data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader, num_classes, class_names

def analyze_plantvillage_dataset(data_dir: str) -> Dict:
    """
    Analyze PlantVillage crop disease dataset statistics
    
    Args:
        data_dir: Directory containing PlantVillage dataset
        
    Returns:
        Dictionary with crop disease dataset statistics
    """
    
    # Create dataset without transforms for analysis
    dataset = PlantVillageDataset(root_dir=data_dir)
    
    # Get crop disease distribution
    disease_dist = dataset.get_disease_distribution()
    crop_types = dataset.get_crop_types()
    
    # Calculate crop disease statistics
    total_images = len(dataset)
    num_classes = dataset.num_classes
    min_samples = min(disease_dist.values())
    max_samples = max(disease_dist.values())
    avg_samples = total_images / num_classes
    
    analysis = {
        'total_crop_disease_images': total_images,
        'num_disease_classes': num_classes,
        'disease_class_names': dataset.class_names,
        'crop_types': crop_types,
        'disease_distribution': disease_dist,
        'min_samples_per_disease': min_samples,
        'max_samples_per_disease': max_samples,
        'avg_samples_per_disease': round(avg_samples, 2),
        'dataset_balanced': max_samples / min_samples < 3.0  # Balanced if ratio < 3
    }
    
    return analysis

def test_plantvillage_pipeline(data_dir: str = "data/plantvillage"):
    """
    Test the complete PlantVillage crop disease pipeline
    
    Args:
        data_dir: Directory containing PlantVillage crop disease dataset
    """
    
    logger.info("Testing PlantVillage crop disease pipeline...")
    
    try:
        # Check if crop disease dataset exists
        if not Path(data_dir).exists():
            logger.error(f"PlantVillage dataset not found at {data_dir}")
            logger.info("Please download PlantVillage crop disease dataset first")
            return False
        
        # Analyze crop disease dataset
        logger.info("Analyzing PlantVillage crop disease dataset...")
        analysis = analyze_plantvillage_dataset(data_dir)
        
        print(f"PlantVillage Crop Disease Dataset Analysis:")
        print(f"- Total crop disease images: {analysis['total_crop_disease_images']:,}")
        print(f"- Number of disease classes: {analysis['num_disease_classes']}")
        print(f"- Crop types: {', '.join(analysis['crop_types'])}")
        print(f"- Balanced dataset: {analysis['dataset_balanced']}")
        print(f"- Samples per disease class: {analysis['min_samples_per_disease']} - {analysis['max_samples_per_disease']}")
        
        # Test crop disease data loaders
        logger.info("Creating PlantVillage crop disease data loaders...")
        train_loader, val_loader, test_loader, num_classes, class_names = create_plantvillage_data_loaders(
            data_dir=data_dir,
            batch_size=16,  # Small batch for testing
            num_workers=2
        )
        
        print(f"PlantVillage data loaders created:")
        print(f"- Train batches: {len(train_loader)}")
        print(f"- Val batches: {len(val_loader)}")
        print(f"- Test batches: {len(test_loader)}")
        print(f"- Number of crop disease classes: {num_classes}")
        
        # Test loading a batch of crop disease images
        logger.info("Testing crop disease batch loading...")
        for crop_images, disease_labels in train_loader:
            print(f"Crop disease batch shape: {crop_images.shape}")
            print(f"Disease label shape: {disease_labels.shape}")
            print(f"Crop image range: [{crop_images.min():.3f}, {crop_images.max():.3f}]")
            print(f"Disease labels in batch: {disease_labels.unique().tolist()}")
            break
        
        logger.info("PlantVillage crop disease pipeline test successful!")
        return True
        
    except Exception as e:
        logger.error(f"PlantVillage pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test the PlantVillage crop disease pipeline
    test_plantvillage_pipeline()
