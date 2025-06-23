"""
Data download utilities for PlantVillage dataset
Compatible with Colab notebook implementation
Author: Ali SU - Hacettepe University
"""

import os
import kaggle
from pathlib import Path
import zipfile
import shutil

def setup_kaggle_api():
    """Setup Kaggle API for dataset download"""
    print("Setting up Kaggle API...")
    
    # Create .kaggle directory
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    # Check if kaggle.json exists
    kaggle_json = kaggle_dir / 'kaggle.json'
    if not kaggle_json.exists():
        print("Please upload your kaggle.json file!")
        print("Download from: Kaggle → Account → Create New API Token")
        return False
    
    # Set permissions
    os.chmod(kaggle_json, 0o600)
    print("Kaggle API configured successfully!")
    return True

def download_plantvillage_dataset(data_dir: str = "data"):
    """Download PlantVillage dataset from Kaggle"""
    
    if not setup_kaggle_api():
        return False
    
    try:
        print("Downloading PlantVillage dataset...")
        
        # Download dataset
        kaggle.api.dataset_download_files(
            'emmarex/plantdisease',
            path=data_dir,
            unzip=True
        )
        
        print(f"Dataset downloaded to {data_dir}")
        
        # Check download
        data_path = Path(data_dir)
        if data_path.exists():
            subdirs = [d for d in data_path.iterdir() if d.is_dir()]
            print(f"Found {len(subdirs)} directories in dataset")
            return True
        else:
            print("Download verification failed")
            return False
            
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def verify_dataset(data_dir: str = "data"):
    """Verify PlantVillage dataset integrity"""
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Dataset directory {data_dir} not found")
        return False
    
    # Count classes and images
    total_images = 0
    class_count = 0
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            class_count += 1
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG'))
            total_images += len(images)
    
    print(f"Dataset verification:")
    print(f"  Classes: {class_count}")
    print(f"  Total images: {total_images}")
    
    # Expected ranges for PlantVillage
    if class_count >= 10 and total_images >= 1000:
        print("Dataset verification: PASSED")
        return True
    else:
        print("Dataset verification: FAILED")
        return False

if __name__ == "__main__":
    # Download and verify dataset
    download_plantvillage_dataset()
    verify_dataset()
