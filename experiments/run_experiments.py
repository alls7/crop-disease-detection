"""
Main Experiment Runner for Crop Disease Detection
CNN vs ViT Comparison on PlantVillage Agricultural Data
Author: Ali SU - Hacettepe University
"""

import argparse
import yaml
import torch
import sys
from pathlib import Path
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules
from models.cnn_models import create_resnet50, create_efficientnet_b0
from models.vit_models import create_vit_base_patch16, create_deit_small
from data.data_loader import create_plantvillage_data_loaders, analyze_plantvillage_dataset
from training.trainer import compare_crop_disease_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration for crop disease detection experiments"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_device() -> torch.device:
    """Setup device for crop disease model training"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        logger.info(f"Using GPU for crop disease detection: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("Using CPU for crop disease detection")
        
    return device

def create_crop_disease_models(num_classes: int) -> dict:
    """Create all models for crop disease detection comparison"""
    
    models = {
        'ResNet-50': create_resnet50(num_classes=num_classes),
        'EfficientNet-B0': create_efficientnet_b0(num_classes=num_classes),
        'ViT-Base/16': create_vit_base_patch16(num_classes=num_classes),
        'DeiT-Small': create_deit_small(num_classes=num_classes)
    }
    
    logger.info("Created crop disease detection models:")
    for name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"  {name}: {param_count:,} parameters")
    
    return models

def visualize_crop_disease_results(results: dict, class_names: list, save_dir: Path):
    """Create visualizations for crop disease detection results"""
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Training curves for crop disease models
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Crop Disease Detection Training Curves', fontsize=16)
    
    for idx, (model_name, result) in enumerate(results.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        history = result['training_history']
        epochs = range(1, len(history['train_acc']) + 1)
        
        ax.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        ax.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
        ax.set_title(f'{model_name} - Crop Disease Detection')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'crop_disease_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Model comparison bar chart
    plt.figure(figsize=(12, 8))
    
    model_names = list(results.keys())
    val_accs = [results[name]['best_val_acc'] for name in model_names]
    test_accs = [results[name]['test_metrics']['test_accuracy'] for name in model_names]
    
    x = range(len(model_names))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], val_accs, width, label='Validation Accuracy', alpha=0.8)
    plt.bar([i + width/2 for i in x], test_accs, width, label='Test Accuracy', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.title('Crop Disease Detection Model Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'crop_disease_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Confusion matrices for best model
    best_model = max(results.keys(), key=lambda x: results[x]['test_metrics']['test_accuracy'])
    best_result = results[best_model]
    
    cm = confusion_matrix(best_result['labels'], best_result['predictions'])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {best_model} (Best Crop Disease Model)')
    plt.xlabel('Predicted Disease')
    plt.ylabel('True Disease')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / f'confusion_matrix_{best_model.lower().replace("-", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def generate_crop_disease_report(results: dict, dataset_info: dict, config: dict, save_dir: Path):
    """Generate comprehensive report for crop disease detection experiments"""
    
    report_path = save_dir / 'crop_disease_detection_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Crop Disease Detection: CNN vs ViT Comparison\n\n")
        f.write("## Dataset Information\n")
        f.write(f"- **Dataset**: PlantVillage\n")
        f.write(f"- **Total Images**: {dataset_info['total_crop_disease_images']:,}\n")
        f.write(f"- **Disease Classes**: {dataset_info['num_disease_classes']}\n")
        f.write(f"- **Crop Types**: {', '.join(dataset_info['crop_types'])}\n")
        f.write(f"- **Balanced Dataset**: {dataset_info['dataset_balanced']}\n\n")
        
        f.write("## Training Configuration\n")
        f.write(f"- **Epochs**: {config['epochs']}\n")
        f.write(f"- **Batch Size**: {config['batch_size']}\n")
        f.write(f"- **Learning Rate**: {config['learning_rate']}\n")
        f.write(f"- **Early Stopping Patience**: {config['early_stopping_patience']}\n\n")
        
        f.write("## Results Summary\n")
        f.write("| Model | Type | Val Acc (%) | Test Acc (%) | Precision | Recall | F1-Score |\n")
        f.write("|-------|------|-------------|--------------|-----------|--------|----------|\n")
        
        for model_name, result in results.items():
            model_type = "CNN" if "Net" in model_name else "ViT"
            val_acc = result['best_val_acc']
            test_metrics = result['test_metrics']
            
            f.write(f"| {model_name} | {model_type} | {val_acc:.2f} | "
                   f"{test_metrics['test_accuracy']:.2f} | "
                   f"{test_metrics['precision']:.3f} | "
                   f"{test_metrics['recall']:.3f} | "
                   f"{test_metrics['f1_score']:.3f} |\n")
        
        # Find best model
        best_model = max(results.keys(), key=lambda x: results[x]['test_metrics']['test_accuracy'])
        best_acc = results[best_model]['test_metrics']['test_accuracy']
        
        f.write(f"\n## Key Findings\n")
        f.write(f"- **Best Model**: {best_model} ({best_acc:.2f}% test accuracy)\n")
        f.write(f"- **Model Type Winner**: {'ViT' if 'ViT' in best_model or 'DeiT' in best_model else 'CNN'}\n")
        
        # Compare CNN vs ViT
        cnn_models = {k: v for k, v in results.items() if "Net" in k}
        vit_models = {k: v for k, v in results.items() if "ViT" in k or "DeiT" in k}
        
        if cnn_models and vit_models:
            avg_cnn_acc = sum(r['test_metrics']['test_accuracy'] for r in cnn_models.values()) / len(cnn_models)
            avg_vit_acc = sum(r['test_metrics']['test_accuracy'] for r in vit_models.values()) / len(vit_models)
            
            f.write(f"- **Average CNN Accuracy**: {avg_cnn_acc:.2f}%\n")
            f.write(f"- **Average ViT Accuracy**: {avg_vit_acc:.2f}%\n")
            f.write(f"- **Performance Gap**: {abs(avg_vit_acc - avg_cnn_acc):.2f}% in favor of {'ViT' if avg_vit_acc > avg_cnn_acc else 'CNN'}\n")
    
    logger.info(f"Report generated: {report_path}")

def run_crop_disease_experiment(config_path: str, data_dir: str):
    """Run complete crop disease detection experiment"""
    
    logger.info("Starting Crop Disease Detection Experiment")
    logger.info("="*60)
    
    # Load configuration
    config = load_config(config_path)
    training_config = config['training']
    
    # Setup device
    device = setup_device()
    
    # Analyze PlantVillage dataset
    logger.info("Analyzing PlantVillage crop disease dataset...")
    dataset_info = analyze_plantvillage_dataset(data_dir)
    
    print(f"\nPlantVillage Dataset Analysis:")
    print(f"- Crop disease images: {dataset_info['total_crop_disease_images']:,}")
    print(f"- Disease classes: {dataset_info['num_disease_classes']}")
    print(f"- Crop types: {', '.join(dataset_info['crop_types'])}")
    
    # Create data loaders
    logger.info("Creating PlantVillage data loaders...")
    train_loader, val_loader, test_loader, num_classes, class_names = create_plantvillage_data_loaders(
        data_dir=data_dir,
        batch_size=training_config['batch_size'],
        num_workers=2
    )
    
    logger.info(f"Data loaders created:")
    logger.info(f"- Train batches: {len(train_loader)}")
    logger.info(f"- Val batches: {len(val_loader)}")
    logger.info(f"- Test batches: {len(test_loader)}")
    
    # Create models for crop disease detection
    logger.info("Creating crop disease detection models...")
    models = create_crop_disease_models(num_classes)
    
    # Run comparison experiment
    logger.info("Starting crop disease model comparison...")
    start_time = time.time()
    
    results = compare_crop_disease_models(
        models, train_loader, val_loader, test_loader, training_config, device
    )
    
    experiment_time = time.time() - start_time
    logger.info(f"Experiment completed in {experiment_time:.2f} seconds")
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    logger.info("Generating crop disease detection visualizations...")
    visualize_crop_disease_results(results, class_names, results_dir)
    
    # Generate report
    logger.info("Generating crop disease detection report...")
    generate_crop_disease_report(results, dataset_info, training_config, results_dir)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("CROP DISEASE DETECTION EXPERIMENT COMPLETED")
    print(f"{'='*60}")
    
    print(f"\nFinal Results:")
    for model_name, result in results.items():
        test_acc = result['test_metrics']['test_accuracy']
        print(f"  {model_name}: {test_acc:.2f}% test accuracy")
    
    best_model = max(results.keys(), key=lambda x: results[x]['test_metrics']['test_accuracy'])
    print(f"\nBest Model for Crop Disease Detection: {best_model}")
    print(f"Achievement: Successfully compared CNN vs ViT on agricultural data!")

def main():
    """Main function for crop disease detection experiments"""
    
    parser = argparse.ArgumentParser(description='Crop Disease Detection: CNN vs ViT Comparison')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data/',
                       help='Path to PlantVillage dataset directory')
    parser.add_argument('--model', type=str, choices=['resnet50', 'efficientnet_b0', 'vit_base_patch16_224', 'deit_small_patch16_224'],
                       help='Train single model (optional)')
    
    args = parser.parse_args()
    
    if args.model:
        # Train single model for crop disease detection
        logger.info(f"Training single model: {args.model}")
        # Implementation for single model training
    else:
        # Run full comparison experiment
        run_crop_disease_experiment(args.config, args.data_dir)

if __name__ == "__main__":
    main()
