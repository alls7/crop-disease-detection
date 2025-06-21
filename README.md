# Crop Disease Detection: ViT vs CNN Comparison

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)

## Overview

This repository contains the implementation for comparing Vision Transformers (ViTs) and Convolutional Neural Networks (CNNs) for crop disease detection using the PlantVillage dataset.

**Author**: Ali SU  
**Institution**: Hacettepe University  
**Course**: CMP 719A Computer Vision  

## Project Description

This project conducts a comprehensive comparison between Vision Transformers and Convolutional Neural Networks for crop disease detection, focusing on:

- 🎯 Classification accuracy
- ⚡ Computational efficiency
- 🛡️ Model robustness
- 🚀 Deployment considerations

## Repository Structure

```
crop-disease-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── config/
│   ├── .gitkeep
│   └── config.yaml
├── data/
│   ├── .gitkeep
│   └── download_data.py
├── models/
│   ├── __init__.py
│   ├── cnn_models.py
│   └── vit_models.py
├── training/
│   ├── __init__.py
│   └── trainer.py
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py
├── utils/
│   ├── __init__.py
│   └── helpers.py
├── experiments/
│   ├── run_experiments.py
│   └── compare_models.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_comparison.ipynb
│   └── results_analysis.ipynb
└── results/
    ├── models/
    ├── plots/
    └── logs/
```

## Models to be Compared

### CNN Models
- **ResNet-50**: Deep residual network with skip connections
- **EfficientNet-B0**: Compound-scaled CNN optimizing depth, width, and resolution

### Vision Transformer Models
- **ViT-B/16**: Base Vision Transformer with 16×16 patch size
- **DeiT-Small**: Data-efficient transformer with knowledge distillation

## Dataset

**PlantVillage Dataset**
- 54,303 images across 38 classes
- Various plant species and associated diseases
- Publicly available for research purposes

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alls7/crop-disease-detection.git
cd crop-disease-detection
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Download Dataset
```bash
python data/download_data.py
```

### Train CNN Models
```bash
python experiments/run_experiments.py --model resnet50
python experiments/run_experiments.py --model efficientnet_b0
```

### Train ViT Models
```bash
python experiments/run_experiments.py --model vit_base_patch16_224
python experiments/run_experiments.py --model deit_small_patch16_224
```

### Compare Results
```bash
python experiments/compare_models.py
```

## Results Summary

| Model | Accuracy (%) | Parameters (M) | Inference Time (ms) | Model Size (MB) |
|-------|-------------|----------------|-------------------|-----------------|
| ResNet-50 | TBD | 25.6 | TBD | 97.8 |
| EfficientNet-B0 | TBD | 5.3 | TBD | 20.1 |
| ViT-B/16 | TBD | 86.6 | TBD | 330.3 |
| DeiT-Small | TBD | 22.1 | TBD | 84.2 |

*Results will be updated after completing experiments*

## Evaluation Metrics

- Classification Accuracy
- Precision, Recall, F1-Score (per class and macro-averaged)
- Confusion Matrix
- Inference Time
- Model Parameters and Size
- Training Time

## Project Status

🚧 **Project in Development** 🚧

- [x] Repository structure created
- [x] Basic file templates
- [ ] Model implementations
- [ ] Data preprocessing pipeline
- [ ] Training scripts
- [ ] Evaluation utilities
- [ ] Experiment execution
- [ ] Results analysis
- [ ] Final report

## Configuration

Modify `config/config.yaml` to adjust:
- Data preprocessing parameters
- Training hyperparameters
- Model selection
- Evaluation settings

## Contributing

This is an academic project for CMP 719A Computer Vision course at Hacettepe University.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@article{su2025comparative,
  title={Comparative Analysis of Vision Transformers and Convolutional Neural Networks for Crop Disease Detection},
  author={Su, Ali},
  journal={ICLR},
  year={2025}
}
```

## Contact

Ali SU - alisu@hacettepe.edu.tr

Project Link: https://github.com/alls7/crop-disease-detection

---

*Last updated: June 2025*
