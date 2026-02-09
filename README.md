# Pediatric Pneumonia Detection from Chest X-rays

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## 🎯 Project Overview

This project implements a deep learning-based binary classifier to detect pneumonia in pediatric chest X-ray images. Using transfer learning with DenseNet121, the model aims to assist in early and accurate diagnosis of pneumonia in children, addressing a critical healthcare challenge in resource-limited settings.

**Key Achievement**: Binary classification of chest X-rays (Normal vs. Pneumonia) using state-of-the-art computer vision techniques.

## 🔬 Medical Context

Pneumonia is one of the leading causes of death in children under five years old globally. Early and accurate diagnosis is crucial for effective treatment. This project leverages machine learning to provide a screening tool that can assist healthcare professionals in making faster, data-driven decisions, particularly in areas with limited access to specialized radiologists.

## 🚀 Features

- **Transfer Learning**: Fine-tuned DenseNet121 model pre-trained on ImageNet
- **Dual Inference Support**: Compatible with both PyTorch (.pt) and ONNX (.onnx) model formats
- **Data Augmentation**: Robust preprocessing pipeline with normalization and resizing
- **Class Balancing**: Implements resampling to handle class imbalance
- **Production-Ready**: Modular architecture with separate train/inference pipelines
- **Experiment Tracking**: Integration with Weights & Biases for model monitoring
- **Comprehensive Metrics**: Tracks accuracy, precision, and recall for thorough evaluation

## 📊 Technical Approach

### Model Architecture
- **Base Model**: DenseNet121 (pre-trained on ImageNet1K)
- **Custom Classifier**: Single output neuron for binary classification
- **Framework**: PyTorch Lightning for scalable training
- **Optimization**: Adam optimizer with configurable learning rate and weight decay

### Dataset
- **Source**: [Pediatric Pneumonia Chest X-ray Dataset](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray/data)
- **Split Strategy**: 70% training, 30% validation from training set
- **Preprocessing**: 
  - Resize to 224×224 pixels
  - Convert grayscale to RGB (3 channels)
  - ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Evaluation Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: Minimize false positives (healthy classified as pneumonia)
- **Recall**: Minimize false negatives (pneumonia missed) - critical for medical applications

## 🛠️ Installation & Setup

### Prerequisites
- Python >= 3.9
- CUDA-compatible GPU (recommended for training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/argi00/pneumonia.git
cd pneumonia
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
- Download the [Pediatric Chest X-ray Pneumonia dataset](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray/data)
- Extract to `data/raw/PediatricChestX-rayPneumonia/`

## 📁 Project Structure

```
├── LICENSE
├── Makefile           <- Convenience commands for data processing and training
├── README.md          <- This file
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data transformations
│   ├── processed      <- Final canonical datasets for modeling
│   └── raw            <- Original, immutable data (chest X-rays)
│
├── docs               <- Documentation
├── models             <- Trained model checkpoints and predictions
├── notebooks          
│   └── data_prep.ipynb <- Data exploration and preprocessing notebook
│
├── pneumonia_cls      <- Source code package
│   ├── __init__.py
│   ├── config.py      <- Configuration and project paths
│   ├── dataset.py     <- Custom PyTorch datasets with preprocessing
│   ├── models.py      <- Model definitions (Classifier, Orchestrator)
│   ├── features.py    <- Feature engineering utilities
│   └── plots.py       <- Visualization functions
│
├── references         <- Data dictionaries and reference materials
├── reports            
│   └── figures        <- Generated visualizations and results
│
├── requirements.txt   <- Python dependencies
├── pyproject.toml     <- Package configuration
└── setup.cfg          <- Linting configuration
```

## 🎓 Usage

### Training a Model

```bash
# Using Makefile (if configured)
make train

# Or directly with Python Lightning
python -m pneumonia_cls.models
```

### Inference

```python
from pneumonia_cls.models import Classifier
from PIL import Image

# Load model
classifier = Classifier(
    model_path="models/best_model.pt",
    threshold=0.5,
    device='cuda'  # or 'cpu'
)

# Predict on new image
image = Image.open("path/to/chest_xray.jpeg")
prediction = classifier(image)
# Output: 0 = Normal, 1 = Pneumonia
```

## 📈 Results & Performance



Example:
- **Validation Accuracy**: 0.88%
- **Validation F1 score**: 0.91%
- **Precision**: 0.84%
- **Recall**: 0.99%

## 🔧 Key Technologies

| Category | Tools |
|----------|-------|
| **Deep Learning** | PyTorch, PyTorch Lightning, torchvision |
| **Data Processing** | pandas, NumPy, scikit-learn, PIL |
| **Visualization** | matplotlib, seaborn |
| **Experiment Tracking** | Weights & Biases (wandb) |
| **Model Deployment** | ONNX (for production inference) |
| **Development** | Jupyter, loguru, tqdm, typer |

## 🎯 Future Improvements

- [ ] Implement ONNX runtime for optimized inference
- [ ] Multi-class classification (bacterial vs. viral pneumonia)
- [ ] Grad-CAM visualization for model interpretability
- [ ] Deployment as REST API or web application
- [ ] Integration with DICOM medical imaging format
- [ ] Cross-validation for robust performance estimation

## 🤝 Contributing

This is an individual project developed for the **ILINA Junior Technical Fellowship** application. Feedback and suggestions are welcome!

## 📝 License

*(Add license information if applicable)*

## 👤 Author

**Yayi make**

- GitHub: [@argi00](https://github.com/argi00)
- Fellowship Application: [ILINA Junior Technical Fellowship](https://www.ilinaprogram.org/jrf)

## 🙏 Acknowledgments

- Dataset: [Andrew Marques - Kaggle](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray)
- Project Template: [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
- Pre-trained Model: DenseNet121 from torchvision

---

**Note**: This project is developed as part of an application to the ILINA Junior Technical Fellowship program, demonstrating practical machine learning skills applied to real-world healthcare challenges.
