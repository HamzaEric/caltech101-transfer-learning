# Caltech-101 Transfer Learning & Fine-Tuning Study

A comprehensive benchmark and interpretability study exploring transfer learning strategies on the **Caltech-101** dataset (102 classes) using **ResNet-18** in PyTorch.

This project evaluates three core paradigms:

1. **Untrained Baseline:** An ImageNet-pretrained backbone initialized with an untrained 102-class classification head.
2. **Feature Extraction:** A pretrained backbone fully frozen; only the custom linear classifier head (`fc`) is trained.
3. **Fine-Tuning:** Progressive unfreezing of deep residual blocks, comparing `layer4` fine-tuning against extended `layer3` + `layer4` + `fc` fine-tuning with lower learning rates.

In addition to quantitative metrics (Accuracy, Macro F1, Weighted F1), the project explores model interpretability through **mean feature map activations** and **Grad-CAM heatmaps**.

---

##  Repository Structure

```text
caltech101-transfer-learning/
│
├── Notebooks/
│   ├── Transfer_Learning.ipynb
│   ├── Feature_Extraction_Pretrained_CNN.ipynb
│   ├── Finetuning_Convolutional_layers.ipynb
│   └── Baseline_vs_Feature_Extracted_vs_Fine_Tuned.ipynb
│
├── Training_logs/
│   ├── training_logs_feature_extraction.json
│   └── training_logs_finetuned.json
│
│
└── README.md
```

##  Notebook Walkthrough

| **Notebook** | **Focus & Description** |
|---|---|
| **`Transfer_Learning.ipynb`** | Dataset ingestion, train/validation/test splits, image augmentations, and PyTorch `DataLoader` pipelines. |
| **`Feature_Extraction_Pretrained_CNN.ipynb`** | ResNet-18 feature extraction: freezes all backbone parameters, adapts the `fc` head to 102 classes, and trains the classifier. |
| **`Finetuning_Convolutional_layers.ipynb`** | Unfreezes upper residual blocks (`layer4` and extended `layer3` + `layer4`) for representation adaptation using Adam. |
| **`Baseline_vs_Feature_Extracted_vs_Fine_Tuned.ipynb`** | Comparative evaluation across all models, including training/validation dynamics, row-normalized confusion matrices, test accuracy, Macro F1, Weighted F1, layer activations, and Grad-CAM visualizations. |

## Methodology & Training Workflow

### 1. Data Preprocessing

- **Input Dimensions:** Images are resized and center-cropped to `224 × 224`.
- **Normalization:** ImageNet distribution parameters:
  - Mean: `[0.485, 0.456, 0.406]`
  - Standard deviation: `[0.229, 0.224, 0.225]`

### 2. Model Architectures & Parameter Freezing

- **Backbone:** ResNet-18 pretrained on `IMAGENET1K_V1`.
- **Classification Head:** Linear layer projecting `512 → 102` logits.
- **Unfreezing Schemes:**
  - **Feature Extraction:** Approximately 52.3K trainable parameters; only `fc` is trained.
  - **Baseline Fine-Tuning:** `layer4` + `fc`.
  - **Extended Fine-Tuning:** `layer3` + `layer4` + `fc` to allow higher-level feature adaptation.

### 3. Evaluation & Metrics

- **Classification Performance:** Multiclass Accuracy, Macro F1, and Weighted F1.
- **Error Analysis:** Large-canvas, row-normalized confusion matrices covering all 102 target categories.
- **Interpretability:**
  - Layer 4 channel-averaged activation maps.
  - Grad-CAM saliency heatmaps overlaid on original inputs to investigate model attention.

##  Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn
```

### Running the Notebooks

1. Clone the repository:

   ```bash
   git clone https://github.com/HamzaEric/caltech101-transfer-learning.git
   cd caltech101-transfer-learning
   ```

2. Open a notebook in Jupyter Notebook:

   ```bash
   jupyter notebook Notebooks/Baseline_vs_Feature_Extracted_vs_Fine_Tuned.ipynb
   ```

   Alternatively, open the notebooks in Google Colab.

##  Summary of Findings

- **Baseline (Untrained Head):** Expected to produce near-random predictions across 102 classes, serving as a sanity check for the evaluation pipeline.
- **Feature Extraction:** Provides an efficient baseline by training only the classifier while preserving the pretrained backbone representations.
- **Fine-Tuning (`layer3` + `layer4` + `fc`):** Enables deeper feature adaptation and may improve Macro F1 on visually challenging categories. Grad-CAM visualizations help investigate whether predictions rely on localized, class-discriminative image features.

**Note:** Actual performance differences should be verified against the recorded test metrics. Interpretability visualizations provide insight into model behavior but do not independently establish prediction correctness.
