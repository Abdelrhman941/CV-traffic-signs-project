# **Traffic Sign Recognition System**

> ![image](GUI/image.png)

Computer vision pipeline for classifying German traffic signs using PyTorch. The project is structured for clarity and reproducibility with distinct modules for data, preprocessing, model architecture, training, and evaluation.

## [**Dataset Link**](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

## Repository Layout

```
traffic-sign-recognition/
├── Data/                    # Provided dataset (kept with original casing)
│
├── GUI/                     # Graphical User Interface
│
├── logs/                    # Training and evaluation logs
│
├── models/                  # Saved checkpoints
│
├── notebooks/               # Jupyter notebooks for experimentation
│
├── outputs/                 # Generated reports, plots, or exports
│
├── src/                     # Source package
│   ├── __init__.py
│   ├── config.py            # Global paths and hyperparameters
│   ├── data.py              # Dataset loading and DataLoader helpers
│   ├── evaluate.py          # Validation loops and metrics
│   ├── features.py          # Optional feature extraction utilities
│   ├── model.py             # CNN architecture factory
│   ├── preprocessing.py     # Image preprocessing pipeline
│   ├── train.py             # Training routine
│   └── utils.py             # Enums, seed helpers, class mappings
│
├── Data.rar                 # Raw archive of the dataset (optional)
│
├── main.py                  # CLI for training and evaluation workflows
│
├── predict.py               # CLI for single/batch image inference
│
├── requirements.txt         # Python dependencies
│
└── README.md
```

## **Installation**

> Tested with Python 3.10+ on Windows. Adjust commands if you use a different shell.

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/Scripts/activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. (Optional) If you plan to use GPU acceleration, install the CUDA-enabled variants of PyTorch from https://pytorch.org/get-started/locally/.

## **Usage**

Before running any script, ensure the dataset remains in `Data/` with the original folder structure (`Train/`, `Test/`, `Meta/`, CSV files).

### Training

```bash
python main.py train --epochs 30 --batch-size 32
```

This command will:
- Set seeds for reproducibility.
- Build training/validation DataLoaders from `Data/Train`.
- Train the CNN with an Adam optimizer and StepLR scheduler.
- Evaluate on the validation split and optionally show a confusion matrix.
- Save a checkpoint to `models/traffic_sign_model.pth` and log history under `logs/`.

Override defaults using the CLI flags (`--learning-rate`, `--weight-decay`, `--model-name`, `--skip-confusion`).

### Evaluation

```bash
python main.py evaluate --checkpoint models/traffic_sign_model.pth --show-confusion
```

Loads an existing checkpoint, rebuilds the model, and reports metrics on the validation split.

### Prediction

Single image:

```bash
python predict.py --image path/to/image.png --checkpoint models/traffic_sign_model.pth
```

Batch directory:

```bash
python predict.py --directory path/to/folder --checkpoint models/traffic_sign_model.pth
```

Use `--grayscale` to enable grayscale preprocessing prior to normalization.

## **Notebooks**

Exploratory work lives in `notebooks/`. The original notebook was preserved and can be updated to consume the new modules (e.g., `from src.data import build_dataloaders`). Keeping notebooks separate from production code ensures cleaner version control and easier collaboration.

## **Extending the Project**

- Implement advanced architectures (e.g., ResNet) inside `src/model.py`.
- Swap preprocessing strategies or augmentations in `src/preprocessing.py`.
- Log metrics to TensorBoard or Weights & Biases by expanding `main.py`.
- Export ONNX or TorchScript artifacts to `outputs/` for deployment.

## **GUI Workbench**

The `GUI/` directory hosts a polished Streamlit application with two pages:

- **Traffic Sign Studio** – welcome and onboarding view.
- **Workbench** – tabbed workspace for Preprocessing, Segmentation, Feature Extraction, and Classification.

Launch the UI from the project root:

```bash
streamlit run GUI/app.py
```

Features include animated tab transitions, reusable components, and direct wiring to the modules in `src/` for real preprocessing, chain-code extraction, and CNN inference. Place a trained checkpoint at `models/traffic_sign_model.pth` (produced via `main.py train`) to enable the classification tab.
