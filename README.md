# Battery State of Health (SOH) & RUL Estimator

This repository contains a deep learning pipeline for estimating the State of Health (SOH) and Remaining Useful Life (RUL) of lithium-ion batteries. It leverages raw battery telemetry (Voltage, Current, Temperature, Time) across multiple charge/discharge cycles.

## 📂 Datasets

The project utilizes a hybrid dataset approach, stored locally in the `Dataset/` directory:
- **NASA Battery Dataset**: Found in `Dataset/Nasa dataset`, containing Type 1 (raw operational) and Type 2 (impedance) measurements.
- **CALCE Battery Dataset**: Found in `Dataset/CALCE dataset`, containing Train and Test splits across various discharge profiles.

## 🧠 Machine Learning Models

The main SOH/RUL predictor is a custom neural network comprising:
- **1D Convolutional Neural Network (CNN)** for local feature extraction.
- **Bidirectional LSTM** for capturing temporal dependencies across charge cycles.
- **Custom Attention Mechanism** to weight the importance of specific time steps.

Baseline models are also included for benchmarking:
- Linear Regression
- Deep Neural Networks (DNN)
- Gated Recurrent Units (GRU)
- Standard LSTM

## 🚀 Getting Started

*(Note: The codebase is currently transitioning from a monolithic script to a modular structure.)*

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- TensorFlow
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn

### Running the Analysis
Currently, the entire pipeline is located in `ajinkya_slape_19_01_26_1_updated.py`. To execute the pipeline (which includes data parsing, visualization, and model training):
```bash
python ajinkya_slape_19_01_26_1_updated.py
```
*(Warning: The script heavily utilizes `plt.show()`, which will render plots interactively. You will need to close the plot windows to allow the script to proceed to the next execution step.)*

## 🔮 Roadmap
1. **Modularization**: Break down the monolithic script into isolated modules (`data_loader`, `models`, `train`).
2. **Local Pathing**: Fully integrate the local `Dataset/` inputs instead of relying on Kaggle downloads.
3. **Automated MLOps**: Save trained model weights and evaluation metrics.
