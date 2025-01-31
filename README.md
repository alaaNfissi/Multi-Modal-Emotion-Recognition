<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/alaaNfissi/Multi-Modal-Emotion-Recognition">
    <img src="figures/logo.png" alt="Logo" width="80" height="80">
  </a>

  # Deep Learning Framework for Multi-modal Emotion Recognition in Mental Health Monitoring

  <p align="center">
    A cutting-edge deep learning framework integrating text, audio, and video for emotion recognition.
    <br />
    <strong>Paper submitted for publication.</strong>
    <br />
  </p>
</div>

<div align="center">

[![View - Documentation](https://img.shields.io/badge/view-Documentation-blue?style=for-the-badge)](https://github.com/alaaNfissi/Multi-Modal-Emotion-Recognition/#readme "Go to project documentation")

</div>  

<div align="center">
    <p align="center">
    ·
    <a href="https://github.com/alaaNfissi/Multi-Modal-Emotion-Recognition/issues">Report Bug</a>
    ·
    <a href="https://github.com/alaaNfissi/Multi-Modal-Emotion-Recognition/issues">Request Feature</a>
  </p>
</div>

## Overview

This repository provides an implementation of a **Multi-modal Emotion Recognition (MMER) system** as described in the research paper:

> *"Deep Learning Framework for Multi-modal Emotion Recognition in Mental Health Monitoring."*

### Key Features:
- **Audio Processing:** CNNs + bi-directional xLSTM (Bi-xLSTM) for raw waveform analysis.
- **Text Processing:** ALBERT for deep semantic representation.
- **Video Processing:** RepVGG-CNN for facial expression detection.
- **Fusion Mechanism:** Cross-attention to enhance emotion recognition across modalities.
- **Loss Function:** Focal Loss to mitigate class imbalance.

---

## Contents
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#abstract">Abstract</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#datasets">Datasets</a></li>
    <li><a href="#evaluation-metrics">Evaluation Metrics</a></li>
    <li><a href="#experimental-results">Experimental Results</a></li>
    <li><a href="#comparison-with-state-of-the-art">Comparison with State-of-the-Art</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#paper-reference">Paper Reference</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

---

## Abstract

Multi-modal Emotion Recognition (MMER) enhances human-computer interaction by enabling systems to accurately interpret and respond to human emotions. This research introduces an MMER system integrating **audio, text, and video** modalities for mental health monitoring and suicide prevention.

### Model Components:
- **Audio:** CNNs + Bi-xLSTM for raw waveform processing.
- **Text:** ALBERT for extracting deep semantic representations.
- **Video:** RepVGG-CNN for capturing fine-grained facial expressions.
- **Fusion:** Cross-attention mechanism for multi-modal feature integration.
- **Loss Function:** Focal Loss to handle class imbalance.

The system achieves **88.31% accuracy, 76.43% F1-score on IEMOCAP**, and **65.2% accuracy, 56.5% F1-score on CMU-MOSEI**, surpassing state-of-the-art approaches.

## Built With

### Tools and Libraries

- ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
- ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
- ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
- ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
- ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## Model Architecture

<p align="center">
  <img src="figures/model_archi1.jpg" width="800" alt="MMER Architecture" />
</p>

## Experimental Results

### IEMOCAP Dataset

| Accuracy | F1-Score |
|----------------|-----------------|
| **88.31%** | **76.43%** |

#### Confusion Matrix

<p align="center">
  <img src="figures/confusion matrix.png" width="600" alt="Confusion Matrix IEMOCAP" />
</p>

### CMU-MOSEI Dataset

| Accuracy | F1-Score |
|----------------|-----------------|
| **65.2%** | **56.5%** |

#### Confusion Matrix

<p align="center">
  <img src="figures/confusionmatrixmosei.png" width="600" alt="Confusion Matrix CMU-MOSEI" />
</p>

## Comparison with State-of-the-Art

| Model | IEMOCAP (F1-Score) | CMU-MOSEI (F1-Score) |
|----------------|-----------------|-----------------|
| M3ER | 82.4% | - |
| FV2ES | 81.06% | - |
| Babaali et al. | 76.8% | - |
| **Ours** | **76.43%** | **56.5%** |

## Installation

1. **Clone** this repo:
   ```bash
   git clone https://github.com/alaaNfissi/Multi-Modal-Emotion-Recognition.git
   cd Multi-Modal-Emotion-Recognition
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Configure dataset paths** in `main.py`.
2. **Train the model**:
   ```bash
   python main.py
   ```
3. **Evaluate on test set**:
   ```bash
   python main.py --test
   ```

## Paper Reference

If you use this code, please cite:

> "Deep Learning Framework for Multi-modal Emotion Recognition in Mental Health Monitoring."

## License

This project is licensed under the [BSD 3-Clause License](https://opensource.org/license/bsd-3-clause).
