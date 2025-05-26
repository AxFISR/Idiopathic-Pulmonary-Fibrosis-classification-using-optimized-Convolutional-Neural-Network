# 🧠 Idiopathic Pulmonary Fibrosis (IPF) Classification using Optimized CNN

This project uses a custom Convolutional Neural Network (CNN) for the **automated detection of Idiopathic Pulmonary Fibrosis (IPF)** from CT scan slices. It explores extensive **hyperparameter tuning**, **data augmentation**, and **transfer learning** to build a reliable diagnostic model.

---

## 📌 Table of Contents

* [Problem Statement](#problem-statement)
* [Proposed Solution](#proposed-solution)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Training Process](#training-process)
* [Hyperparameter Optimization](#hyperparameter-optimization)
* [Evaluation Metrics](#evaluation-metrics)
* [Transfer Learning](#transfer-learning)
* [GUI Diagnosis Tool](#gui-diagnosis-tool)
* [Results](#results)
* [Installation & Usage](#installation--usage)
* [Credits](#credits)
* [License](#license)

---

## 🚨 Problem Statement

Idiopathic Pulmonary Fibrosis (IPF) is a **progressive and irreversible lung disease** with a median survival of \~4 years. Early detection is crucial, yet diagnosis is time-consuming and reliant on expert radiologists. The goal is to **automate diagnosis** via deep learning.

---

## 💡 Proposed Solution

We implemented a **custom CNN** trained on grayscale CT images (256x256), with features including:

* Data Augmentation
* Dropout Regularization
* Hyperparameter tuning
* Evaluation using metrics: accuracy, precision, recall, F1-score
* Final deployment with GUI-based real-time CT scan diagnosis

We also explored **transfer learning** to further improve model performance.

---

## 🗃️ Dataset

* Total images: **4,391**

  * IPF Positive: 2,122
  * IPF Negative: 2,269
* Image size: 512x512 converted to 256x256 grayscale
* Split:

  * Train: 70% (\~3073 images)
  * Validation: 25% (\~1098 images)
  * Test: 5% (\~220 images)

---

## 🧱 Model Architecture

A custom CNN with the following structure:

* 4 Convolution + MaxPooling blocks
* Flatten layer
* Dense layer with dropout
* Final dense layer (Softmax for 2 classes)

> Activation: ReLU in hidden layers, Softmax in output

---

## 🎯 Training Process

* Normalization: Pixel values scaled to \[0, 1]
* Augmentations: flips, brightness, contrast jitter
* Loss: `sparse_categorical_crossentropy`
* Optimizer: Adam
* Batch sizes: 32 or 64
* Epochs: up to 150
* Google Colab Pro+ used for computation

---

## ⚙️ Hyperparameter Optimization

We tested **7 different configurations**. Variables included:

* Learning rates: 5e-4, 5e-5, 5e-6
* Dropout: 0.3–0.5
* Batch size: 32 or 64
* Epochs: 50–150
* Augmentation: Enabled/Disabled

| Run | Batch | Epochs | LR   | Dropout | Aug | Accuracy   | F1 Score   |
| --- | ----- | ------ | ---- | ------- | --- | ---------- | ---------- |
| 1   | 32    | 70     | 5e-5 | 0.3     | No  | 83.18%     | 0.8477     |
| 4   | 64    | 100    | 5e-6 | 0.5     | Yes | 90.91%     | 0.9099     |
| 7 ✅ | 64    | 150    | 5e-6 | 0.45    | Yes | **95.45%** | **0.9569** |

---

## 📏 Evaluation Metrics

Best configuration achieved:

* **Accuracy**: 95.45%
* **Precision**: 96.52%
* **Recall**: 94.87%
* **F1 Score**: 95.69%
* Misclassifications: 10/220 test images

---

## 🔁 Transfer Learning

We fine-tuned the trained model on a new dataset using frozen layers.

* Used `load_model()` and recompiled with a new learning rate
* Augmentation applied to new data
* Achieved improved metrics without retraining from scratch

---

## 🖥️ GUI Diagnosis Tool

Developed a real-time **IPF detection GUI in Google Colab** using `ipywidgets`:

* Upload folder of CT scan images
* Images are displayed with predicted labels and probability scores
* Final diagnosis (Positive or Negative) based on majority voting
* Flagged predictions highlighted in red if they disagree with final diagnosis

---

## 📊 Final Results Summary

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 95.45% |
| Precision | 96.52% |
| Recall    | 94.87% |
| F1-Score  | 95.69% |
| FP        | 4      |
| FN        | 6      |

✅ Excellent generalization and balance between sensitivity and specificity.

---

## 🚀 Installation & Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/ipf-cnn-classification.git
```

2. Run the Colab notebook:

* Open `ipf_classification.ipynb` in Google Colab
* Follow instructions to train, validate, or test
* For GUI: run `gui_diagnosis_tool.ipynb`

---

## 🙌 Credits

**Authors:** Alex Baboshin, Asaf Shnaider
**Supervisor:** Prof. Miri Weiss Cohen
**Institution:** Braude Academic College of Engineering

---

## 📄 License

This project is for academic research and demonstration purposes. For commercial or clinical use, further validation and regulatory approval are required.

---

## 📬 Contact

* Alex: [Alex.Baboshin@e.braude.ac.il](mailto:Alex.Baboshin@e.braude.ac.il)
* Asaf: [Asaf.Schneiderman@e.braude.ac.il](mailto:Asaf.Schneiderman@e.braude.ac.il)
