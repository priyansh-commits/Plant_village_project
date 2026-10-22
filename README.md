# 🌿 Plant Disease Detection Using Deep Learning

**[🚀 Try the Live Demo](https://aahgtydqtwfgsxkffwpgcf.streamlit.app/)**

A deep learning project that classifies plant leaf images into 15 disease and healthy-leaf categories using EfficientNetB0, MobileNetV2, transfer learning, and fine-tuning.

## 📌 Project Overview

Plant diseases can affect crop health and agricultural productivity. This project explores image classification using pretrained convolutional neural networks (CNNs) to identify diseases from leaf images.

The model accepts a leaf image and predicts one of 15 classes covering pepper, potato, and tomato plants.

**Key highlights**
- Transfer learning with ImageNet-pretrained CNNs.
- Comparison of EfficientNetB0 and MobileNetV2.
- Fine-tuning pretrained layers to improve disease classification.
- Per-class precision, recall, and F1-score evaluation.
- Streamlit interface for uploading images and displaying predictions.

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- EfficientNetB0
- MobileNetV2
- NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit

## 🌱 Dataset

The project uses the PlantVillage-style dataset containing 15 classes:

| Plant | Classes |
|---|---|
| Pepper | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

The evaluated classification reports contain approximately 3,096 images.

*Source-https://www.kaggle.com/datasets/arjuntejaswi/plant-village*

## 🏗️ Model Architecture

### 1. EfficientNetB0 — Transfer Learning

The initial EfficientNetB0 model uses ImageNet-pretrained weights with the original classification head removed.

The custom classification head consists of:

- Global Average Pooling
- Batch Normalization
- Dense layer with 128 ReLU units
- Dropout with a rate of 0.3
- Output Dense layer with 15 softmax units

Initially, the pretrained backbone was frozen while the custom classification head was trained.

### 2. MobileNetV2

MobileNetV2 was also evaluated as an alternative pretrained CNN. It provides a useful comparison when considering model performance and potential deployment efficiency.

### 3. EfficientNetB0 Fine-Tuning

After evaluating the initial model, the final 30 layers of the pretrained backbone were unfrozen while earlier layers and Batch Normalization layers remained frozen.

Fine-tuning used a lower learning rate of `1e-5` to adapt the pretrained features to plant disease classification.

Early stopping and learning-rate reduction were used as training controls.

## 📊 Experimental Results

The following results summarize the separate experiments conducted during development.

| Experiment | Accuracy | Macro F1-score |
|---|---:|---:|
| Initial model experiment | 93% | 0.92 |
| MobileNetV2 baseline | 94% | 0.93 |
| EfficientNetB0 after fine-tuning | Approximately 98% | 0.98 |

The first two rows represent the earlier classification reports; the final row represents the later fine-tuned EfficientNetB0 evaluation.

The figures are rounded. Ensure that the same independent test split and class mapping were used before making a direct model comparison.

### Final Fine-Tuned EfficientNetB0: Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Pepper — Bacterial Spot | 0.99 | 1.00 | 0.99 | 149 |
| Pepper — Healthy | 1.00 | 1.00 | 1.00 | 221 |
| Potato — Early Blight | 1.00 | 0.99 | 1.00 | 150 |
| Potato — Late Blight | 1.00 | 0.99 | 0.99 | 150 |
| Potato — Healthy | 0.96 | 1.00 | 0.98 | 23 |
| Tomato — Bacterial Spot | 0.99 | 0.99 | 0.99 | 319 |
| Tomato — Early Blight | 0.96 | 0.93 | 0.94 | 150 |
| Tomato — Late Blight | 0.97 | 0.98 | 0.98 | 287 |
| Tomato — Leaf Mold | 0.99 | 0.95 | 0.97 | 143 |
| Tomato — Septoria Leaf Spot | 0.97 | 0.96 | 0.97 | 266 |
| Tomato — Spider Mites | 0.96 | 0.98 | 0.97 | 252 |
| Tomato — Target Spot | 0.93 | 0.95 | 0.94 | 210 |
| Tomato — Yellow Leaf Curl Virus | 0.99 | 1.00 | 0.99 | 482 |
| Tomato — Mosaic Virus | 0.98 | 1.00 | 0.99 | 56 |
| Tomato — Healthy | 1.00 | 0.98 | 0.99 | 238 |
| **Overall accuracy** | | | **Approximately 0.98** | **3,096** |
| **Macro average** | **0.98** | **0.98** | **0.98** | **3,096** |
| **Weighted average** | **0.98** | **0.98** | **0.98** | **3,096** |

*Classification metrics are rounded to two decimal places, so individual rows may not reproduce the aggregate metrics exactly.*

## 🔍 Challenges and Improvements

### Problem 1: Difficulty distinguishing similar diseases

The initial model struggled to distinguish visually similar tomato diseases.

In the earlier classification report:
- Tomato Early Blight recall was 0.58.
- Tomato Target Spot precision was 0.77.
- Tomato Septoria Leaf Spot recall was 0.86.

This indicated that some diseased leaves were being assigned to visually similar classes.

### Solution: Fine-tuning

The pretrained EfficientNetB0 backbone was partially unfrozen and trained with a much smaller learning rate.

In the later evaluation:
- Tomato Early Blight recall increased from 0.58 to 0.93.
- Tomato Target Spot recall increased from 0.88 to 0.95.
- Tomato Early Blight F1-score increased from 0.71 to 0.94.

These are results from the reported experiments; a controlled comparison requires identical test samples.

### Problem 2: Model selection

MobileNetV2 and EfficientNetB0 were evaluated to understand the trade-off between classification performance and deployment considerations.

The final choice should also consider model size, inference latency, memory requirements, and performance on real-world photographs.

### Problem 3: Generalization to real-world images

High accuracy on a curated dataset does not guarantee equivalent performance on photographs taken under different lighting, backgrounds, camera angles, or disease severity.

Testing on independently collected leaf images remains an important next step.

## 🚀 Streamlit Application

The trained model can be integrated into a Streamlit application that allows users to upload a leaf image and view the predicted class and confidence score.

### Run Locally

1. Clone the repository:

   ```bash
   git clone <YOUR_GITHUB_REPOSITORY_URL>
   cd <YOUR_PROJECT_FOLDER>
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the application:

   ```bash
   streamlit run app.py
   ```

4. Open the local URL displayed in the terminal and upload a leaf image.

### Expected Project Structure

```text
PlantDiseaseDetection/
├── app.py
├── plant_disease_finetuned.keras
├── class_names.json
├── requirements.txt
└── README.md
```

The saved model, class-name mapping, input dimensions, and image preprocessing must match the training configuration.

## ⚠️ Limitations

- The classifier supports only the 15 trained categories.
- Predictions may be unreliable for unfamiliar plants, diseases, or poor-quality images.
- Dataset accuracy does not establish field-level diagnostic accuracy.
- The model should be treated as an educational decision-support tool, not a replacement for expert diagnosis.
- A separate, representative real-world test set is needed to establish practical reliability.

## 🔮 Future Improvements

- Evaluate performance on real-world leaf photographs.
- Optimize inference latency and model size.
- Add better handling of unsupported or low-confidence images.
- Explore model quantization or lightweight deployment options.


## 👨‍💻 Author

**Priyansh Khetarpal**

Electronics and Communication Engineering, Delhi Technological University (DTU)

GitHub: [priyansh-commits](https://github.com/priyansh-commits)

---

*This project was developed to explore transfer learning, CNN fine-tuning, multiclass image classification, model evaluation, and deployment using Streamlit.*
