# Plant Disease Detection with Transfer Learning

**[🚀 Try the Live Demo](https://aahgtydqtwfgsxkffwpgcf.streamlit.app/)**

Classifies leaf images from pepper, potato and tomato plants into 15 disease / healthy classes using ImageNet-pretrained CNNs (EfficientNetB0 and MobileNetV2), with a Streamlit app for uploading a leaf photo and viewing the prediction and confidence.

## Results (held-out test set, 2,949 images)

| Model | Test accuracy |
|---|---|
| EfficientNetB0 (fine-tuned) | 97.3% (macro F1 0.97) |
| MobileNetV2 (fine-tuned) | 96.4% |

Both models were trained for 20 epochs on the identical split. This is a single run per model: the 0.9-point gap is about 27 images, so treat it as a modest difference, not a settled ranking. Scores are on PlantVillage images and should not be read as field accuracy (see [Limitations](#limitations)).

## Dataset and split

[PlantVillage (Kaggle)](https://www.kaggle.com/datasets/arjuntejaswi/plant-village): 20,638 images, 15 classes.

| Plant | Classes |
|---|---|
| Pepper | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

**Split:** stratified 14,741 train / 2,948 validation / 2,949 test. The validation set is used for early stopping and checkpointing; the test set is used only for reported scores.

**De-duplication:** the split was made with `StratifiedGroupKFold`, grouping images by filename stem. That produced 20,638 groups for 20,638 images, i.e. the filenames contain no duplicate IDs, so the split is effectively a stratified random split. As a separate check, perceptual hashing (pHash, Hamming distance <= 4, also comparing mirrored, flipped and 180-degree-rotated test images) found a near-duplicate in the training set for 17 of 2,949 test images (0.6%). Accuracy was 100% on those 17 and 97.3% on the remaining 2,932, so near-duplicates do not noticeably inflate the headline score. The check does not catch 90-degree rotations, crops, or different photos of the same leaf.

## Models

Both use ImageNet-pretrained backbones with a small classification head (pooling, dropout, 15-way softmax). Training first fits the head with a frozen backbone, then fine-tunes the last 30 backbone layers at a low learning rate (1e-5) with BatchNormalization layers kept frozen. Exact settings are in the notebook.

## Per-class results: EfficientNetB0 on the test set

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Pepper: Bacterial Spot | 0.98 | 0.99 | 0.99 | 128 |
| Pepper: Healthy | 0.99 | 1.00 | 1.00 | 193 |
| Potato: Early Blight | 0.99 | 1.00 | 0.99 | 141 |
| Potato: Late Blight | 0.99 | 0.96 | 0.98 | 126 |
| Potato: Healthy | 0.92 | 0.96 | 0.94 | 25 |
| Tomato: Bacterial Spot | 0.99 | 0.98 | 0.99 | 296 |
| Tomato: Early Blight | 0.97 | 0.88 | 0.92 | 148 |
| Tomato: Late Blight | 0.97 | 0.98 | 0.98 | 301 |
| Tomato: Leaf Mold | 0.99 | 0.97 | 0.98 | 148 |
| Tomato: Septoria Leaf Spot | 0.96 | 0.99 | 0.97 | 248 |
| Tomato: Spider Mites | 0.98 | 0.93 | 0.95 | 254 |
| Tomato: Target Spot | 0.86 | 0.95 | 0.90 | 191 |
| Tomato: Yellow Leaf Curl Virus | 1.00 | 0.99 | 1.00 | 470 |
| Tomato: Mosaic Virus | 0.98 | 0.98 | 0.98 | 48 |
| Tomato: Healthy | 0.97 | 0.99 | 0.98 | 232 |
| **Accuracy / macro avg** | 0.97 | 0.97 | 0.97 | 2,949 |

The weakest classes are Tomato Target Spot (F1 0.90) and Tomato Early Blight (F1 0.92, recall 0.88). Potato Healthy has only 25 test images, so its scores are noisy.

## Model interpretation (Grad-CAM)

![Grad-CAM on test images](https://img.sanishtech.com/u/c97c34ebdfda094f0deaf69067deb1e9.png
)

On a sample of correctly and incorrectly classified test images, the EfficientNetB0 heatmaps concentrate on leaf tissue. This is a qualitative check on a small sample, not a quantitative analysis.

## Deployment: size, accuracy and latency (TFLite)

Measured on a Colab CPU, batch size 1, model inference only (no image preprocessing). Accuracy is on the same 2,949-image test set.

| Model | Format | Size | Test acc | Latency, mean / p95 |
|---|---|---|---|---|
| EfficientNetB0 | float32 | 16.7 MB | 97.32% | 19.0 / 19.8 ms |
| EfficientNetB0 | float16 | 8.41 MB | 97.32% | 19.2 / 20.7 ms |
| EfficientNetB0 | dynamic-range int8 | 4.72 MB | 97.12% | 35.0 / 36.4 ms |
| MobileNetV2 | float32 | 8.94 MB | 96.41% | 8.0 / 8.5 ms |
| MobileNetV2 | float16 | 4.50 MB | 96.30% | 7.9 / 8.7 ms |
| MobileNetV2 | dynamic-range int8 | 2.53 MB | 96.20% | 22.0 / 23.1 ms |

- EfficientNetB0 float16 halves the size with no accuracy change.
- MobileNetV2 is about 2.4x faster and 1.9x smaller than EfficientNetB0 (float32) for about 0.9 points lower accuracy.
- Dynamic-range quantization gives the smallest files but was slower on this x86 CPU; results on ARM devices may differ.

## Run locally

```bash
git clone https://github.com/priyansh-commits/Plant_village_project.git
cd Plant_village_project
pip install -r requirements.txt
streamlit run app.py
```

Preprocessing must match training: resize to 224x224 and pass raw 0-255 pixel values (do not apply MobileNetV2's `preprocess_input` to the EfficientNetB0 model). Class order is the alphabetical folder order stored in `class_names.json`.

## Limitations

- **Lab-style data.** PlantVillage images are single leaves on plain backgrounds. An informal check on a separate dataset of field photos (multiple leaves, cluttered scenes), run outside this repository, gave roughly 21% accuracy for both models. It is not a formal evaluation, but it shows that the scores above do not transfer to field photos.
- **Train/test overlap.** Only 17 of 2,949 test images have a detectable near-duplicate in the training set (see Dataset and split). Photos of the same leaf taken at different angles cannot be detected this way, so some residual overlap is possible.
- Only the 15 trained classes are supported; other plants, diseases or poor-quality images will still receive a prediction from one of them, often with high confidence.
- Educational tool, not a substitute for expert diagnosis.

## Author

Priyansh Khetarpal · Electronics and Communication Engineering, Delhi Technological University · [GitHub](https://github.com/priyansh-commits)
