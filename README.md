# SignLink-ISL

SignLink-ISL is a deep learning-based system designed to recognize and interpret Indian Sign Language (ISL) gestures, facilitating effective communication between the deaf community and others. This project aims to build a robust Sign Language Recognition (SLR) model, with a specific focus on Indian Sign Language (ISL). The goal is to preprocess video datasets efficiently and train machine learning models to recognize and classify ISL gestures accurately.

## Table of Contents

- [Introduction](#introduction)
- [Approach](#approach)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset & References](#dataset--references)

## Introduction

Effective communication with individuals who are deaf or hard of hearing is increasingly important. To address this need, we developed SignLink-ISL. While the system is currently limited to recognizing and translating isolated sign language words due to dataset constraints, it lays the groundwork for future advancements. We hope this project serves as a starting point for others to build upon, eventually leading to a real-time translation system capable of converting entire sign language sentences into spoken language.

## Approach

### Dataset Preparation and Processing

We built a custom dataset loader to convert videos into frames, making them easier to load into memory for training or inference. To enhance our model's learning capabilities, we integrated Mediapipe into the dataset loader to extract hand landmarks. However, this integration significantly slowed down the dataset loading process. To address this, we pickled the processed dataset and applied augmentations to the pickled data for training. While this approach worked, it introduced some limitations that could be improved upon, such as exploring alternative landmark extraction models to speed up dataset processing.

Mediapipe also faced challenges in detecting hands consistently, which negatively impacted model performance. To counter this, we implemented interpolation to handle missing landmark values.

#### INCLUDE SUBSET 50 Creation

For the INCLUDE SUBSET 50 dataset, we randomly selected words from the larger INCLUDE dataset that had at least 20-21 videos per word. This ensured sufficient samples for training and validation, making the subset manageable while maintaining variability.

#### Dataset Loader Versions

We created two versions of the dataset loader:

 1. **Landmark-based Tensor Loader**: This version returns a tensor of hand landmarks, suitable for training transformer models. Using this, we achieved an accuracy of 54.55% on the INCLUDE SUBSET 50 (details in the repo). Tuning hyperparameters significantly improved performance, but one challenge with transformers is diagnosing performance bottlenecks. For example, adding interpolation to handle missing Mediapipe detections improved results, highlighting the importance of addressing dataset quality.
 2. **Video Tensor Loader**: This version returns tensors of video frames, making it suitable for CV-based models. This approach significantly improved performance, as discussed in the Model Performance section.

#### Data Augmentation

Due to the large dataset size and slow speed of Mediapipe, we pickled the processed dataset for efficient use. We then applied the following augmentations:

- **INCLUDE SUBSET 50** (12 frames per video):
  - Base dataset with a 30% probability of horizontal flip (used for all augmentations).
  - Additional datasets with:
    - Random crop.
    - Random rotation (-7.5°, 7.5°).
    - Combination of random crop and rotation.
- **Complete INCLUDE Dataset** (24 frames per video):
  - Base dataset with no augmentation.
  - Additional datasets with:
    - Horizontal flip (100% probability).
    - Random crop.
    - Random rotation (-5°, 5°).
    - Combination of horizontal flip (50%) and random rotation (-5°, 5°).

We used these augmented datasets combined for final training models on both INCLUDE SUBSET 50 and the complete INCLUDE dataset.

### Model Exploration

We experimented with several models, and r3d_18 consistently performed well across all datasets. While we also explored combinations like ResNet + GRU/LSTM, we observed their potential as lightweight models but could not explore them further due to hardware limitations.

Transformers were also used to explore their capabilities. While they showed promising results (54.55% accuracy on INCLUDE SUBSET 50), we realized that hyperparameter tuning plays a significant role in their performance. However, unlike traditional computer vision tasks where images can be directly visualized to diagnose issues, the inability to easily comprehend the dataset makes it challenging to identify data-related problems.

### Challenges Faced

1. **Mediapipe Speed**: Mediapipe significantly slowed down the dataset processing. We mitigated this by pickling the processed dataset to avoid repeated computations.
2. **Hardware Limitations**: We addressed this by using Kaggle or our own devices for testing and rented hardware from Vast AI for intensive training tasks.
3. **Dataset Loading**: Loading video datasets for training was challenging. We tackled this by creating our custom dataset loader with the help of FFmpeg.

### Future Improvements

1. Landmark Extraction: Exploring alternatives to Mediapipe or improving its implementation in the dataset loader could enable real-time translation.
2. Lightweight Models: ResNet + GRU/LSTM is a promising lightweight model combination that warrants further exploration.
3. Enhanced Dataset: Using a more comprehensive dataset with both word-level and sentence-level videos could enable proper translation of sign language sentences into spoken language.
4. Hyperparameter Tuning: Further tuning transformer hyperparameters could unlock better performance.

## Model Performance

We evaluated our models on both the INCLUDE SUBSET 50 and the complete INCLUDE dataset. Below are the detailed performance metrics:

### INCLUDE SUBSET 50

1. With Horizontal Flip (30% Probability) only
   - r3d_18: Accuracy: 89.18%
   - ResNet18 + GRU: Accuracy: 88.74%
2. With All Augmentations
   - r3d_18: Accuracy: 79.44%

### Complete INCLUDE Dataset

We trained the **r3d_18** model and achieved the following metrics:

**Accuracy: 89.51%**

Micro Metrics:
 - F1 Score: 0.8951
 - Precision: 0.8951
 - Recall: 0.8951

Macro Metrics:
 - F1 Score: 0.8895
 - Precision: 0.9069
 - Recall: 0.8934

Weighted Metrics:
 - F1 Score: 0.8911
 - Precision: 0.9071
 - Recall: 0.8951

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Naneet/SignLink-ISL.git
   ```

2. Navigate to the project directory
   ```
   cd SignLink-ISL
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To understand the flow of the project and replicate the results, follow the notebooks in the recommended order:

1. **Video Dataset Conversion**
   - Notebook: `Video_dataset_conversion_refined.ipynb`
   - This notebook demonstrates the process of converting video datasets into frame-based tensors
2. **Dataset Augmentation**
   - Notebook: `Dataset_augmentation_pickled.ipynb`
   - Apply various augmentations (e.g., horizontal flip, random crop, random rotation) to pickled dataset.
3. **Training the Models**
   - Notebook: `Complete_INCLUDE_training_notebook.ipynb` / `Subset_50_training-notebook.ipynb`
   - Loads the pickled dataset.
   - Covers the training process using different models like r3d_18 and ResNet18 + GRU/LSTM.
   - Training configurations (e.g., hyperparameters, loss functions, optimizers).
   - Model evaluation metrics and saving trained models.
  
🚨🚨 We will also be creating docs for it soon for better understanding 🚨🚨 :)

## Dataset & References

[INCLUDE Dataset](https://zenodo.org/records/4010759)

[INCLUDE SUBSET 50](/dataset/ISL_SUBSET_50_WORDS.txt)

[INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition](https://dl.acm.org/doi/10.1145/3394171.3413528)

[Indian Sign Language Character Recognition](https://cse.iitk.ac.in/users/cs365/2015/_submissions/vinsam/report.pdf)

[Deep Visual-Semantic Alignments for Generating Image Descriptions](https://arxiv.org/pdf/1412.2306)

[Sign language Recognition Using Machine Learning Algorithm](https://www.irjet.net/archives/V7/i3/IRJET-V7I3418.pdf)

[PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

[Our version of dataset](/dataset/README.md)










