# Emotion Control in MusicGen via Activation Steering

**Authors:** Lucia Fornetti & Tommaso Federici

This repository contains the official code for the project **"Emotion control in MusicGen via Activation Steering"**.
The project implements a lightweight framework to control high-level emotional attributes (e.g., "Happy" vs "Sad") in **MusicGen** using **Activation Steering**, eliminating the need for expensive fine-tuning.

## 🎵 Key Features

* **Zero Fine-Tuning:** Emotional control is performed entirely at inference time by injecting steering vectors into the model's internal activations.
* **Multi-Block Strategy:** We employ an innovative approach that injects distinct vectors into specific Transformer blocks to separately control rhythmic features (Mid-Block, Layer 12) and timbral features (Deep-Block, Layer 30).
* **Alpha Decay:** Implementation of a steering coefficient decay mechanism ($\gamma = 0.998$) to maintain audio structural coherence and prevent artifacts.

## 🧠 Methodology in Brief

The system intervenes directly on the residual stream of the Transformer during autoregressive generation. Through *Silhouette Score* analysis, we identified two optimal intervention points:

1.  **Mid-Block (Layers 11-14):** Controls low-level features such as tempo and brightness.
2.  **Deep-Block (Layers 27-29):** Controls timbral and textural features.

The steering intensity decays over time according to the formula $\alpha(t) = \alpha_0 \cdot \gamma^t$ to ensure natural transitions and prevent signal saturation.

## 🎧 Audio Samples

To evaluate the quality of the steering, you can listen to the generated audio samples:

* **Repository:** Examples used for the *blind listening test* (Original/Happy/Sad triplets) are available locally in the folder [`server_blind_evaluation/blind_evaluation_test_audio/`](server_blind_evaluation/blind_evaluation_test_audio/).
* **Google Drive:** Or at the following link:
    👉 **[LISTEN TO SAMPLES HERE (Google Drive)](https://drive.google.com/drive/folders/1CyR_g8qNG2x98fzPSFNg0Jg23TOs1SDC?usp=sharing)**


## 📂 Repository Structure

* `Emotion_Control_ActivationSteering_demo_code_colab.ipynb`: **Main Notebook**. It contains all the code required to load the model, extract (or load) vectors, and generate steered music. Ready to use on Google Colab.
* `core_colab_melody.py`: Contains the core logic of the project, including the custom classes for hooking the MusicGen model and implementing the steering mechanism.
* `data/`:
    * `Happy_Sad/`: Datasets of prompts used for vector extraction (`extraction.csv`) and testing (`inference.csv`).
    * `vectors/`: Contains pre-computed tensors (`steering_vectors.pt`) for the Happy/Sad directions, allowing for immediate inference without re-running the extraction phase.
* `server_blind_evaluation/`: Python/HTML code used to conduct the *blind listening test* described in the report for human validation.

## 🚀 Quick Start

The easiest way to test the model is using the provided notebook:

1.  Open `Emotion_Control_ActivationSteering_demo_code_colab.ipynb` (Google Colab with a T4 GPU is recommended).
2.  Install the required dependencies (run the first cell).
3.  Load the pre-computed vectors from `data/vectors/steering_vectors.pt` or run the extraction phase on your own prompts.
4.  Run the generation by modifying the `steering_strength` parameter (positive values for "Happy", negative values for "Sad").