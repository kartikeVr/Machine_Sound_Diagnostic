# Machine Sound Diagnostic

A deep learning project for detecting anomalies in machine sounds using autoencoders. The system is trained on normal machine sounds and can detect faults or anomalies by analyzing reconstruction errors.

## Overview

This project uses a **Convolutional Autoencoder** to learn the patterns of normal machine sounds. When an unfamiliar or anomalous sound is presented, the reconstruction error is high, indicating a potential fault.

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | AutoEncoder model definition and training script |
| `esc50_loader.py` | Custom PyTorch Dataset for ESC-50 spectrogram loading |
| `generate_spectrograms.py` | Converts audio WAV files to spectrogram images |
| `use_model.py` | Inference script for anomaly detection |
| `best_autoencoder.pth` | Pre-trained model weights |

## How It Works

### 1. Data Preparation
- Audio files are converted to Mel-spectrograms using `generate_spectrograms.py`
- Spectrograms are normalized and resized to 5 seconds (16kHz sample rate, 64 mel bands)
- Data is split: Folds 1-4 for training, Fold 5 for testing

### 2. Training
- The AutoEncoder is trained ONLY on "normal" machine sounds (e.g., engine)
- The encoder compresses the spectrogram into a latent representation (64 channels, 8×39)
- The decoder reconstructs the original spectrogram from the latent representation
- Loss function: Mean Squared Error (MSE)

### 3. Anomaly Detection
- During inference, the model tries to reconstruct the input spectrogram
- **Normal sounds**: Low reconstruction error (high accuracy)
- **Anomalous sounds**: High reconstruction error (low accuracy)
- A threshold value is used to classify sounds as normal or anomalous

## Model Architecture

```
Encoder:
  Conv2d(1→16) → ReLU → MaxPool2d
  Conv2d(16→32) → ReLU → MaxPool2d
  Conv2d(32→64) → ReLU → MaxPool2d  → Latent Representation

Decoder:
  ConvTranspose2d(64→32) → ReLU
  ConvTranspose2d(32→16) → ReLU
  ConvTranspose2d(16→1) → Sigmoid
```

## Requirements

- Python 3.x
- PyTorch
- torchaudio
- pandas
- matplotlib
- tqdm
- numpy

## Usage

### Step 1: Generate Spectrograms
```
bash
python generate_spectrograms.py
```

### Step 2: Train the Model
```
bash
python main.py
```

### Step 3: Run Inference
```
bash
# Single file prediction
python use_model.py --file path/to/audio.wav

# Batch evaluation with visualization
python use_model.py
```

## Example Results

The model calculates accuracy as: `(1 - √loss) * 100%`

- **Normal sounds**: Higher accuracy (low reconstruction loss)
- **Anomalous sounds**: Lower accuracy (high reconstruction loss)

## Dataset

This project uses the [ESC-50 Dataset](https://github.com/karoldvl/ESC-50), a collection of 2000 environmental audio recordings (5 seconds each) organized into 50 semantic classes.

### Available Sound Classes
- Engine, chainsaw, keyboard typing, mouse click
- Vacuum cleaner, washing machine
- Footsteps (used as anomaly example)

