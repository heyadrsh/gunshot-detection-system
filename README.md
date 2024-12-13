# Gunshot Detection System

A real-time gunshot detection and classification system using deep learning. This system can identify different types of firearms based on their acoustic signatures using a laptop's microphone.

## Supported Weapons
- AK-12
- AK-47
- IMI Desert Eagle
- M16
- M249
- M4
- MG-42
- MP5
- Zastava M92

## Requirements
- Python 3.7+
- Dependencies listed in `requirements.txt`

## Installation
```bash
pip install -r requirements.txt
```

## Usage

1. **Training the Model**
   ```bash
   python gunshot_detection.py
   # Choose option 1 to train
   ```

2. **Real-time Detection**
   ```bash
   python gunshot_detection.py
   # Choose option 2 to start detection
   ```

## Dataset Structure
```
dataset/
    ├── AK-12/
    │   └── [wav files]
    ├── AK-47/
    │   └── [wav files]
    └── ...
```

## Model Architecture
- CNN-based architecture
- Uses mel spectrograms for audio feature extraction
- Real-time processing capability
- Confidence threshold of 0.8 to minimize false positives

## Pre-trained Model
The trained model will be saved as `gunshot_model.h5`. You can share this file to use the trained weights on other machines.

## Note
- Ensure audio files are in WAV format
- Model must be trained before detection can work
- Training duration depends on dataset size and computational resources 