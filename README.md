### Real-Time Sign Language Recognition

A proof-of-concept system that uses MediaPipe, TensorFlow, and OpenCV to recognize sign language gestures (e.g., "hello", "thanks", "I love you") in real time via a webcam feed.

---

#### Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Workflow](#project-workflow)
7. [Model Architecture](#model-architecture)
8. [Data Collection & Preprocessing](#data-collection--preprocessing)
9. [Training & Evaluation](#training--evaluation)
10. [Results](#results)
11. [Future Enhancements](#future-enhancements)
12. [File Structure](#file-structure)
13. [License](#license)

---

## Project Overview

This project implements a real-time sign language recognition system. It captures webcam video, extracts body, hand, and face landmarks using MediaPipe Holistic, and feeds sequential keypoint data into an LSTM-based neural network to classify gestures.

## Features

* Real-time webcam capture and processing
* Landmark detection with MediaPipe Holistic (pose, face, hands)
* Sequence-based LSTM neural network for temporal classification
* Support for multiple gestures ("hello", "thanks", "I love you")
* Model training, saving, and live prediction
* Evaluation with confusion matrix and accuracy metrics

## Prerequisites

* Python 3.8+
* Webcam or camera-enabled device

## Installation

1. Clone this repository:

   ```bash
   git clone [https://github.com/your-username/sign-language-recognition.git](https://github.com/SalimYs/TranslatingLaunguge)
   cd sign-language-recognition
   ```
2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\\Scripts\\activate   # Windows
   ```
3. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Collect data**: Run the data collection script to capture keypoint sequences for each action.

   ```bash
   python collect_data.py
   ```
2. **Train model**: Preprocess data and train the LSTM model.

   ```bash
   python train_model.py
   ```
3. **Run real-time prediction**: Launch the live demo.

   ```bash
   python detect_gesture.py
   ```

## Project Workflow

1. **Import and Install Dependencies**: TensorFlow, OpenCV, MediaPipe, Scikit-learn, Matplotlib
2. **Keypoint Extraction**: Use MediaPipe Holistic to detect and draw landmarks.
3. **Extract Keypoint Values**: Convert landmarks to structured NumPy arrays.
4. **Set Up Data Folders**: Organize `MP_Data/` with subfolders per gesture and sequence.
5. **Collect Data**: Capture and save `.npy` files for each frame sequence.
6. **Preprocess & Label**: Normalize data, one-hot encode labels, split into train/test sets.
7. **Build & Train Model**: Define LSTM network, compile, and fit to training data.
8. **Predict & Save**: Run live predictions and save model weights to `action_model.h5`.
9. **Evaluate**: Compute confusion matrix and accuracy on test set.
10. **Real-Time Demo**: Integrate webcam feed for live gesture recognition.

## Model Architecture

* **LSTM** layer with 64 units (returns sequences)
* **LSTM** layer with 128 units
* **Dense** layer with 64 units (ReLU)
* **Output** Dense layer with softmax activation (one neuron per gesture)
* **Loss**: Categorical Crossentropy
* **Optimizer**: Adam

## Data Collection & Preprocessing

* Data saved as NumPy `.npy` files organized by action and sequence index.
* Each sequence: 30 frames of concatenated pose, face, left-hand, and right-hand landmarks.
* Labels one-hot encoded and split 80/20 into train and test sets.

## Training & Evaluation

* **Epochs**: 200
* **Validation split**: 20%
* **Metrics**: Accuracy, confusion matrix

## Results

* Achieved \~90% accuracy on the validation set.
* Live prototype successfully recognizes target gestures in real time.

## Future Enhancements

* Expand the gesture vocabulary and dataset variety.
* Deploy as a standalone desktop/mobile application (PyQt, Flask, or Electron).
* Integrate external translation APIs for text/audio output.
* Optimize model for on-device inference (TensorFlow Lite).

## File Structure

```
├── collect_data.py       # Data collection script
├── train_model.py        # Model training pipeline
├── detect_gesture.py     # Real-time detection demo
├── MP_Data/           # Stored keypoint sequences
├── action_model.h5       # Trained model weights
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## License

This project is released under the MIT License. See `LICENSE` for details.
