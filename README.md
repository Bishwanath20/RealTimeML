# RealTimeML – Sign Language Detection

A real-time sign language recognition system using PyTorch CNN.

## Project Structure

```
RealTimeML-main/
├── main.py                 # Main script (train, evaluate, webcam, predict)
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── models/                # Trained models and labels
│   ├── sign_language_model.h5
│   └── labels.json
├── outputs/               # Training outputs and visualizations
│   ├── training_history.png
│   └── confusion_matrix.png
└── data/                  # Dataset
    ├── train/             # Training images (organized by class)
    │   ├── hello/
    │   ├── iloveyou/
    │   ├── no/
    │   ├── thanks/
    │   └── yes/
    └── test/              # Test images (organized by class)
        ├── hello/
        └── iloveyou/
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train the model
```bash
python main.py --mode train --epochs 20
```

### Real-time webcam detection
```bash
python main.py --mode webcam
```
Press **Q** to quit.

### Evaluate on test set
```bash
python main.py --mode evaluate
```

### Predict a single image
```bash
python main.py --mode predict --image path/to/image.jpg
```

## Options

- `--epochs N` - Number of training epochs (default: 20)
- `--batch_size N` - Batch size (default: 32)
- `--img_size W H` - Image dimensions (default: 64 64)
- `--no_augment` - Disable data augmentation
- `--train_dir PATH` - Custom training folder (default: data/train)
- `--test_dir PATH` - Custom test folder (default: data/test)

## Model Details

- **Architecture:** CNN with 4 convolutional layers, batch normalization, and dropout
- **Framework:** PyTorch
- **Classes:** hello, iloveyou, no, thanks, yes
- **Input Size:** 64×64 RGB images
