"""
RealTimeML - Sign Language Detection
======================================
Author: Bishwanath20
Description: Full pipeline for training a CNN on sign language dataset
             and running real-time inference via webcam or test images.

Expected folder structure:
    train/
        A/  img1.jpg  img2.jpg ...
        B/  img1.jpg ...
        ...
    test/
        A/  img1.jpg ...
        B/  img1.jpg ...
        ...
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Deep-learning / vision ──────────────────────────────────────────────────
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (edit these or pass via CLI flags)
# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (edit these or pass via CLI flags)
# ────────────────────────────────────────────────────────────────────────────
CFG = {
    "train_dir":   "data/train",
    "test_dir":    "data/test",
    "model_path":  "models/sign_language_model.h5",
    "labels_path": "models/labels.json",
    "img_size":    (64, 64),       # resize all frames/images to this
    "batch_size":  32,
    "epochs":      20,
    "lr":          1e-3,
    "val_split":   0.15,           # fraction of train set used as validation
    "augment":     True,           # apply random augmentation during training
    "early_stop_patience": 5,
}


# ────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ────────────────────────────────────────────────────────────────────────────

def build_generators(cfg: dict):
    """
    Build PyTorch DataLoaders for train / validation / test splits.
    Handles missing test directory gracefully.
    """
    IMG_SIZE = cfg["img_size"]
    BATCH    = cfg["batch_size"]

    # ── Augmentation for training ──
    if cfg["augment"]:
        train_transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    test_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"\n[DATA] Loading training data from: {cfg['train_dir']}")
    
    from torchvision.datasets import ImageFolder
    
    train_dataset = ImageFolder(cfg["train_dir"], transform=train_transform)
    val_dataset = ImageFolder(cfg["train_dir"], transform=test_transform)
    
    # Split train into train/val
    val_split = cfg["val_split"]
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    _, val_dataset = torch.utils.data.random_split(val_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)

    test_loader = None
    if Path(cfg["test_dir"]).exists():
        print(f"[DATA] Loading test data from:     {cfg['test_dir']}")
        test_dataset = ImageFolder(cfg["test_dir"], transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)
    else:
        print(f"[DATA] ⚠ Test directory '{cfg['test_dir']}' not found – skipping.")

    # Save label map
    label_map = {v: k for k, v in train_dataset.dataset.class_to_idx.items()}
    # Convert to string keys for JSON storage
    label_map_json = {str(k): v for k, v in label_map.items()}
    with open(cfg["labels_path"], "w") as f:
        json.dump(label_map_json, f, indent=2)
    print(f"[DATA] {len(label_map)} classes found: {list(train_dataset.dataset.class_to_idx.keys())}")
    print(f"[DATA] Label map saved → {cfg['labels_path']}")

    return train_loader, val_loader, test_loader, label_map


# ────────────────────────────────────────────────────────────────────────────
# 2.  MODEL DEFINITION
# ────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, img_size: tuple) -> nn.Module:
    """
    Lightweight but effective CNN for sign-language recognition.
    Uses BatchNorm + Dropout to reduce over-fitting on small datasets.
    """
    class SignLanguageCNN(nn.Module):
        def __init__(self, num_classes):
            super(SignLanguageCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool = nn.MaxPool2d(2, 2)
            
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            
            self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
            self.bn4 = nn.BatchNorm2d(256)
            
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            
            self.fc1 = nn.Linear(256, 256)
            self.dropout1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(256, 128)
            self.dropout2 = nn.Dropout(0.3)
            self.fc3 = nn.Linear(128, num_classes)
            
        def forward(self, x):
            x = self.pool(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool(torch.relu(self.bn2(self.conv2(x))))
            x = self.pool(torch.relu(self.bn3(self.conv3(x))))
            x = self.pool(torch.relu(self.bn4(self.conv4(x))))
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.dropout1(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout2(x)
            x = self.fc3(x)
            return x
    
    model = SignLanguageCNN(num_classes)
    return model


# ────────────────────────────────────────────────────────────────────────────
# 3.  TRAINING
# ────────────────────────────────────────────────────────────────────────────

def train(cfg: dict):
    train_loader, val_loader, test_loader, label_map = build_generators(cfg)
    num_classes = len(label_map)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes, cfg["img_size"]).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    
    print(f"[TRAIN] Using device: {device}")
    model.train()

    best_val_acc = 0.0
    patience = cfg["early_stop_patience"]
    patience_counter = 0

    print("\n[TRAIN] Starting training …")
    for epoch in range(cfg["epochs"]):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{cfg['epochs']}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), cfg["model_path"])
            print(f"[CHECKPOINT] Saved best model with val_acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[EARLY STOP] No improvement for {patience} epochs. Stopping training.")
                break

    # Load best model
    model.load_state_dict(torch.load(cfg["model_path"]))
    
    _plot_history_from_logs()  # We'll need to modify this

    # ── Evaluate on test set ──
    if test_loader is not None:
        print("\n[EVAL] Evaluating on test set …")
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_acc = 100. * test_correct / test_total
        test_loss = test_loss / len(test_loader)
        print(f"[EVAL] Test accuracy: {test_acc:.2f}%  |  Test loss: {test_loss:.4f}")

        # Get unique classes in test set and their names
        unique_labels = np.unique(all_labels)
        class_names = [label_map[int(i)] for i in unique_labels]

        print("\n[EVAL] Classification Report:")
        print(classification_report(all_labels, all_preds, labels=list(unique_labels), target_names=class_names))

        _plot_confusion_matrix(all_labels, all_preds, class_names)

    print(f"\n[DONE] Model saved → {cfg['model_path']}")


def _plot_history_from_logs():
    # For now, just create a placeholder plot
    # In a full implementation, we'd collect metrics during training
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].set_title("Accuracy (placeholder)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_title("Loss (placeholder)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    plt.tight_layout()
    plt.savefig("outputs/training_history.png", dpi=120)
    print("[PLOT] Training curves placeholder saved → outputs/training_history.png")
    plt.close()


def _plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names)-2)))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=120)
    print("[PLOT] Confusion matrix saved → outputs/confusion_matrix.png")
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# 4.  INFERENCE HELPERS
# ────────────────────────────────────────────────────────────────────────────

def load_inference_assets(cfg: dict):
    if not Path(cfg["model_path"]).exists():
        sys.exit(f"[ERROR] Model not found at '{cfg['model_path']}'. Run with --mode train first.")
    if not Path(cfg["labels_path"]).exists():
        sys.exit(f"[ERROR] Labels not found at '{cfg['labels_path']}'. Run with --mode train first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(cfg["labels_path"]) as f:
        label_map = json.load(f)   # {str(idx): class_name}
    
    num_classes = len(label_map)
    model = build_model(num_classes, cfg["img_size"])
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device))
    model.to(device)
    model.eval()
    
    return model, label_map, device


def preprocess_frame(frame: np.ndarray, img_size: tuple) -> torch.Tensor:
    """Resize, normalise and expand dims for model input."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(frame).unsqueeze(0)


def predict_frame(model, frame, label_map, img_size, device, top_k=3):
    """Return list of (label, confidence) sorted by confidence desc."""
    x = preprocess_frame(frame, img_size).to(device)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [(label_map[str(i)], float(probs[i])) for i in top_indices]


# ────────────────────────────────────────────────────────────────────────────
# 5.  REAL-TIME WEBCAM INFERENCE
# ────────────────────────────────────────────────────────────────────────────

def run_webcam(cfg: dict):
    model, label_map, device = load_inference_assets(cfg)
    IMG_SIZE = cfg["img_size"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("[ERROR] Cannot open webcam.")

    print("[WEBCAM] Press  Q  to quit.")
    BOX_X1, BOX_Y1, BOX_X2, BOX_Y2 = 100, 100, 400, 400   # region-of-interest

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Draw ROI box
        cv2.rectangle(frame, (BOX_X1, BOX_Y1), (BOX_X2, BOX_Y2), (0, 255, 0), 2)
        roi = frame[BOX_Y1:BOX_Y2, BOX_X1:BOX_X2]

        results = predict_frame(model, roi, label_map, IMG_SIZE, device)

        # Overlay predictions
        y_offset = BOX_Y2 + 25
        for rank, (label, conf) in enumerate(results):
            color = (0, 255, 0) if rank == 0 else (200, 200, 200)
            text  = f"{'→ ' if rank==0 else '   '}{label}: {conf*100:.1f}%"
            cv2.putText(frame, text, (BOX_X1, y_offset + rank * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        cv2.imshow("Sign Language Detection  [Q = quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ────────────────────────────────────────────────────────────────────────────
# 6.  BATCH TEST-SET EVALUATION  (no camera required)
# ────────────────────────────────────────────────────────────────────────────

def run_evaluate(cfg: dict):
    model, label_map, device = load_inference_assets(cfg)

    if not Path(cfg["test_dir"]).exists():
        sys.exit(f"[ERROR] Test directory '{cfg['test_dir']}' not found.")

    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    
    test_transform = transforms.Compose([
        transforms.Resize(cfg["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = ImageFolder(cfg["test_dir"], transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100. * test_correct / test_total
    test_loss = test_loss / len(test_loader)
    print(f"\n[EVAL] Accuracy: {test_acc:.2f}%  |  Loss: {test_loss:.4f}")

    # Get unique classes in test set and their names
    unique_labels = np.unique(all_labels)
    class_to_idx = test_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[int(i)] for i in unique_labels]

    print("\n[EVAL] Classification Report:")
    print(classification_report(all_labels, all_preds, labels=list(unique_labels), target_names=class_names))
    _plot_confusion_matrix(all_labels, all_preds, class_names)


# ────────────────────────────────────────────────────────────────────────────
# 7.  PREDICT A SINGLE IMAGE FILE
# ────────────────────────────────────────────────────────────────────────────

def run_predict_image(cfg: dict, image_path: str):
    model, label_map, device = load_inference_assets(cfg)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        sys.exit(f"[ERROR] Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = predict_frame(model, img_rgb, label_map, cfg["img_size"], device)
    print(f"\n[PREDICT] Results for: {image_path}")
    for rank, (label, conf) in enumerate(results):
        marker = "★" if rank == 0 else " "
        print(f"  {marker} {label:20s} {conf*100:6.2f}%")


# ────────────────────────────────────────────────────────────────────────────
# 8.  ENTRY POINT
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="RealTimeML – Sign Language Detection",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["train", "webcam", "evaluate", "predict"],
        default="train",
        help=(
            "train    – train model on train/ folder\n"
            "webcam   – live prediction via webcam\n"
            "evaluate – batch evaluation on test/ folder\n"
            "predict  – predict a single image (use --image)\n"
        ),
    )
    p.add_argument("--train_dir",   default=CFG["train_dir"])
    p.add_argument("--test_dir",    default=CFG["test_dir"])
    p.add_argument("--model_path",  default=CFG["model_path"])
    p.add_argument("--labels_path", default=CFG["labels_path"])
    p.add_argument("--epochs",      type=int,   default=CFG["epochs"])
    p.add_argument("--batch_size",  type=int,   default=CFG["batch_size"])
    p.add_argument("--img_size",    type=int,   nargs=2, default=list(CFG["img_size"]),
                   metavar=("W", "H"))
    p.add_argument("--no_augment",  action="store_true",
                   help="Disable training augmentation")
    p.add_argument("--image",       default=None,
                   help="Path to image file (used with --mode predict)")
    return p.parse_args()


def main():
    args = parse_args()

    # Merge CLI args into config
    cfg = dict(CFG)
    cfg["train_dir"]   = args.train_dir
    cfg["test_dir"]    = args.test_dir
    cfg["model_path"]  = args.model_path
    cfg["labels_path"] = args.labels_path
    cfg["epochs"]      = args.epochs
    cfg["batch_size"]  = args.batch_size
    cfg["img_size"]    = tuple(args.img_size)
    cfg["augment"]     = not args.no_augment

    print("=" * 60)
    print("  RealTimeML – Sign Language Detection")
    print(f"  Mode      : {args.mode}")
    print(f"  PyTorch   : {torch.__version__}")
    print(f"  CUDA      : {torch.cuda.is_available()}")
    print("=" * 60)

    if args.mode == "train":
        train(cfg)
    elif args.mode == "webcam":
        run_webcam(cfg)
    elif args.mode == "evaluate":
        run_evaluate(cfg)
    elif args.mode == "predict":
        if not args.image:
            sys.exit("[ERROR] --mode predict requires --image <path>")
        run_predict_image(cfg, args.image)


if __name__ == "__main__":
    main()