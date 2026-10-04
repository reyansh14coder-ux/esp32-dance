import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import (
    BATCH_SIZE,
    DATASET_DIR,
    DANCE_CLASSES,
    EPOCHS,
    LABEL_MAP_PATH,
    LEARNING_RATE,
    MODEL_PATH,
    SEQUENCE_LENGTH,
)
from model import DanceLSTM, feature_size, pad_or_trim


class PoseDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        seq = np.load(path).astype(np.float32)
        seq = pad_or_trim(seq, SEQUENCE_LENGTH)
        return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)


def load_items():
    items = []
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".npy")]
    for fname in files:
        label = fname.rsplit("_", 1)[0]
        if label not in DANCE_CLASSES:
            print(f"skipping unknown label: {fname}")
            continue
        items.append((os.path.join(DATASET_DIR, fname), DANCE_CLASSES.index(label)))
    return items


def augment(seq):
    seq = seq.copy()
    if random.random() < 0.3:
        jitter = np.random.normal(0, 0.02, seq.shape).astype(np.float32)
        seq[:, :, :2] += jitter[:, :, :2]
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.01, seq.shape).astype(np.float32)
        seq = seq + noise
    return seq


def main():
    items = load_items()
    if not items:
        raise SystemExit(f"no .npy files in {DATASET_DIR}, run collect.py first")

    counts = {}
    for _, y in items:
        counts[DANCE_CLASSES[y]] = counts.get(DANCE_CLASSES[y], 0) + 1
    print("class counts:", counts)
    if len(counts) < 2:
        raise SystemExit("need at least 2 classes with data")

    random.seed(42)
    random.shuffle(items)
    split = int(len(items) * 0.8)
    train_items, val_items = items[:split], items[split:]
    if not val_items:
        val_items = train_items[-1:]

    train_ds = PoseDataset(train_items)
    val_ds = PoseDataset(val_items)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DanceLSTM(num_classes=len(DANCE_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_items)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    label_map = {name: i for i, name in enumerate(DANCE_CLASSES)}
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, correct, running = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if torch.rand(1).item() < 0.5:
                x = x + torch.randn_like(x) * 0.01
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        train_acc = correct / total

        model.eval()
        v_total, v_correct, v_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                v_loss += criterion(logits, y).item() * len(y)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total += len(y)
        val_acc = v_correct / v_total
        scheduler.step(v_loss / v_total)

        print(f"epoch {epoch:03d}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {"model": model.state_dict(), "classes": DANCE_CLASSES, "val_acc": val_acc},
                MODEL_PATH,
            )

    print(f"best val_acc={best_acc:.3f}  saved to {MODEL_PATH}")


def class_weights(items):
    counts = np.zeros(len(DANCE_CLASSES))
    for _, y in items:
        counts[y] += 1
    counts = np.maximum(counts, 1)
    w = counts.sum() / (len(DANCE_CLASSES) * counts)
    return torch.tensor(w, dtype=torch.float32)


if __name__ == "__main__":
    main()
