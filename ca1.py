#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics import classification_report

# ==========================================
# 0. SETUP & REPRODUCIBILITY
# ==========================================
os.makedirs('./ca1/data', exist_ok=True)
os.makedirs('./ca1/img', exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==========================================
# 1. DATA PREPARATION (All Loaders)
# ==========================================
print("--- PREPARING DATA ---")
train_data_raw = datasets.FashionMNIST(root='./ca1/data', train=True, download=True)
data_tensor = train_data_raw.data.float() / 255.0
mean, std = data_tensor.mean(), data_tensor.std()

# 1. Baseline Transform (No Augmentation)
transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean.item(),), (std.item(),))
])

# 2. Augmented Transform
transform_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),      
    transforms.RandomRotation(degrees=10),       
    transforms.ToTensor(),
    transforms.Normalize((mean.item(),), (std.item(),))
])

# Load Datasets
train_data_base = datasets.FashionMNIST(root='./ca1/data', train=True, download=True, transform=transform_base)
train_data_aug = datasets.FashionMNIST(root='./ca1/data', train=True, download=True, transform=transform_aug)
val_test_data = datasets.FashionMNIST(root='./ca1/data', train=True, download=True, transform=transform_base)

# 80/20 Dynamic Split
train_size = int(0.8 * len(train_data_base))
val_size = len(train_data_base) - train_size
split_generator = torch.Generator().manual_seed(42)
indices = torch.randperm(len(train_data_base), generator=split_generator).tolist()

# Create Loaders
train_loader_base = DataLoader(torch.utils.data.Subset(train_data_base, indices[:train_size]), batch_size=64, shuffle=True)
train_loader_aug = DataLoader(torch.utils.data.Subset(train_data_aug, indices[:train_size]), batch_size=64, shuffle=True)
val_loader = DataLoader(torch.utils.data.Subset(val_test_data, indices[train_size:]), batch_size=64, shuffle=False)

test_data = datasets.FashionMNIST(root='./ca1/data', train=False, download=True, transform=transform_base)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

def get_accuracy(outputs, labels):
    _, predictions = torch.max(outputs, 1)
    correct = (predictions == labels).sum().item()
    return correct / len(labels)


# ==========================================
# PHASE 1: BASELINE MODEL
# ==========================================
print("\n======================================")
print("   PHASE 1: BASELINE CNN (NO AUGMENTATION)   ")
print("======================================")

base_model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(0.2),
    nn.Flatten(),               
    nn.Linear(32*12*12, 128),
    nn.ReLU(),
    nn.Linear(128, 10)          
)

optimizer = torch.optim.Adam(base_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

base_train_loss, base_val_loss = [], []
base_train_acc, base_val_acc = [], []

epochs = 10
for epoch in range(epochs):
    base_model.train()  
    t_loss, t_acc, start_time = 0, 0, time.time()
    
    for images, labels in train_loader_base: # <--- USING BASELINE LOADER
        optimizer.zero_grad()
        output = base_model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        t_acc += get_accuracy(output, labels)
        
    base_train_loss.append(t_loss / len(train_loader_base))
    base_train_acc.append(t_acc / len(train_loader_base))

    base_model.eval() 
    with torch.no_grad():
        v_loss, v_acc = 0, 0
        for val_images, val_labels in val_loader:
            val_output = base_model(val_images)
            v_loss += criterion(val_output, val_labels).item()
            v_acc += get_accuracy(val_output, val_labels)
        base_val_loss.append(v_loss / len(val_loader))
        base_val_acc.append(v_acc / len(val_loader))
        
    print(f"Epoch {epoch+1:02d}/{epochs} | Train Acc: {base_train_acc[-1]:.4f} | Val Acc: {base_val_acc[-1]:.4f}")

plt.figure(figsize=(6, 8))
plt.subplot(2, 1, 1); plt.plot(base_train_loss, 'k', label='Train'); plt.plot(base_val_loss, 'darkred', label='Val'); plt.title('Baseline Loss'); plt.legend(frameon=False)
plt.subplot(2, 1, 2); plt.plot(base_train_acc, 'k'); plt.plot(base_val_acc, 'darkred'); plt.title('Baseline Accuracy')
plt.tight_layout(); plt.savefig("./ca1/img/baseline_performance.png", bbox_inches='tight', dpi=300); plt.close()


# ==========================================
# PHASE 2: AUGMENTATION MODEL
# ==========================================
print("\n======================================")
print(" PHASE 2: BASELINE CNN + AUGMENTATION ")
print("======================================")

# Reinitialize the exact same architecture for a fair test
aug_model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(0.2),
    nn.Flatten(),               
    nn.Linear(32*12*12, 128),
    nn.ReLU(),
    nn.Linear(128, 10)          
)

optimizer = torch.optim.Adam(aug_model.parameters(), lr=0.001)

aug_train_loss, aug_val_loss = [], []
aug_train_acc, aug_val_acc = [], []

epochs = 15

for epoch in range(epochs):
    aug_model.train()  
    t_loss, t_acc = 0, 0
    
    for images, labels in train_loader_aug: # <--- USING AUGMENTED LOADER
        optimizer.zero_grad()
        output = aug_model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        t_acc += get_accuracy(output, labels)
        
    aug_train_loss.append(t_loss / len(train_loader_aug))
    aug_train_acc.append(t_acc / len(train_loader_aug))

    aug_model.eval() 
    with torch.no_grad():
        v_loss, v_acc = 0, 0
        for val_images, val_labels in val_loader:
            val_output = aug_model(val_images)
            v_loss += criterion(val_output, val_labels).item()
            v_acc += get_accuracy(val_output, val_labels)
        aug_val_loss.append(v_loss / len(val_loader))
        aug_val_acc.append(v_acc / len(val_loader))
        
    print(f"Epoch {epoch+1:02d}/{epochs} | Train Acc: {aug_train_acc[-1]:.4f} | Val Acc: {aug_val_acc[-1]:.4f}")

plt.figure(figsize=(6, 8))
plt.subplot(2, 1, 1); plt.plot(aug_train_loss, 'k', label='Train'); plt.plot(aug_val_loss, 'darkred', label='Val'); plt.title('Augmented Loss'); plt.legend(frameon=False)
plt.subplot(2, 1, 2); plt.plot(aug_train_acc, 'k'); plt.plot(aug_val_acc, 'darkred'); plt.title('Augmented Accuracy')
plt.tight_layout(); plt.savefig("./ca1/img/augment_performance.png", bbox_inches='tight', dpi=300); plt.close()


# ==========================================
# PHASE 3: FINAL UPGRADED MODEL
# ==========================================
print("\n======================================")
print(" PHASE 3: UPGRADED CNN + LR SCHEDULER ")
print("======================================")

final_model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2), 
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    
    nn.Flatten(),               
    nn.Dropout(0.5),             
    nn.Linear(64 * 7 * 7, 128),
    nn.ReLU(),
    nn.Linear(128, 10)          
)

optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

final_train_loss, final_val_loss = [], []
final_train_acc, final_val_acc = [], []

epochs_final = 20  # Train longer to see the scheduler work
for epoch in range(epochs_final):
    final_model.train()  
    t_loss, t_acc = 0, 0
    
    for images, labels in train_loader_aug: 
        optimizer.zero_grad()
        output = final_model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        t_acc += get_accuracy(output, labels)
        
    final_train_loss.append(t_loss / len(train_loader_aug))
    final_train_acc.append(t_acc / len(train_loader_aug))

    final_model.eval() 
    with torch.no_grad():
        v_loss, v_acc = 0, 0
        for val_images, val_labels in val_loader:
            val_output = final_model(val_images)
            v_loss += criterion(val_output, val_labels).item()
            v_acc += get_accuracy(val_output, val_labels)
        final_val_loss.append(v_loss / len(val_loader))
        final_val_acc.append(v_acc / len(val_loader))
        
    print(f"Epoch {epoch+1:02d}/{epochs_final} | Train Acc: {final_train_acc[-1]:.4f} | Val Acc: {final_val_acc[-1]:.4f}")
    scheduler.step(final_val_loss[-1])

plt.figure(figsize=(6, 8))
plt.subplot(2, 1, 1); plt.plot(final_train_loss, 'k', label='Train'); plt.plot(final_val_loss, 'darkred', label='Val'); plt.title('Final Upgraded Loss'); plt.legend(frameon=False)
plt.subplot(2, 1, 2); plt.plot(final_train_acc, 'k'); plt.plot(final_val_acc, 'darkred'); plt.title('Final Upgraded Accuracy')
plt.tight_layout(); plt.savefig("./ca1/img/final_performance.png", bbox_inches='tight', dpi=300); plt.close()


# ==========================================
# 4. TESTING & CLASSIFICATION REPORT
# ==========================================
print("\n======================================")
print("     FINAL TEST SET EVALUATION        ")
print("======================================")

final_model.eval()
all_predictions = []
all_true_labels = []

with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_output = final_model(test_images)
        _, predictions = torch.max(test_output, 1)
        all_predictions.extend(predictions.cpu().numpy())
        all_true_labels.extend(test_labels.cpu().numpy())

fashion_mnist_classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print(classification_report(all_true_labels, all_predictions, target_names=fashion_mnist_classes))
print("\n✅ Script complete. All three performance graphs saved to ./ca1/img/")
