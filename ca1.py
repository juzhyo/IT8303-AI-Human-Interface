#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt
import sys

train_data = datasets.FashionMNIST(root='./ca1/data', train=True, download=True)

# Determine normalization parameters
data = train_data.data.float()/255.0
mean = data.mean()
std = data.std()

print(f"Calculated Mean: {mean:.4f}")
print(f"Calculated Std Dev: {std:.4f}\n")

# Define the transform to convert images to PyTorch tensors
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean.item(),), (std.item(),))
])

# Split into training and validation sets
train_data = datasets.FashionMNIST(root='./ca1/data', train=True, download=True, transform=transform)
train_size = int(0.8*len(train_data))
val_size = len(train_data) - train_size 
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# Load test data
test_data = datasets.FashionMNIST(root='./ca1/data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
test_size = len(test_data)

print(f'***NUMBER OF DATASET IMAGES***')
print(f'Training:   {train_size}')
print(f'Validation: {val_size}')
print(f'Test:       {test_size}')
print(f'Total:      {train_size+val_size+test_size}\n')

# Sample 20 training images
sample_indices = range(20)
sample_images, sample_labels = zip(*[train_data[i] for i in sample_indices])

# Visualize the 20 sample images
plt.figure(figsize=(4, 6))
for i in range(20):
    plt.subplot(5, 4, i + 1)
    plt.imshow(sample_images[i].squeeze(), cmap='gray')
    plt.axis('off')
plt.savefig('./ca1/img/samples.png',bbox_inches='tight',dpi=300)

# Metrics
def get_accuracy(outputs, labels):
    _, predictions = torch.max(outputs, 1)
    correct = (predictions == labels).sum().item()
    return correct / len(labels)

######################
# Baseline CNN model #
######################
print("***IMPLEMENT BASELINE CNN MODEL***")

# Define a simple CNN model
cnn_model = nn.Sequential(
    # First Block: (1, 28, 28) --conv--> (32, 24, 24) --maxpool--> (32, 12, 12)
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    
    # Classifier
    nn.Dropout(0.2),
    nn.Flatten(),               # Flattens 32*12*12 into 4608
    nn.Linear(32*12*12, 128),
    nn.ReLU(),
    nn.Linear(128, 10)          # 10 output classes
)

# Train CNN model
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Variables to store loss and accuracy history
cnn_train_loss_history = []
cnn_val_loss_history = []
cnn_train_acc_history = []
cnn_val_acc_history = []

# Train the model for a few epochs
epochs = 10
for epoch in range(epochs):
    cnn_model.train()
    train_loss = 0
    train_acc = 0
    start_time = time.time()
    
    for images, labels in train_loader:
        optimizer.zero_grad()           # Reset gradients
        output = cnn_model(images)      # Forward pass
        loss = criterion(output, labels) # Calculate loss
        loss.backward()                 # Backward pass
        optimizer.step()                # Update weights
        
        train_loss += loss.item()
        train_acc += get_accuracy(output, labels)
        
    end_time = time.time()
    
    # Track loss and accuracy for training set
    cnn_train_loss_history.append(train_loss / len(train_loader))
    cnn_train_acc_history.append(train_acc / len(train_loader))

    # Validate the model
    cnn_model.eval()
    with torch.no_grad():
        val_loss = 0
        val_acc = 0
        for val_images, val_labels in val_loader:
            val_output = cnn_model(val_images)
            val_loss += criterion(val_output, val_labels).item()
            val_acc += get_accuracy(val_output, val_labels)
            
        cnn_val_loss_history.append(val_loss / len(val_loader))
        cnn_val_acc_history.append(val_acc / len(val_loader))
        
    print(f"Epoch {epoch+1} done in {end_time - start_time:.2f} seconds. "
          f"Train Loss: {cnn_train_loss_history[-1]:.4f}, Train Acc: {cnn_train_acc_history[-1]:.4f}, "
          f"Val Loss: {cnn_val_loss_history[-1]:.4f}, Val Acc: {cnn_val_acc_history[-1]:.4f}")

# Test the model on the test set
cnn_model.eval()  # Set model to evaluation mode
test_loss = 0
test_acc = 0
with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_output = cnn_model(test_images)
        test_loss += criterion(test_output, test_labels).item()
        test_acc += get_accuracy(test_output, test_labels)
print(f"\nTest Loss: {test_loss / len(test_loader):.4f}, Test Acc: {test_acc / len(test_loader):.4f}\n")

# Plot training and validation loss/accuracy curves
plt.figure(figsize=(6, 8))

plt.subplot(2, 1, 1)
plt.plot(cnn_train_loss_history, label='Train Loss',color='k',alpha=0.6)
plt.plot(cnn_val_loss_history, label='Validation Loss',color='darkred',alpha=0.6)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(frameon=False)
plt.title('CNN Training and Validation Loss')

plt.subplot(2, 1, 2)
plt.plot(cnn_train_acc_history, label='Train Accuracy',color='k',alpha=0.6)
plt.plot(cnn_val_acc_history, label='Validation Accuracy',color='darkred',alpha=0.6)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(frameon=False)
plt.title('CNN Training and Validation Accuracy')

plt.tight_layout()

plt.savefig("./ca1/img/baseline_perfomance.png",bbox_inches='tight',dpi=300)

#####################
# Data Augmentation #
#####################
print("***IMPLEMENT DATA AUGMENTATION***")

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),      
    transforms.RandomRotation(degrees=10),       
    transforms.ToTensor(),
    transforms.Normalize((mean.item(),), (std.item(),))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean.item(),), (std.item(),))
])

train_data_full = datasets.FashionMNIST(root='./ca1/data', train=True, download=True, transform=train_transform)
val_data_full = datasets.FashionMNIST(root='./ca1/data', train=True, download=True, transform=test_transform)

train_size = int(0.8*len(train_data_full))
val_size = len(train_data_full) - train_size

split_generator = torch.Generator().manual_seed(42)
indices = torch.randperm(len(train_data_full), generator=split_generator).tolist()

train_data = torch.utils.data.Subset(train_data_full, indices[:train_size])
val_data = torch.utils.data.Subset(val_data_full, indices[train_size:])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

test_data = datasets.FashionMNIST(root='./ca1/data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Define a simple CNN model
cnn_model = nn.Sequential(
    # First Block: (1, 28, 28) --conv--> (32, 24, 24) --maxpool--> (32, 12, 12)
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    
    # Classifier
    nn.Dropout(0.2),
    nn.Flatten(),               # Flattens 32*12*12 into 4608
    nn.Linear(32*12*12, 128),
    nn.ReLU(),
    nn.Linear(128, 10)          # 10 output classes
)

# Train CNN model
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Variables to store loss and accuracy history
cnn_train_loss_history = []
cnn_val_loss_history = []
cnn_train_acc_history = []
cnn_val_acc_history = []

# Train the model for a few epochs
epochs = 15
for epoch in range(epochs):
    cnn_model.train()  # IMPORTANT: Set model to training mode at the start of each epoch
    train_loss = 0
    train_acc = 0
    start_time = time.time()
    
    for images, labels in train_loader:
        optimizer.zero_grad()           # Reset gradients
        output = cnn_model(images)      # Forward pass
        loss = criterion(output, labels) # Calculate loss
        loss.backward()                 # Backward pass
        optimizer.step()                # Update weights
        
        train_loss += loss.item()
        train_acc += get_accuracy(output, labels)
        
    end_time = time.time()
    
    # Track loss and accuracy for training set
    cnn_train_loss_history.append(train_loss / len(train_loader))
    cnn_train_acc_history.append(train_acc / len(train_loader))

    # Validate the model
    cnn_model.eval()  # IMPORTANT: Set to evaluation mode to disable Dropout during validation
    with torch.no_grad():
        val_loss = 0
        val_acc = 0
        for val_images, val_labels in val_loader:
            val_output = cnn_model(val_images)
            val_loss += criterion(val_output, val_labels).item()
            val_acc += get_accuracy(val_output, val_labels)
            
        cnn_val_loss_history.append(val_loss / len(val_loader))
        cnn_val_acc_history.append(val_acc / len(val_loader))
        
    print(f"Epoch {epoch+1} done in {end_time - start_time:.2f} seconds. "
          f"Train Loss: {cnn_train_loss_history[-1]:.4f}, Train Acc: {cnn_train_acc_history[-1]:.4f}, "
          f"Val Loss: {cnn_val_loss_history[-1]:.4f}, Val Acc: {cnn_val_acc_history[-1]:.4f}")

# Test the model on the test set
cnn_model.eval()  # Set model to evaluation mode
test_loss = 0
test_acc = 0
with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_output = cnn_model(test_images)
        test_loss += criterion(test_output, test_labels).item()
        test_acc += get_accuracy(test_output, test_labels)
print(f"\nTest Loss: {test_loss / len(test_loader):.4f}, Test Acc: {test_acc / len(test_loader):.4f}\n")

# Plot training and validation loss/accuracy curves
plt.figure(figsize=(6, 8))

plt.subplot(2, 1, 1)
plt.plot(cnn_train_loss_history, label='Train Loss',color='k',alpha=0.6)
plt.plot(cnn_val_loss_history, label='Validation Loss',color='darkred',alpha=0.6)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(frameon=False)
plt.title('CNN Training and Validation Loss')

plt.subplot(2, 1, 2)
plt.plot(cnn_train_acc_history, label='Train Accuracy',color='k',alpha=0.6)
plt.plot(cnn_val_acc_history, label='Validation Accuracy',color='darkred',alpha=0.6)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(frameon=False)
plt.title('CNN Training and Validation Accuracy')
plt.tight_layout()
plt.savefig("./ca1/img/augment_perfomance.png",bbox_inches='tight',dpi=300)

#####################
# Model improvement #
#####################
print("***IMPLEMENT ARCHITECTURAL UPGRADES***")

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),      # 50% chance to flip horizontally
    transforms.RandomRotation(degrees=10),       # Rotate by up to +/- 10 degrees
    transforms.ToTensor(),
    transforms.Normalize((mean.item(),), (std.item(),))
])

# Val/Test transform ONLY gets tensor conversion and normalization
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean.item(),), (std.item(),))
])

# 2. Load the dataset twice with the different transforms
train_data_full = datasets.FashionMNIST(root='./ca1/data', train=True, download=True, transform=train_transform)
val_data_full = datasets.FashionMNIST(root='./ca1/data', train=True, download=True, transform=test_transform)

# 3. Calculate sizes and generate fixed random indices
train_size = int(0.8 * len(train_data_full))
val_size = len(train_data_full) - train_size

# We use randperm to generate a shuffled list of indices, locked by our generator
split_generator = torch.Generator().manual_seed(17)
indices = torch.randperm(len(train_data_full), generator=split_generator).tolist()

# 4. Create the final datasets using Subsets
train_data = torch.utils.data.Subset(train_data_full, indices[:train_size])
val_data = torch.utils.data.Subset(val_data_full, indices[train_size:])

# 5. Create DataLoaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# Load test data (using test_transform)
test_data = datasets.FashionMNIST(root='./ca1/data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

cnn_model = nn.Sequential(
    # First Block: (1, 28, 28) --> (32, 12, 12)
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2), # Added padding to keep spatial size
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    
    # Second Block: (32, 14, 14) --> (64, 7, 7)
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    
    # Classifier
    nn.Flatten(),               # Flattens 64 * 7 * 7 into 3136
    nn.Dropout(0.5),            # Increased dropout for better regularization
    nn.Linear(64 * 7 * 7, 128),
    nn.ReLU(),
    nn.Linear(128, 10)          
)

# Train CNN model
cnn_model.train()  # Set model to training mode
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Variables to store loss and accuracy history
cnn_train_loss_history = []
cnn_val_loss_history = []
cnn_train_acc_history = []
cnn_val_acc_history = []

# Train the model for a few epochs
epochs = 30
for epoch in range(epochs):
    cnn_model.train()  # IMPORTANT: Set model to training mode at the start of each epoch
    train_loss = 0
    train_acc = 0
    start_time = time.time()
    
    for images, labels in train_loader:
        optimizer.zero_grad()           # Reset gradients
        output = cnn_model(images)      # Forward pass
        loss = criterion(output, labels) # Calculate loss
        loss.backward()                 # Backward pass
        optimizer.step()                # Update weights
        
        train_loss += loss.item()
        train_acc += get_accuracy(output, labels)
        
    end_time = time.time()
    
    # Track loss and accuracy for training set
    cnn_train_loss_history.append(train_loss / len(train_loader))
    cnn_train_acc_history.append(train_acc / len(train_loader))

    # Validate the model
    cnn_model.eval()  # IMPORTANT: Set to evaluation mode to disable Dropout during validation
    with torch.no_grad():
        val_loss = 0
        val_acc = 0
        for val_images, val_labels in val_loader:
            val_output = cnn_model(val_images)
            val_loss += criterion(val_output, val_labels).item()
            val_acc += get_accuracy(val_output, val_labels)
            
        cnn_val_loss_history.append(val_loss / len(val_loader))
        cnn_val_acc_history.append(val_acc / len(val_loader))
        
    print(f"Epoch {epoch+1} done in {end_time - start_time:.2f} seconds. "
          f"Train Loss: {cnn_train_loss_history[-1]:.4f}, Train Acc: {cnn_train_acc_history[-1]:.4f}, "
          f"Val Loss: {cnn_val_loss_history[-1]:.4f}, Val Acc: {cnn_val_acc_history[-1]:.4f}")

# Test the model on the test set
cnn_model.eval()  # Set model to evaluation mode
test_loss = 0
test_acc = 0
with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_output = cnn_model(test_images)
        test_loss += criterion(test_output, test_labels).item()
        test_acc += get_accuracy(test_output, test_labels)
print(f"\nTest Loss: {test_loss / len(test_loader):.4f}, Test Acc: {test_acc / len(test_loader):.4f}\n")

# Plot training and validation loss/accuracy curves
plt.figure(figsize=(6, 8))

plt.subplot(2, 1, 1)
plt.plot(cnn_train_loss_history, label='Train Loss',color='k',alpha=0.6)
plt.plot(cnn_val_loss_history, label='Validation Loss',color='darkred',alpha=0.6)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(frameon=False)
plt.title('CNN Training and Validation Loss')

plt.subplot(2, 1, 2)
plt.plot(cnn_train_acc_history, label='Train Accuracy',color='k',alpha=0.6)
plt.plot(cnn_val_acc_history, label='Validation Accuracy',color='darkred',alpha=0.6)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(frameon=False)
plt.title('CNN Training and Validation Accuracy')
plt.tight_layout()
plt.savefig("./ca1/img/architecture_perfomance.png",bbox_inches='tight',dpi=300)

#########################
# HYPERPARAMETER TUNING #
#########################
print("\n***IMPLEMENT HYPERPARAMETER TUNING***")
import copy
import random

# 1. Expanded Parameter Space (Now including architectural depth!)
starting_lrs = [0.005, 0.001, 0.0005, 0.0001]
dropout_rates = [0.3, 0.4, 0.5, 0.6, 0.7]         
weight_decays = [1e-3, 1e-4, 1e-5, 0.0]  
kernel_configs = [(5, 2), (3, 1)]        
batch_sizes = [32, 64, 128]
dense_units = [64, 128, 256]             
conv_layers = [2, 3]  # <--- NEW: Test 2 blocks vs 3 blocks!

best_val_acc = 0.0
best_params = {}

tuning_epochs = 10 
num_random_trials = 60 

print(f"Starting Random Search: Testing {num_random_trials} combinations...\n")

for trial in range(num_random_trials):
    lr = random.choice(starting_lrs)
    drop_rate = random.choice(dropout_rates)
    wd = random.choice(weight_decays)
    k_size, k_pad = random.choice(kernel_configs)
    b_size = random.choice(batch_sizes)
    d_units = random.choice(dense_units)
    n_layers = random.choice(conv_layers) # Pick 2 or 3 layers
    
    print(f"--- Trial {trial+1:02d}/{num_random_trials} | Layers: {n_layers} | LR: {lr} | Drop: {drop_rate} | Ker: {k_size} | Batch: {b_size} | Dense: {d_units} ---")
    
    tune_train_loader = DataLoader(train_data, batch_size=b_size, shuffle=True)
    tune_val_loader = DataLoader(val_data, batch_size=b_size, shuffle=False)

    # 2. Dynamically build the model architecture
    layers = []
    
    # Block 1
    layers.extend([
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=k_size, padding=k_pad), 
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    ])
    
    # Block 2
    layers.extend([
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    ])
    
    # Block 3 (Optional!)
    if n_layers == 3:
        layers.extend([
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
            # Notice: No MaxPool2d here to avoid shrinking the image too much!
        ])
        flatten_size = 128 * 7 * 7
    else:
        flatten_size = 64 * 7 * 7
        
    # Classifier
    layers.extend([
        nn.Flatten(),                
        nn.Dropout(drop_rate), 
        nn.Linear(flatten_size, d_units),
        nn.ReLU(),
        nn.Linear(d_units, 10)
    ])
    
    # Unpack the list into a Sequential model using the * operator
    model = nn.Sequential(*layers)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()
    
    current_val_acc = 0
    
    for epoch in range(tuning_epochs):
        model.train()
        for images, labels in tune_train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for val_images, val_labels in tune_val_loader:
                val_output = model(val_images)
                val_loss += criterion(val_output, val_labels).item()
                val_acc += get_accuracy(val_output, val_labels)
        
        val_loss /= len(tune_val_loader)
        current_val_acc = val_acc / len(tune_val_loader)
        scheduler.step(val_loss)
    
    print(f"Result -> Final Val Accuracy: {current_val_acc:.4f}\n")
    
    if current_val_acc > best_val_acc:
        best_val_acc = current_val_acc
        best_params = {
            'num_layers': n_layers, 'lr': lr, 'dropout': drop_rate, 'weight_decay': wd, 
            'kernel_size': k_size, 'padding': k_pad, 
            'batch_size': b_size, 'dense_units': d_units
        }

print("======================================")
print(f"🥇 RANDOM SEARCH COMPLETE! Best Accuracy: {best_val_acc:.4f}")
print(f"🥇 Best Parameters: {best_params}")
print("======================================")

# 6. Rebuild the WINNING model architecture dynamically
final_layers = []
final_layers.extend([
    nn.Conv2d(1, 32, kernel_size=best_params['kernel_size'], padding=best_params['padding']), 
    nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
    
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
])

if best_params['num_layers'] == 3:
    final_layers.extend([
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128), nn.ReLU()
    ])
    final_flatten_size = 128 * 7 * 7
else:
    final_flatten_size = 64 * 7 * 7
    
final_layers.extend([
    nn.Flatten(),
    nn.Dropout(best_params['dropout']),
    nn.Linear(final_flatten_size, best_params['dense_units']),
    nn.ReLU(),
    nn.Linear(best_params['dense_units'], 10)
])

final_tuned_model = nn.Sequential(*final_layers)

final_train_loader = DataLoader(train_data, batch_size=best_params['batch_size'], shuffle=True)
final_val_loader = DataLoader(val_data, batch_size=best_params['batch_size'], shuffle=False)
final_test_loader = DataLoader(test_data, batch_size=best_params['batch_size'], shuffle=False)

# 7. Train the winning model for a FULL run to reach maximum potential
print(f"\n*** TRAINING FINAL WINNING MODEL ({best_params['num_layers']} LAYERS) FOR 30 EPOCHS ***")
final_optimizer = torch.optim.Adam(final_tuned_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
final_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, mode='min', factor=0.5, patience=2)

final_train_loss_hist, final_val_loss_hist = [], []
final_train_acc_hist, final_val_acc_hist = [], []

final_epochs = 30
for epoch in range(final_epochs):
    final_tuned_model.train()  
    t_loss, t_acc, start_time = 0, 0, time.time()
    
    for images, labels in final_train_loader:
        final_optimizer.zero_grad()
        output = final_tuned_model(images)
        loss = criterion(output, labels)
        loss.backward()
        final_optimizer.step()
        t_loss += loss.item()
        t_acc += get_accuracy(output, labels)
        
    final_train_loss_hist.append(t_loss / len(final_train_loader))
    final_train_acc_hist.append(t_acc / len(final_train_loader))

    final_tuned_model.eval()  
    with torch.no_grad():
        v_loss, v_acc = 0, 0
        for val_images, val_labels in final_val_loader:
            val_output = final_tuned_model(val_images)
            v_loss += criterion(val_output, val_labels).item()
            v_acc += get_accuracy(val_output, val_labels)
            
        final_val_loss_hist.append(v_loss / len(final_val_loader))
        final_val_acc_hist.append(v_acc / len(final_val_loader))
        
    final_scheduler.step(final_val_loss_hist[-1])
    print(f"Final Model Epoch {epoch+1:02d}/{final_epochs} ({time.time() - start_time:.1f}s) | Train Acc: {final_train_acc_hist[-1]:.4f} | Val Acc: {final_val_acc_hist[-1]:.4f}")

# 8. Plot the Final Winning Model's Learning Curves
plt.figure(figsize=(6, 8))

plt.subplot(2, 1, 1)
plt.plot(final_train_loss_hist, label='Train Loss', color='k', alpha=0.6)
plt.plot(final_val_loss_hist, label='Validation Loss', color='darkred', alpha=0.6)
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(frameon=False)
plt.title(f"Final Model Loss ({best_params['num_layers']} Layers, LR: {best_params['lr']})")

plt.subplot(2, 1, 2)
plt.plot(final_train_acc_hist, label='Train Accuracy', color='k', alpha=0.6)
plt.plot(final_val_acc_hist, label='Validation Accuracy', color='darkred', alpha=0.6)
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(frameon=False)
plt.title(f"Final Model Accuracy (Drop: {best_params['dropout']}, WD: {best_params['weight_decay']})")

plt.tight_layout()
plt.savefig("./ca1/img/final_performance.png", bbox_inches='tight', dpi=300)
plt.close()

#########################
# CLASSIFICATION REPORT #
#########################
print("\n***CLASSIFICATION REPORT***")
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

final_tuned_model.eval()  
test_loss = 0
test_acc = 0

all_predictions = []
all_true_labels = []

with torch.no_grad():
    for test_images, test_labels in final_test_loader:
        test_output = final_tuned_model(test_images)
        test_loss += criterion(test_output, test_labels).item()
        test_acc += get_accuracy(test_output, test_labels)
        
        _, predictions = torch.max(test_output, 1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_true_labels.extend(test_labels.cpu().numpy())

final_test_loss = test_loss / len(final_test_loader)
final_test_acc = test_acc / len(final_test_loader)

print(f"Final Test Loss: {final_test_loss:.4f}, Final Test Acc: {final_test_acc:.4f}\n")

fashion_mnist_classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print("======================================")
print("     FINAL TEST SET EVALUATION        ")
print("======================================")

# 1. Generate and print the Classification Report
report_str = classification_report(all_true_labels, all_predictions, target_names=fashion_mnist_classes)
print(report_str)

# 2. Plot and save the Confusion Matrix
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(all_true_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=fashion_mnist_classes)

# Create a figure large enough to cleanly display the labels
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)

plt.title('Final Model Confusion Matrix')
plt.tight_layout()
plt.savefig("./ca1/img/confusion_matrix.png", bbox_inches='tight', dpi=300)
plt.close()

# 3. Save all text results to the tuning_results.txt file
with open('./ca1/data/tuning_results.txt', 'w') as file:
    file.write("=== BEST HYPERPARAMETERS ===\n")
    for key, value in best_params.items():
        file.write(f"{key}: {value}\n")
    file.write(f"\nFinal Test Loss: {final_test_loss:.4f}\n")
    file.write(f"Final Test Accuracy: {final_test_acc:.4f}\n\n")
    file.write("=== CLASSIFICATION REPORT ===\n")
    file.write(report_str)
