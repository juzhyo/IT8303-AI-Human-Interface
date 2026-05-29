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
epochs = 10
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
print("***IMPLEMENT MODEL IMPROVEMENTS***")

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
epochs = 20
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
print("\n***IMPLEMENT HYPERPARAMETER TUNING (Grid Search + LR Scheduler)***")
import copy

# 1. Define the parameters to test in our Grid Search
# Keeping it to 4 combinations (2x2) so it doesn't take hours on a CPU
dropout_rates = [0.4, 0.6]         
weight_decays = [1e-4, 1e-5]       

best_val_acc = 0.0
best_params = {}
best_model_state = None  # This will store the winning model's exact weights

# We'll use 8 epochs per test. (Increase this if you ever use a GPU!)
tuning_epochs = 8 

for drop_rate in dropout_rates:
    for wd in weight_decays:
        print(f"\n--- Testing Dropout: {drop_rate} | Weight Decay: {wd} ---")
        
        # 2. Initialize a fresh model for this specific loop
        model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Flatten(),               
            nn.Dropout(drop_rate), # <--- Inject the loop's dropout rate
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)          
        )
        
        # 3. Setup Optimizer with loop's weight decay, and attach the LR Scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        criterion = nn.CrossEntropyLoss()
        
        current_val_acc = 0
        
        # 4. Short training loop
        for epoch in range(tuning_epochs):
            model.train()
            for images, labels in train_loader:
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_acc = 0
            with torch.no_grad():
                for val_images, val_labels in val_loader:
                    val_output = model(val_images)
                    val_loss += criterion(val_output, val_labels).item()
                    val_acc += get_accuracy(val_output, val_labels)
            
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            current_val_acc = val_acc 
            
            # Step the scheduler dynamically based on validation loss
            scheduler.step(val_loss)
        
        print(f"Result -> Final Val Accuracy: {current_val_acc:.4f}")
        
        # 5. Save the model if it beat the previous high score!
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            best_params = {'dropout': drop_rate, 'weight_decay': wd}
            # Deepcopy saves the actual weights, not just the reference
            best_model_state = copy.deepcopy(model.state_dict())

print("\n======================================")
print(f"🥇 TUNING COMPLETE! Best Accuracy: {best_val_acc:.4f}")
print(f"🥇 Best Parameters: {best_params}")
print("======================================")

# 6. Rebuild the winning model architecture using the best parameters
final_tuned_model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2), 
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Flatten(),               
    nn.Dropout(best_params['dropout']), # Use the winning dropout
    nn.Linear(64 * 7 * 7, 128),
    nn.ReLU(),
    nn.Linear(128, 10)          
)

# 7. Load the winning "brain" (weights) into the model
final_tuned_model.load_state_dict(best_model_state)

#########################
# CLASSIFICATION REPORT #
#########################
print("\n***IMPLEMENT CLASSIFICATION REPORT***")
from sklearn.metrics import classification_report

# Test the fully tuned, absolute best model on the untouched test set
final_tuned_model.eval()  
test_loss = 0
test_acc = 0

all_predictions = []
all_true_labels = []

with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_output = final_tuned_model(test_images)
        test_loss += criterion(test_output, test_labels).item()
        test_acc += get_accuracy(test_output, test_labels)
        
        _, predictions = torch.max(test_output, 1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_true_labels.extend(test_labels.cpu().numpy())

print(f"Final Test Loss: {test_loss / len(test_loader):.4f}, Final Test Acc: {test_acc / len(test_loader):.4f}\n")

fashion_mnist_classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print("======================================")
print("     FINAL TEST SET EVALUATION        ")
print("======================================")
print(classification_report(all_true_labels, all_predictions, target_names=fashion_mnist_classes))

print("\n✅ Script execution completely finished!")
