import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score

# ======================
# Path and class setup
# ======================

# Path to the dataset
base_path = r"D:\ai_data\ai_data_advanced\week03\data\ BananaLSD\OriginalSet"

# Class labels for banana leaf disease classification
classes = ["cordana", "healthy", "pestalotiopsis", "sigatoka"]

# ======================
# Display sample images
# ======================

for class_name in classes:
    class_dir = os.path.join(base_path, class_name)
    images = os.listdir(class_dir)[:2]  # Take first 2 images from each class

    for img_name in images:
        img_path = os.path.join(class_dir, img_name)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')
        plt.show()

# ======================
# Data transformations
# ======================
# Train transform: includes augmentation to improve generalization
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ColorJitter(brightness=0.1, contrast=0.3),  # Random brightness/contrast changes
    transforms.RandomHorizontalFlip(),                     # Random horizontal flips
    transforms.ToTensor(),                                 # Convert to tensor
    transforms.Normalize(                                   # Normalize to ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])

# Validation transform: no augmentation, just resizing & normalization
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ======================
# Dataset & DataLoader
# ======================
# Load dataset
dataset = datasets.ImageFolder(root=base_path, transform=train_transform)

# Split into train (80%) and validation (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Apply validation transforms to validation dataset
val_dataset.dataset.transform = val_transform

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

# ======================
# Utility functions
# ======================

# Function to denormalize and display an image
def img_show(img_tensor):
    img = img_tensor.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):  # Undo normalization
        img[:, :, i] = img[:, :, i] * std[i] + mean[i]
    img = img.clip(0, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Show augmented images for inspection
def augmented_images(dataset, num_images=5):
    for i in range(num_images):
        img, label = dataset[i]
        plt.title(dataset.classes[label])
        img_show(img)

augmented_images(dataset, 5)

# ======================
# Model Definition
# ======================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        # Convolutional layer 1: input channels=3 (RGB), output channels=16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Reduces spatial size by half
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # 32 channels, 32x32 features
        self.fc2 = nn.Linear(128, num_classes)   # Output layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pool
        x = x.view(-1, 32 * 32 * 32)          # Flatten
        x = F.relu(self.fc1(x))               # Fully connected layer
        x = self.fc2(x)                       # Output logits
        return x

# ======================
# Training Setup
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()  # Classification loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ======================
# Training Loop
# ======================
for epoch in range(4):  # Number of epochs
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # ======================
    # Validation phase
    # ======================
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute precision and recall for validation
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    print(f"Epoch {epoch + 1} - Running loss: {running_loss:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f}")

    # Plot metrics
    plt.bar(['precision', 'recall'], [precision, recall])
    plt.ylim(0, 1)
    plt.show()
