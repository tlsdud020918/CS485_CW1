import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from scipy.linalg import svd
import math

# Custom Dataset Class
class Caltech101Dataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.data = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        
        for label, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name)
            images = sorted(os.listdir(class_folder))
            if self.train:
                images = images[:15]  # First 15 images for training
            else:
                images = images[15:31]  # Last 15 images for testing
            
            for img_name in images:
                img_path = os.path.join(class_folder, img_name)
                self.data.append(img_path)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define Dataset Paths and Transformations
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, "../RF_code/Caltech_101")
transform = transforms.Compose([
    transforms.Resize((128,128)),  # Resize images
    transforms.ToTensor(),         # Convert to Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Create Training and Testing Datasets
train_dataset = Caltech101Dataset(root_dir, transform=transform, train=True)
test_dataset = Caltech101Dataset(root_dir, transform=transform, train=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the CNN Architecture
class CustomCNN(nn.Module):
    def __init__(self, total_layers=4, dropout=True, conv_kernel_size = 3, normalization="batch", skip_connection = 1):
        super(CustomCNN, self).__init__()
        # self.compress_rank = compress_rank
        self.total_layers = total_layers
        self.dropout = dropout
        self.conv_kernel_size = conv_kernel_size
        self.skip_connection = skip_connection

        def get_normalization(num_features, input_size):
            if normalization == "batch":
                return nn.BatchNorm2d(num_features)
            elif normalization == "layer":
                return nn.LayerNorm([num_features, input_size, input_size])  # Channel-first input
            elif normalization == "group":
                return nn.GroupNorm(4, num_features)  # 4 groups as an example
            elif normalization == "instance":
                return nn.InstanceNorm2d(num_features)
            else:
                raise ValueError(f"Unsupported normalization type: {normalization}")
            
        self.conv_blocks = nn.ModuleList()
        in_channels = 3
        out_channels_list = [32, 64, 128, 256, 512][:total_layers]

        input_size = 128
        for out_channels in out_channels_list:
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size, stride=1, padding=1),
                get_normalization(out_channels, input_size),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.conv_blocks.append(block)
            in_channels = out_channels
            input_size = math.floor((input_size - conv_kernel_size + 2 * 1) / 1 + 1)  # Conv2D
            input_size = math.floor(input_size / 2)

        self.channel_matchers = nn.ModuleList([
            nn.Conv2d(3 if i == 0 else out_channels_list[i-1], out_channels_list[i], kernel_size=1)
            for i in range(len(out_channels_list))
        ])

        fc_input_size = out_channels_list[-1] * (input_size ** 2)

        fc_blocks = [
            nn.Linear(fc_input_size, 512),
            nn.ReLU()
        ]
        if self.dropout:
            fc_blocks.append(nn.Dropout(0.2))
        fc_blocks.extend([
            nn.Linear(512, 128),
            nn.ReLU()
        ])
        if self.dropout:
            fc_blocks.append(nn.Dropout(0.2))
        fc_blocks.extend([
            nn.Linear(128, 32),
            nn.ReLU()
        ])
        if self.dropout:
            fc_blocks.append(nn.Dropout(0.1))
        fc_blocks.append(nn.Linear(32, len(train_dataset.classes)))

        self.fc_layers = nn.Sequential(*fc_blocks)
        
    def forward(self, x):
        residual = x 

        for i, block in enumerate(self.conv_blocks):
            x = block(x) # if self.skip_connection == 0, do nothing with residual
            if self.skip_connection == 1: # connect every layers
                residual = nn.functional.interpolate(residual, size=x.size()[2:], mode='bilinear', align_corners=False)  
                residual = self.channel_matchers[i](residual)
                x += residual
                residual = x
            elif self.skip_connection == 2: # connect every 2 layers
                residual = nn.functional.interpolate(residual, size=x.size()[2:], mode='bilinear', align_corners=False)  
                residual = self.channel_matchers[i](residual)
                if i % 2 == 1:  # Add skip connection every 2 blocks
                    x += residual  # Add residual input to the current output
                residual = x  

        x = x.view(x.size(0), -1)  # Flatten before fully connected layers
        x = self.fc_layers(x)

        return x
    
    def compress_fc_layers(self, compress_ratio):
        for i, layer in enumerate(self.fc_layers):
            if isinstance(layer, nn.Linear):
                rank = int(torch.linalg.matrix_rank(layer.weight.data))
                if rank>=128:
                    compress_rank = int(rank * compress_ratio)
                    self.fc_layers[i] = truncated_svd(layer, compress_rank)
    
class SquaredHingeLoss(nn.Module):
    def __init__(self):
        super(SquaredHingeLoss, self).__init__()

    def forward(self, outputs, targets):
        # Convert integer targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=outputs.size(1)).float()
        targets_one_hot = 2 * targets_one_hot - 1  # Convert to +1/-1
        
        # Compute hinge loss
        hinge_loss = torch.clamp(1 - outputs * targets_one_hot, min=0)  # Hinge loss
        squared_hinge_loss = hinge_loss ** 2  # Square the hinge loss
        
        # Take mean across classes and batch
        return torch.mean(squared_hinge_loss)
    
def truncated_svd(layer, rank):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = layer.weight.data.cpu().numpy()  
    print(np.linalg.matrix_rank(W))
    B = layer.bias.data.cpu().numpy()
    U, S, Vh = np.linalg.svd(W, full_matrices=False) # w/o full_matrices=False -> too much computing time

    # Truncate SVD
    U_k = U[:, :rank] 
    S_k = np.diag(S[:rank])
    V_k = Vh[:rank, :]

    compressed_W = U_k@S_k@V_k

    in_features = compressed_W.shape[1]
    out_features = compressed_W.shape[0]
    linear_layer = nn.Linear(in_features, out_features)

    weight = torch.tensor(U_k@S_k@V_k, dtype=torch.float32).to(layer.weight.device)
    bias = torch.tensor(B, dtype=torch.float32).to(layer.weight.device)

    with torch.no_grad():
        linear_layer.weight.data = weight
        linear_layer.bias.data = bias

    return linear_layer

def train_model(model, train_loader, optimizer, criterion=nn.CrossEntropyLoss(), num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
    return train_losses

# Evaluate the CNN
def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    model = CustomCNN()

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()  # Default softmax-based loss
    # criterion = SquaredHingeLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Train the CNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Run Training and Evaluation
    train_losses = train_model(model, train_loader, optimizer, criterion=nn.CrossEntropyLoss(), num_epochs=100)
    # model.compress_fc_layers()
    test_accuracy = evaluate_model(model, test_loader)

    # Visualize Training Loss
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()
