import os
from transformers import ResNetForImageClassification
import torch
from torch.utils.data import DataLoader

from ResNet_train import ResNetCaltech101Dataset

# preparing data
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, "../RF_code/Caltech_101")


test_dataset = ResNetCaltech101Dataset(root_dir, transform=None)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = ResNetForImageClassification.from_pretrained("random_init_resnet", ignore_mismatched_sizes=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(pixel_values=inputs)
        _, predicted = torch.max(outputs.logits, dim=-1)
        
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy: .4f}")