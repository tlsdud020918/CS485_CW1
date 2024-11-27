from transformers import AutoImageProcessor, ResNetForImageClassification, ResNetConfig
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import get_scheduler
from tqdm import tqdm
from PIL import Image

from cnn import Caltech101Dataset



class ResNetCaltech101Dataset(Caltech101Dataset):
    def __getitem__(self, idx):
        image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", do_rescale=False)
        img_path = self.data[idx]
        label = self.labels[idx]
        image = image_processor(Image.open(img_path).convert("RGB"), return_tensors="pt")
        image = image['pixel_values'].squeeze(0)
        
        return image, label


# preparing data
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, "../RF_code/Caltech_101")


train_dataset = ResNetCaltech101Dataset(root_dir, transform=None, train=True)
labels = list(set(train_dataset.labels))
num_labels = len(labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# # load pre-trained resnet model
# model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
# load random initialization model
config = ResNetConfig()
model = ResNetForImageClassification(config)

model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(model.config.hidden_sizes[-1], num_labels)
)

# Freeze the backbone (ResNet layers) to train only the classifier
for param in model.resnet.parameters():
    param.requires_grad = False

# setting GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005)

num_epochs = 10
num_training_steps = len(train_loader) * num_epochs
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

if __name__ == "__main__":
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # forward
            outputs = model(pixel_values=inputs)
            loss = loss_fn(outputs.logits, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss :{avg_train_loss: .4f}")

    model.config.num_labels = 10
    # model.save_pretrained("fine_tuned_resnet")
    model.save_pretrained("random_init_resnet")
