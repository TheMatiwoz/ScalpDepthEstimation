import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize

from model import models
from dataset import SkalpDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 10
batch_size = 1
f = 1  # ???
B = 1  # ???

# Dataset - in progress
transforms = Compose([ToTensor(),
                      Resize((320, 512))
                      ])
dataset = SkalpDataset(r"/content/DepthEstimation/training_data/bag_1/_start_saki4",
                       r"/content/DepthEstimation/training_data/bag_1/_start_saki4/selected_indexes", transforms)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# Model, loss, optimizer
model = models.UNet(3).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

warping = models.WarpingLayer(f, B)

total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)
print(total_samples, n_iterations)
n_total_steps = len(dataloader)

for epoch in range(num_epochs):
    for i, (images_left, images_right) in enumerate(dataloader):

        with torch.no_grad():
            images_left = images_left.to(device)
            images_right = images_right.to(device)

        # print(images_left, images_right)

        # Forward pass
        outputs = model(images_left)
        print(outputs)
        warped = warping((images_right, outputs))
        print(images_left, warped)
        loss = criterion(images_left, warped)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
