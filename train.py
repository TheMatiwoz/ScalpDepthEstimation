import torch
import torch.nn as nn
import torchvision.transforms as transforms
import models
from tools.inverse_warping import warp

# Device configuration
device = torch.device('cuda' if torch.cuda.is_avalible() else 'cpu')


# Hyper-parameters
num_epochs = 10
batch_size = 4
f = 1  # ???
B = 1  # ???

# Dataset - in progress
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# Model, loss, optimizer
model = models.UNet(3).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

warping = models.WarpingLayer(f, B)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images_left, images_right) in enumerate(train_loader):

        images_left = images_left.to(device)
        images_right = images_right.to(device)

        # Forward pass
        outputs = model(images_left)
        warped = warping(images_right, outputs)
        loss = criterion(images_left, warped)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

