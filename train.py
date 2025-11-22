import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset_cg import ColorizationDataset
from models import UNetGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
train_dataset = ColorizationDataset(root_dir="data", split='train')
val_dataset = ColorizationDataset(root_dir="data", split='val')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

model = UNetGenerator().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)

EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for grey, color in train_loader:
        grey, color = grey.to(device), color.to(device)

        optimizer.zero_grad()
        output = model(grey)
        loss = criterion(output, color)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "colorization_unet.pth")
print("Model saved!")
