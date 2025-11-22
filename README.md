Image Colorization using U-Net

This project performs automatic image colorization using a U-Net–based neural network.
A grayscale image is given as input, and the model generates a colorized version.

Features

U-Net generator model

Train on your own dataset

Convert grayscale images to color

Simple and lightweight training script

Easy inference script (colorize.py)

Project Structure
├── train.py               # Training script
├── colorize.py            # Colorization / inference script
├── models.py              # U-Net model
├── requirements.txt       # Python dependencies
├── dataset/               # Your training images (grayscale + color)
├── colorization_unet.pth  # Trained model file (after training)
└── output.png             # Example output

Installation

Install Python packages:

pip install -r requirements.txt


Make sure PyTorch is installed for your system:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


(Replace CUDA version if needed.)

Training

Place your training images inside a folder such as:

dataset/train/
dataset/val/


Run training:

python train.py


After training finishes, a model file is saved:

colorization_unet.pth

Usage (Colorizing an Image)

Place any grayscale image (e.g., 0009.png) in the project directory.

Run:

python colorize.py


The output will be saved as:

output.png

colorize.py Example
import torch
from PIL import Image
from torchvision import transforms
from models import UNetGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

model = UNetGenerator().to(device)
model.load_state_dict(torch.load("colorization_unet.pth", map_location=device))
model.eval()

def colorize(input_path):
    img = Image.open(input_path).convert("L")
    grey = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(grey)[0].permute(1, 2, 0).cpu().numpy()

    out_img = (output * 255).astype("uint8")
    Image.fromarray(out_img).save("output.png")
    print("Saved output.png")

colorize("0009.png")
