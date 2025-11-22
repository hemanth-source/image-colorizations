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
    img = Image.open(input_path).convert("L")   # <-- FIXED
    grey = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(grey)[0].permute(1, 2, 0).cpu().numpy()

    out_img = (output * 255).astype("uint8")
    Image.fromarray(out_img).save("output.png")
    print("Saved output.png")


# Run the function
colorize("0009.png")
