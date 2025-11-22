# inference.py
import torch
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms
from models import UNetGenerator

def colorize(input_path, output_path, ckpt='checkpoints/gen_epoch_9.pth', size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor()])
    img = Image.open(input_path).convert('RGB')
    img_t = transform(img)
    img_np = (img_t.numpy().transpose(1,2,0) * 255).astype(np.uint8)
    lab = rgb2lab(img_np).astype(np.float32)
    L = lab[:,:,0] / 100.0
    import torch
    L_t = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).float().to(device)
    gen = UNetGenerator().to(device)
    gen.load_state_dict(torch.load(ckpt, map_location=device))
    gen.eval()
    with torch.no_grad():
        pred_ab = gen(L_t).cpu().squeeze(0).numpy().transpose(1,2,0) * 128
        L_denorm = (L * 100)[:,:,None]
        lab_out = np.concatenate([L_denorm, pred_ab], axis=2)
        rgb = lab2rgb(lab_out.astype(np.float64))
        # convert to PIL
        from PIL import Image
        out = Image.fromarray((rgb*255).astype(np.uint8))
        out.save(output_path)
        print("Saved", output_path)

if __name__ == '__main__':
    import sys
    colorize(sys.argv[1], sys.argv[2], ckpt=sys.argv[3] if len(sys.argv)>3 else 'checkpoints/gen_epoch_9.pth')
