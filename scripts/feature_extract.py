import torch
from PIL import Image
import torchvision.transforms as T
import os 
# import hubconf
from tqdm import tqdm
import shutil
import numpy as np

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=str, required=True)
    parser.add_argument('--rate', type=int, default=4)
    parser.add_argument('--H', type=int, default=3024)
    parser.add_argument('--W', type=int, default=4032)

    args = parser.parse_args()
    base_path = f"./Datasets/on-the-go/{args.seq}"

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2_vits14.to(device)
    extractor = dinov2_vits14

    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406) 
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225) 
    RATE = args.rate
    RESIZE_H = (args.H // RATE) // 14 * 14
    RESIZE_W = (args.W // RATE) // 14 * 14

    if os.path.exists(os.path.join(base_path, f'features_{RATE}')):
        shutil.rmtree(os.path.join(base_path, f'features_{RATE}'))
    folder = os.path.join(base_path, 'images')
    files = os.listdir(folder)
    files = [os.path.join(folder, f) for f in files]
    features = []
    for f in tqdm(files):
        img = Image.open(f).convert('RGB')
        transform = T.Compose([
            T.Resize((RESIZE_H, RESIZE_W)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])
        img = transform(img)[:3].unsqueeze(0)
        with torch.no_grad():
            features_dict = extractor.forward_features(img.cuda())
            features = features_dict['x_norm_patchtokens'].view(RESIZE_H // 14, RESIZE_W // 14, -1)
        img_type = f[-4:]
        save_path = f.replace(f'{img_type}', '.npy').replace('/images/', f'/features_{RATE}/')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, features.detach().cpu().numpy())
