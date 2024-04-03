import os
import argparse
from torchvision.utils import save_image, make_grid
from torchvision.io import read_image
from PIL import Image
import numpy as np
import torch
import dnnlib
import legacy
from tqdm import tqdm
import re

def generate_image(G, dataset_folder, img_name, out_folder, device=torch.device('cuda')):
    os.makedirs(out_folder, exist_ok=True)
    w_loc = os.path.join(dataset_folder, f'{img_name}_latent.npy')
    c_loc = os.path.join(dataset_folder, f'{img_name}.npy')

    # Generate image.
    ws = torch.from_numpy(np.load(w_loc)).to(device)
    c = torch.from_numpy(np.load(c_loc).reshape(1,25)).to(device)
    
    img = G.synthesis(ws, c, noise_mode='const')['image'].detach().cpu()

    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    Image.fromarray(img, 'RGB').save(f'{out_folder}/{img_name}.png')

def compile_images(out_folder):
    years = [year for year in os.listdir(out_folder) if year.isdigit()]
    for t in years:
        # make a grid of images in the folder
        imgs = []
        for img_path in os.listdir(os.path.join(out_folder, str(t))):
            img = read_image(os.path.join(out_folder, str(t), img_path)).to(torch.float)
            imgs.append(img)
        grid = make_grid(imgs, nrow=10, normalize=True, range=(-1, 1))
        save_image(grid, os.path.join(out_folder, f'{t}.png'))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--celeb', type=str, help='Celeb', required=True)
    args = parser.parse_args()
    celeb = args.celeb
    model = 'eg3d-finetune'

    try:
        network_pkl = '/playpen-nas-ssd/awang/eg3d/eg3d/training-runs/Margot_t0/00050-ffhq-preprocessed-gpus4-batch16-gamma20/network-snapshot-000500.pkl'
        device = torch.device('cuda')
        
        years = [year for year in os.listdir(f'/playpen-nas-ssd/awang/data/{celeb}') if year.isdigit()]
        for t in years:
            with dnnlib.util.open_url(network_pkl) as f:
                G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

            folder_path = f'/playpen-nas-ssd/awang/data/{celeb}/{t}/train/preprocessed'
            out_folder = os.path.join('out', 'sanity_check_training_reconstructions', celeb, model, str(t))
            for filename in tqdm([x for x in os.listdir(folder_path) if x.endswith('.png')]):
                img_name = filename.replace('.png', '')
                generate_image(G, folder_path, img_name, out_folder)
        out_folder = os.path.join('out', 'sanity_check_training_reconstructions', celeb, model)
        compile_images(out_folder)
    except KeyboardInterrupt:
        print('Interrupted!')
        exit(0)