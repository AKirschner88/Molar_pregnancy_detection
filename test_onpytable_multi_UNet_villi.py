import os
import shutil
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from UNet_villi import *

import tables
from skimage import color


def blend2Images(img1, img2):
    if img1.ndim == 3:
        img1 = color.rgb2gray(img1)
    if img2.ndim == 3:
        img2 = color.rgb2gray(img2)
    img1 = img1[:, :, None] * 1.0
    img2 = img2[:, :, None] * 1.0
    out = np.concatenate((img2, img1, img2), 2) * 255
    return out.astype('uint8')

def blend2Images_threeclass(img1, img2):
    out = np.zeros((img1.shape[0], img1.shape[1], 3))
    out[(img1 == 0)*(img2 == 0)] = [0, 0, 0]
    out[(img1 == 1)*(img2 == 1)] = [255, 255, 255]
    out[(img1 == 2)*(img2 == 2)] = [255, 255, 255]

    out[(img1 == 0)*(img2 == 1)] = [255, 0, 0]
    out[(img1 == 0)*(img2 == 2)] = [0, 255, 0]

    out[(img1 == 1)*(img2 == 0)] = [0, 0, 255]
    out[(img1 == 1)*(img2 == 2)] = [255, 255, 0]

    out[(img1 == 2)*(img2 == 0)] = [0, 255, 255]
    out[(img1 == 2)*(img2 == 1)] = [255, 0, 255]
    return out/255

def visualize_two_class(img, mask, unet_out, unet_out_class):
    """Visualization for 2-class UNet: classes {0,1}."""
    probs = torch.softmax(torch.from_numpy(unet_out), dim=0).numpy()

    fig, ax = plt.subplots(2, 3, num=100, figsize=(10, 5))

    ax[0, 0].imshow(img)
    ax[0, 1].imshow(mask, cmap='plasma', vmin=0, vmax=1)
    ax[0, 2].imshow(unet_out_class, cmap='plasma', vmin=0, vmax=1)
    ax[1, 0].imshow(probs[1], cmap='viridis', vmin=0, vmax=1)
    ax[1, 1].imshow(blend2Images(unet_out_class == 1, mask == 1))

    plt.tight_layout()
    plt.show()


def visualize_three_class(img, mask, unet_out, unet_out_class):
    """Visualization for 3-class UNet: classes {0,1,2}."""
    probs = torch.softmax(torch.from_numpy(unet_out), dim=0).numpy()

    fig, ax = plt.subplots(2, 4, num=100, figsize=(10, 5))

    ax[0, 0].imshow(img)
    ax[0, 1].imshow(mask, cmap='plasma', vmin=0, vmax=2)
    ax[0, 2].imshow(unet_out_class, cmap='plasma', vmin=0, vmax=2)
    ax[0, 3].imshow(probs[1], cmap='viridis', vmin=0, vmax=1)

    ax[1, 0].imshow(probs[2], cmap='viridis', vmin=0, vmax=1)
    ax[1, 1].imshow(blend2Images(unet_out_class == 1, mask == 1))
    ax[1, 2].imshow(blend2Images(unet_out_class == 2, mask == 2))
    ax[1, 3].imshow(blend2Images_threeclass(unet_out_class, mask))

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    # where to evaluate
    table_name = 'E:\Digital_Pathology\Project\Aron_to_share\Datasets\AKV7_9_data_villi_multi_test.pytable'
    #table_name = './data_villi_multi_train.pytable'

    # load model
    model_name  = "AKV7_9_E100_data_villi_multi_best_model_multi_UNet"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model_path = fr'E:\Digital_Pathology\Project\Aron_to_share\Models\{model_name }.pth'
    checkpoint = torch.load(model_path, map_location=device)
    model = UNet(n_classes=checkpoint["n_classes"], in_channels=checkpoint["in_channels"], padding=checkpoint["padding"], depth=checkpoint["depth"], wf=checkpoint["wf"], up_mode=checkpoint["up_mode"], batch_norm=checkpoint["batch_norm"]).to(device)
    print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")
    model.load_state_dict(checkpoint["model_dict"])
    model.eval()

    # prepare folders to save data
    path_save_data = fr'E:\Digital_Pathology\Project\Aron_to_share\UNet_villi_multi_output_pytable\{model_name }'
    os.makedirs(path_save_data, exist_ok=True)
    # shutil.rmtree(path_save_data) # uncomment both line  if you want it to delete 
    # os.mkdir(path_save_data)      # the folder everytime you run the code
    os.makedirs(os.path.join(path_save_data, 'tile'), exist_ok=True)
    os.makedirs(os.path.join(path_save_data, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(path_save_data, 'output'), exist_ok=True)

    with tables.open_file(table_name, 'r') as db:
        images = db.root.img
        masks = db.root.mask

        for i in tqdm(range(len(images))):
            img = images[i]
            mask = masks[i]

            # prepare data appropriately
            img_gpu = torch.from_numpy(np.expand_dims(img, axis=0)).to(device, memory_format=torch.channels_last)
            img_gpu = img_gpu.permute(0, 3, 1, 2) / 255
            img_gpu.type(torch.float32)

            # apply unet to current patch
            unet_out_ext = model(img_gpu)
            unet_out_ext = unet_out_ext.detach().cpu().numpy()
            unet_out = np.squeeze(unet_out_ext, axis=0)

            # get mask (unet returns the values before softmax)
            unet_out_class = np.argmax(unet_out, axis=0)
            n_classes = unet_out.shape[0]
            
            if n_classes == 2:
                visualize_two_class(img, mask, unet_out, unet_out_class)
            elif n_classes == 3:
                visualize_three_class(img, mask, unet_out, unet_out_class)
            else:
                raise ValueError(f"Unsupported number of classes: {n_classes}")

            # save
            tile, mask, output = Image.fromarray(img), Image.fromarray(np.uint8(255 * np.int64(mask)/2)), Image.fromarray(np.uint8(255 * unet_out_class/2))
            tile.save(os.path.join(path_save_data, f'tile/region_{i}_tile.png'))
            mask.save(os.path.join(path_save_data, f'mask/region_{i}_mask.png'))
            output.save(os.path.join(path_save_data, f'output/region_{i}_output.png'))

plt.show()

