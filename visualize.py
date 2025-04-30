import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import numpy as np
import torch
import os
from dataset import *


def visualalize_reconstruction(input, recon, target):
    plt.figure(figsize=(11,11))
    plt.subplot(1, 3, 1).axis('off')
    plt.subplot(1, 3, 2).axis('off')
    plt.subplot(1, 3, 3).axis('off')

    plt.subplot(1, 3, 1)
    plt.imshow(show_tensor_image(input))
    plt.title('input image')
    

    plt.subplot(1, 3, 2)
    plt.imshow(show_tensor_mask(recon))
    plt.title('recon image')

    plt.subplot(1, 3, 3)
    plt.imshow(show_tensor_mask(target))
    plt.title('target image')


    k = 0
    while os.path.exists('results/heatmap{}.png'.format(k)):
        k += 1
    plt.savefig('results/heatmap{}.png'.format(k))
    plt.close()


def visualize(image, noisy_image, GT, pred_mask, anomaly_map, category) :
    for idx, img in enumerate(image):
        plt.figure(figsize=(11,11))
        plt.subplot(1, 2, 1).axis('off')
        plt.subplot(1, 2, 2).axis('off')
        plt.subplot(1, 2, 1)
        plt.imshow(show_tensor_image(image[idx]))
        plt.title('clear image')

        plt.subplot(1, 2, 2)

        plt.imshow(show_tensor_image(noisy_image[idx]))
        plt.title('reconstructed image')
        plt.savefig('results/{}sample{}.png'.format(category,idx))
        plt.close()

        plt.figure(figsize=(11,11))
        plt.subplot(1, 3, 1).axis('off')
        plt.subplot(1, 3, 2).axis('off')
        plt.subplot(1, 3, 3).axis('off')

        plt.subplot(1, 3, 1)
        plt.imshow(show_tensor_mask(GT[idx]))
        plt.title('ground truth')

        plt.subplot(1, 3, 2)
        plt.imshow(show_tensor_mask(pred_mask[idx]))
        plt.title('normal' if torch.max(pred_mask[idx]) == 0 else 'abnormal', color="g" if torch.max(pred_mask[idx]) == 0 else "r")

        plt.subplot(1, 3, 3)
        plt.imshow(show_tensor_image(anomaly_map[idx]))
        plt.title('heat map')
        plt.savefig('results/{}sample{}heatmap.png'.format(category,idx))
        plt.close()



def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / (2)),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
    ])

    # Takes the first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    return reverse_transforms(image)

def show_tensor_mask(image):
    reverse_transforms = transforms.Compose([
        # transforms.Lambda(lambda t: (t + 1) / (2)),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.int8)),
    ])

    # Takes the first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    return reverse_transforms(image)
        

# AUROC: (95.6,87.9)