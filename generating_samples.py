import os
import sys
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

import dataset
from helpers import *
from diffusion_model import GaussianDiffusionModel, get_beta_schedule
from UNet import UNetModel



def single_instance_output():
    args, output = load_parameters(device)
    in_channels = args["channels"] if args["channels"] != "" else (3 if args["dataset"].lower() == "leather" else 1)

    unet = UNetModel(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], in_channels=in_channels)
    betas = get_beta_schedule(args['T'], args['beta_schedule'])
    diff = GaussianDiffusionModel(args['img_size'], betas, loss_weight=args['loss_weight'],
                                  loss_type=args['loss-type'], noise=args['noise_fn'], img_channels=in_channels)
    unet.load_state_dict(output["ema"])
    unet.to(device).eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
    ano_dataset = dataset.BOSCHDATASET("/content/Thesis678/BOSCHv2.0/", anomalous=True, img_size=args["img_size"], transform=transform)

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    loader = cycle(DataLoader(ano_dataset, batch_size=args['Batch_Size'], num_workers=2, drop_last=True))
    plt.rcParams['figure.dpi'] = 1000

    os.makedirs(f'./final-outputs/ARGS={args["arg_num"]}', exist_ok=True)

    for i in range(30):
        predictions, sequences, masks, mse_thresholds = [], [], [], []
        rows, t_distance, threshold = 1, 550, 0.4

        for _ in range(rows):
            batch = next(loader)
            img = batch["image"].to(device)
            img_mask = batch["mask"].to(device)

            out_seq = diff.forward_backward(unet, img, see_whole_sequence="whole", t_distance=t_distance, denoise_fn=args["noise_fn"])
            recon_img = out_seq[-1].squeeze(1).to(device)
            x_t_img = out_seq[t_distance // 2].squeeze(1).to(device)

            pred_tensor, threshold_mask = make_prediction(img, normalize_image(recon_img), img_mask, normalize_image(x_t_img), threshold=threshold)
            predictions.append(pred_tensor)
            mse_thresholds.append(threshold_mask)
            masks.append(img_mask)
            sequences.append(out_seq)

        filename = f'./final-outputs/ARGS={args["arg_num"]}/attempt={len(os.listdir(f"./final-outputs/ARGS={args["arg_num"]}")) + 1}-{threshold}-predictions.png'
        output_masked_comparison(predictions, filename, t_distance)
        plt.close('all')


def different_frequency_output():
    args, output = load_parameters(device)

    unet = UNetModel(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'])
    betas = get_beta_schedule(args['T'], args['beta_schedule'])
    diff = GaussianDiffusionModel(args['img_size'], betas, loss_weight=args['loss_weight'],
                                  loss_type=args['loss-type'], noise=args['noise_fn'])

    unet.load_state_dict(output["ema"])
    unet.to(device).eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
    ano_dataset = dataset.BOSCHDATASET("/content/Thesis678/BOSCHv2.0/", anomalous=True, img_size=args["img_size"], transform=transform)

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    loader = cycle(DataLoader(ano_dataset, batch_size=args['Batch_Size'], num_workers=2, drop_last=True))

    os.makedirs(f'./final-outputs/ARGS={args["arg_num"]}', exist_ok=True)

    for i in range(22):
        print(f"epoch {i}")
        batch = next(loader)
        img = batch["image"].to(device).reshape(-1, 1, *args["img_size"])
        img_mask = batch["mask"].to(device)

        output = diff.detection_A_fixedT(unet, img[0].unsqueeze(0), args, img_mask[0].unsqueeze(0))

        fig, subplots = plt.subplots(6, 6, sharex=True, sharey=True, figsize=(6, 6),
                                     gridspec_kw={'wspace': 0, 'hspace': 0}, squeeze=False)

        for r in range(6):
            for c in range(6):
                img_data = output[6 * r + c].reshape(*output.shape[-2:]).cpu().numpy()
                cmap = "hot" if c == 3 else "gray"
                subplots[r][c].imshow(img_data, cmap=cmap)
                subplots[r][c].axis('off')

        for i, label in enumerate(["$x_0$", "$x_{250}$", "Reconstruction", "Square Error", "Prediction", "Ground Truth"]):
            subplots[0][i].set_xlabel(label, fontsize=6)
            subplots[0][i].xaxis.set_label_position("top")

        for i in range(6):
            subplots[i][0].set_ylabel(f"$2^{{{i + 1}}}={2 ** (i + 1)}$", fontsize=6)
            subplots[i][0].yaxis.set_label_position("left")

        plt.tick_params(labelcolor='none', top=False, left=False, bottom=False, right=False)
        plt.ylabel("Starting Frequency\n", fontsize=6)

        filename = f'./final-outputs/ARGS={args["arg_num"]}/{batch["filename"][0][-9:-4]}-frequency-attempt={len(os.listdir(f"./final-outputs/ARGS={args["arg_num"]}")) + 1}.png'
        plt.savefig(filename)
        plt.close('all')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scale_img = lambda img: ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)

    if str(sys.argv[1]) == "30":
        different_frequency_output()
    elif str(sys.argv[1]) == "31":
        single_instance_output()
