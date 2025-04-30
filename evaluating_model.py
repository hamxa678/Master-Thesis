import os
import torch
from helpers import PSNR, print_stat
import dataset
from diffusion_model import GaussianDiffusionModel, get_beta_schedule
from UNet import UNetModel
from anomaly_matirc import load_parameters
from tqdm import tqdm

def testing(test_loader, diffusion, args, ema, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/test-set/', exist_ok=True)
    ema.eval()
    model.eval()
    
    vlb_stats = []
    # Calculating VLB value for Test Set
    for i, data in enumerate(tqdm(test_loader, desc="Testing-VLB")):
        x = data["image"].to(device) if args["dataset"] != "cifar" else data[0].to(device)
        vlb_terms = diffusion.calc_total_vlb(x, model, args)
        print(f"VLB: {vlb_terms['total_vlb'].mean(dim=-1).cpu().item()}")
        vlb_stats.append(vlb_terms)


    # Calculating PSNR value for Test Set
    psnr_values = []
    for _, data in enumerate(tqdm(test_loader, desc="Testing-PSNR")):
        x = data["image"].to(device) if args["dataset"] != "cifar" else data[0].to(device)
        output = diffusion.forward_backward(ema, x, see_whole_sequence=None, t_distance=args["T"] // 2)
        psnr_val = PSNR(output, x)
        print(psnr_val)
        psnr_values.append(psnr_val)

    total_vlb_vals = [i['total_vlb'].mean(dim=-1).cpu().item() for i in vlb_stats]
    prior_vlb_vals = [i['prior_vlb'].mean(dim=-1).cpu().item() for i in vlb_stats]

    print_stat("Test set total VLB", total_vlb_vals)
    print_stat("Test set prior VLB", prior_vlb_vals)
    print_stat("Test set PSNR", psnr_values)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args, loaded_state = load_parameters(device)

    in_channels = 1
    unet = UNetModel(
        args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], in_channels=in_channels
    )
    ema = UNetModel(
        args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], in_channels=in_channels
    )

    betas = get_beta_schedule(args['T'], args['beta_schedule'])
    diffusion = GaussianDiffusionModel(
        args['img_size'], betas, loss_weight=args['loss_weight'],
        loss_type=args['loss-type'], noise=args['noise_fn']
    )

    ema.load_state_dict(loaded_state["ema"])
    ema.to(device).eval()

    unet.load_state_dict(loaded_state["model_state_dict"])
    unet.to(device).eval()

    _, test_data = dataset.init_datasets("/content/Thesis678/BOSCHv2.0", args)
    test_loader = dataset.init_dataset_loader(test_data, args, shuffle=False)
    testing(test_loader, diffusion, args, ema, unet)
