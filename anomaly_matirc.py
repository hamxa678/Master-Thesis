import time
import matplotlib.pyplot as plt
import numpy as np
import dataset
from diffusion_model import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from UNet import UNetModel
from torchvision import transforms

    
def anomalous_metric_calculation():
    args, output = load_parameters(device)
    in_channels = 1
    unet = UNetModel(
            args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], in_channels=in_channels
            )
    betas = get_beta_schedule(args['T'], args['beta_schedule'])
    diff = GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=in_channels
            )
    unet.load_state_dict(output["ema"])
    unet.to(device)
    unet.eval()
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
    ])
    d_set = dataset.BOSCHDATASET(
        "/content/Thesis678/BOSCHv2.0/", anomalous=True, img_size=args["img_size"],transform=transform,)
    d_set_size = len(d_set)
    loader = dataset.init_dataset_loader(d_set, args)
    plt.rcParams['figure.dpi'] = 200

    dice_data = []
    ssim_data = []
    IOU = []
    precision = []
    recall = []
    FPR = []
    AUC_scores = []

    start_time = time.time()
    for i in range(d_set_size):

        if args["dataset"].lower() != "carpet" and args["dataset"].lower() != "leather":
            new = next(loader)
            image = new["image"].to(device)
            mask = new["mask"].to(device)
        else:
            new = next(loader)
            image = new["image"].to(device)
            mask = new["mask"].to(device)


        output = diff.forward_backward(
                unet, image,
                see_whole_sequence="whole",
                t_distance=150, denoise_fn=args["noise_fn"]
                )

        image = normalize_image(image)
        output = normalize_image(output)

        image = image.to(device)
        output = output.to(device)


        mse = ((image - output).square() * 2) - 1
        threshold = 0.5
        mse = (mse > ((threshold * 2) - 1)).float()
        mask = (mask == 1).int()



        fpr_simplex, tpr_simplex, _ = ROC_AUC(mask.to(torch.uint8), mse)
        AUC_scores.append(AUC_score(fpr_simplex, tpr_simplex))
        dice_data.append(
                dice_coeff( image, output.to(device), mask, mse=mse).cpu().item())

        
        ssim_data.append(
                SSIM(image.permute(0, 2, 3, 1).reshape(*args["img_size"], image.shape[1]),output.permute(0, 2, 3, 1).reshape(*args["img_size"], image.shape[1])))
        precision.append(precision(mask, mse).cpu().numpy())
        recall.append(recall(mask, mse).cpu().numpy())
        IOU.append(IoU(mask, mse))
        FPR.append(FPR(mask, mse).cpu().numpy())
        plt.close('all')

        if i % 8 == 0:
            time_taken = time.time() - start_time
            remaining_epochs = d_set_size - i
            time_per_epoch = time_taken / (i + 1)
            hours = remaining_epochs * time_per_epoch / 3600
            mins = (hours % 1) * 60
            hours = int(hours)

            print(
                    f"elapsed time: {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                    f"remaining time: {hours}:{mins:02.0f}"
                    )

    print()
    print("Final results: ")
    print(f"Dice coefficient: {np.mean(dice_data)} +- {np.std(dice_data)}")
    print(f"Structural Similarity Index (SSIM): {np.mean(ssim_data)} +- {np.std(ssim_data)}")
    print(f"Precision: {np.mean(precision)} +- {np.std(precision)}")
    print(f"Recall: {np.mean(recall)} +- {np.std(recall)}")
    print(f"FPR: {np.mean(FPR)} +- {np.std(FPR)}")
    print(f"IOU: {np.mean(IOU)} +- {np.std(IOU)}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    anomalous_metric_calculation()