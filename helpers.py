import json
import os
from collections import defaultdict
import random
from matplotlib import transforms
import torch
import torchvision.utils
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import auc, roc_curve
import argparse

from UNet import UNetModel


def gridify_output(img, row_size=-1):
    scale_img = lambda img: ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    return torchvision.utils.make_grid(scale_img(img), nrow=row_size, pad_value=-1).cpu().data.permute(
            0, 2,
            1
            ).contiguous().permute(
            2, 1, 0
            )


def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd


def load_checkpoint(param, use_checkpoint, device):
    """
    loads the most recent (non-corrupted) checkpoint or the final model
    :param param: args number
    :param use_checkpoint: checkpointed or final model
    :return:
    """
    if not use_checkpoint:
        return torch.load(f'./model/diff-params-ARGS={param}/params-final.pt', map_location=device, weights_only=False)
    else:
        checkpoints = os.listdir(f'./model/diff-params-ARGS={param}/checkpoint')
        checkpoints.sort(reverse=True)
        for i in checkpoints:
            try:
                file_dir = f"./model/diff-params-ARGS={param}/checkpoint/{i}"
                loaded_model = torch.load(file_dir, map_location=device)
                break
            except RuntimeError:
                continue
        return loaded_model


def load_parameters(device):
    """
    Loads the trained parameters for the detection model
    :return:
    """
    import sys

    if len(sys.argv[1:]) > 0:
        params = sys.argv[1:]
    else:
        params = os.listdir("./model")
    if ".DS_Store" in params:
        params.remove(".DS_Store")

    if params[0] == "CHECKPOINT":
        use_checkpoint = True
        params = params[1:]
    else:
        use_checkpoint = False

    print(params)
    for param in params:
        if param.isnumeric():
            output = load_checkpoint(param, use_checkpoint, device)
        elif param[:4] == "args" and param[-5:] == ".json":
            output = load_checkpoint(param[4:-5], use_checkpoint, device)
        elif param[:4] == "args":
            output = load_checkpoint(param[4:], use_checkpoint, device)
        else:
            raise ValueError(f"Unsupported input {param}")

        # if "args" in output:
        #     args = output["args"]

        #     print(args)
        # else:
        try:
            with open(f'./test_args/args{param}.json', 'r') as f:
                args = json.load(f)
            args['arg_num'] = param
            args = defaultdict_from_json(args)
        except FileNotFoundError:
            raise ValueError(f"args{param} doesn't exist for {param}")

        if "noise_fn" not in args:
            args["noise_fn"] = "gauss"

        return args, output

def heatmap(real: torch.Tensor, recon: torch.Tensor, mask, filename, save=True):
    mse = ((recon - real).square() * 2) - 1
    thresholded = ((mse > 0).float() * 2) - 1
    
    if save:
        output = torch.cat((real, recon.reshape(1, *recon.shape), mse, thresholded, mask))
        plt.imshow(gridify_output(output, 5)[..., 0], cmap="gray")
        plt.axis('off')
        plt.savefig(filename)
        plt.clf()


def normalize_image(image):
    image = image.to(torch.float32)
    min_val, max_val = image.min(), image.max()
    if 0 <= min_val and max_val <= 1:
        return image


def dice_coeff(real: torch.Tensor, recon: torch.Tensor, real_mask: torch.Tensor, smooth=1e-6, mse=None):
    intersection = torch.sum(mse * real_mask, dim=[1, 2, 3])
    union = torch.sum(mse, dim=[1, 2, 3]) + torch.sum(real_mask, dim=[1, 2, 3])
    return torch.mean((2. * intersection + smooth) / (union + smooth))


def PSNR(recon, real):
    mse = torch.mean((real - recon).square(), dim=list(range(real.dim())))
    return (20 * torch.log10(torch.max(real) / torch.sqrt(mse))).detach().cpu().numpy()


def SSIM(real, recon):
    return ssim(real.detach().cpu().numpy(), recon.detach().cpu().numpy(), channel_axis=2, data_range=1)


def IoU(real, recon):
    real, recon = real.cpu().numpy(), recon.cpu().numpy()
    intersection = np.logical_and(real, recon)
    union = np.logical_or(real, recon)
    return np.sum(intersection) / (np.sum(union) + 1e-8)


def precision(real_mask, recon_mask):
    TP = (real_mask == 1) & (recon_mask == 1)
    FP = (real_mask == 1) & (recon_mask == 0)
    return torch.sum(TP).float() / (torch.sum(TP + FP).float() + 1e-6)


def recall(real_mask, recon_mask):
    TP = (real_mask == 1) & (recon_mask == 1)
    FN = (real_mask == 0) & (recon_mask == 1)
    return torch.sum(TP).float() / (torch.sum(TP + FN).float() + 1e-6)


def FPR(real_mask, recon_mask):
    FP = (real_mask == 1) & (recon_mask == 0)
    TN = (real_mask == 0) & (recon_mask == 0)
    return torch.sum(FP).float() / (torch.sum(FP + TN).float() + 1e-6)


def ROC_AUC(real_mask, square_error):
    if isinstance(real_mask, torch.Tensor):
        real_mask = real_mask.detach().cpu().numpy().flatten()
        square_error = square_error.detach().cpu().numpy().flatten()
    return roc_curve(real_mask, square_error)


def AUC_score(fpr, tpr):
    return auc(fpr, tpr)


def print_stat(name, values):
    mean_val = np.mean(values)
    std_val = np.std(values)
    print(f"{name}: {mean_val:.4f} ± {std_val:.4f}")



def get_beta_schedule(num_diffusion_steps, schedule_type="cosine"):
    if schedule_type == "cosine":
        f = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        betas = [min(1 - f((i + 1) / num_diffusion_steps) / f(i / num_diffusion_steps), 0.999)
                 for i in range(num_diffusion_steps)]
        return np.array(betas)
    elif schedule_type == "linear":
        beta_start = 1000 / num_diffusion_steps * 0.0001
        beta_end = 1000 / num_diffusion_steps * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_steps, dtype=np.float64)
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule_type}")


def extract(array, timesteps, shape, device):
    tensor = torch.from_numpy(array).to(device=timesteps.device)[timesteps].float()
    while len(tensor.shape) < len(shape):
        tensor = tensor[..., None]
    return tensor.expand(shape).to(device)


def mean_flat(tensor):
    return torch.mean(tensor, dim=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (-1 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def discretised_gaussian_log_likelihood(x, means, log_scales):
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus = approx_standard_normal_cdf(inv_stdv * (centered_x + 1.0 / 255.0))
    minus = approx_standard_normal_cdf(inv_stdv * (centered_x - 1.0 / 255.0))

    log_cdf_plus = torch.log(plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - minus).clamp(min=1e-12))
    cdf_delta = plus - minus

    return torch.where(x < -0.999, log_cdf_plus,
                       torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))))


def generate_simplex_noise(simplex, x, t, randomize=False, octave=6, persistence=0.8, frequency=64, in_channels=1):
    B, C, H, W = x.shape
    noise = torch.zeros((B, in_channels, H, W), device=x.device)
    t_np = t.detach().cpu().numpy()

    for ch in range(in_channels):
        simplex.newSeed()
        simplex_noise = simplex.rand_3d_fixed_T_octaves((H, W), t_np, octave, persistence, frequency)
        noise[:, ch, :, :] = torch.from_numpy(simplex_noise).to(x.device).float()
    return noise


def random_noise(simplex, x, t):
    return torch.randn_like(x) if random.choice(["gauss", "simplex"]) == "gauss" else generate_simplex_noise(simplex, x, t)


def make_prediction(real, recon, mask, x_t, threshold=0.5, error_fn="sq"):
    if error_fn == "sq":
        error = ((recon - real).square() * 2) - 1
    elif error_fn == "l1":
        error = recon - real

    thresholded = ((error > (threshold * 2) - 1).float() * 2) - 1
    return torch.cat((real, x_t, recon, error, thresholded, mask)), thresholded



def output_masked_comparison(sequence, filename, t_distance=250):
    if isinstance(sequence, torch.Tensor):
        sequence = [sequence]

    fig, subplots = plt.subplots(len(sequence), 6, figsize=(6, len(sequence)), squeeze=False,
                                 gridspec_kw={'wspace': 0, 'hspace': 0})

    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

    for i, item in enumerate(sequence):
        for j in range(item.shape[0]):
            cmap = "gray" if j != 3 else "hot"
            img = (item[j] + 1).permute(1, 2, 0).cpu().numpy() if j <= 2 else \
                  (transforms.functional.rgb_to_grayscale(item[j] + 1).permute(1, 2, 0).cpu().numpy()
                   if item[j].shape[-3] == 3 else (item[j] + 1).permute(1, 2, 0).cpu().numpy())
            if j == 4 and item[j].shape[-3] == 3:
                img = ((img > 0.1).astype(float) * 2) - 1
            subplots[i][j].imshow(img, cmap=cmap)
            subplots[i][j].axis('off')

    labels = ["$x_0$", f"$x_{{{t_distance}}}$", "Reconstruction", "Square Error", "Prediction", "Ground Truth"]
    for i, label in enumerate(labels):
        subplots[0][i].set_xlabel(label, fontsize=6)
        subplots[0][i].xaxis.set_label_position("top")

    plt.savefig(filename)

def parse_args():
    cmdline_parser = argparse.ArgumentParser('DDAD')    
    cmdline_parser.add_argument('-cfg', '--config', 
                                default= os.path.join(os.path.dirname(os.path.abspath(__file__)),'config.yaml'), 
                                help='config file')
    cmdline_parser.add_argument('--train', 
                                default= False, 
                                help='Train the diffusion model')
    cmdline_parser.add_argument('--detection', 
                                default= False, 
                                help='Detection anomalies')
    cmdline_parser.add_argument('--domain_adaptation', 
                                default= False, 
                                help='Domain adaptation')
    args, unknowns = cmdline_parser.parse_known_args()
    return args

def build_model(config):
    if config.model.DDADS:
        unet = UNetModel(config.data.image_size, base_channels=128,dropout=0, n_heads=2 ,in_channels=1)
    else:
        unet = UNetModel(config.data.image_size, base_channels=128,dropout=0, n_heads=2 ,in_channels=1)

    return unet