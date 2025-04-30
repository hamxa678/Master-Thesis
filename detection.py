from asyncio import constants
from typing import Any
import torch
from UNet import *
from dataset import *
from visualize import *
from dataset import Dataset_DA_DET
from anomaly_map import *
from metrics import *
from domain_adaptation import *
from reconstruction import *
import csv
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

class Detection:
    def __init__(self, unet, config) -> None:
        self.test_dataset = Dataset_DA_DET(
            root= config.data.data_dir,
            category=config.data.category,
            config = config,
            is_train=False,
        )
        self.testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size= config.data.test_batch_size,
            shuffle=False,
            num_workers= config.model.num_workers,
            drop_last=False,
        )
        self.unet = unet
        self.config = config
        self.reconstruction = Reconstruction(self.unet, self.config)
        self.transform = transforms.Compose([
                            transforms.CenterCrop((224)), 
                        ])
        
    def save_image(self, tensor, path):
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        image = transforms.ToPILImage()(tensor.cpu().squeeze(0))
        # Save image
        image.save(path)

    def __call__(self) -> Any:
        feature_extractor = domain_adaptation(self.unet, self.config, fine_tune=False)
        feature_extractor.eval()
        
        labels_list = []
        predictions= []
        anomaly_map_list = []
        gt_list = []
        reconstructed_list = []
        forward_list = []
        transform = transforms.Compose([
            # transforms.Lambda(lambda t: (t + 1) / (2)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        ls = []

        with torch.no_grad():
            i = 1
            for input, gt, labels, file_name in self.testloader:

                input = input.to(self.config.model.device)
                x_0 = self.reconstruction(input, input, self.config.model.w)
                x0 = x_0[-1]

                # Convert tensors to PIL images in grayscale
                input_image_gray = transforms.ToPILImage()(input[0].cpu()).convert("L")
                x0_image_gray = transforms.ToPILImage()(x0[0].cpu()).convert("L")
                import matplotlib.pyplot as plt
                # Create a figure to display the grayscale images
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(input_image_gray, cmap="gray")
                axes[0].set_title("Input (Grayscale)")
                axes[0].axis("off")

                axes[1].imshow(x0_image_gray, cmap="gray")
                axes[1].set_title("Reconstructed (x_0) (Grayscale)")
                axes[1].axis("off")

                # Save the figure
                save_path = "/content/DDADC/test_images"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.savefig(f"{save_path}/grayscale_comparison{i}.png")
                plt.close(fig)

                # self.save_image(x0[0], '/content/DDADC/recons/Fully_denoised.png'),
                # self.save_image(x_0[-5][0], '/content/DDADC/recons/Partially_demoised.png'),
                x0_f = x0.repeat(1, 3, 1, 1)
                input_f = input.repeat(1, 3, 1, 1)

                x0_f = transform(x0_f)
                input_f = transform(input_f)
                anomaly_map, p_anomaly_map, f_anomaly_map = heat_map(x0, input, feature_extractor, self.config)

                anomaly_map = self.transform(anomaly_map)
                gt = self.transform(gt)

                forward_list.append(input)
                anomaly_map_list.append(anomaly_map)

                # TODO: printing the image as a gray scale.
                
                # Display and save all images together
                import matplotlib.pyplot as plt

                # Convert tensors to PIL images
                input_image = transforms.ToPILImage()(input[0].cpu())
                x0_image = transforms.ToPILImage()(x0[0].cpu())
                gt_image = transforms.ToPILImage()(gt[0].cpu())
                anomaly_map_image = transforms.ToPILImage()(anomaly_map[0].cpu())
                p_anomaly_image = transforms.ToPILImage()(p_anomaly_map[0].cpu())
                f_anomaly_map_image = transforms.ToPILImage()(f_anomaly_map[0].cpu())

                # Create a figure to display the images
                fig, axes = plt.subplots(1, 6, figsize=(20, 5))
                axes[0].imshow(input_image)
                axes[0].set_title("Input")
                axes[0].axis("off")

                axes[1].imshow(x0_image)
                axes[1].set_title("Reconstructed (x_0)")
                axes[1].axis("off")

                axes[2].imshow(gt_image)
                axes[2].set_title("Ground Truth (GT)")
                axes[2].axis("off")

                axes[3].imshow(anomaly_map_image, cmap="hot")
                axes[3].set_title("Anomaly Map")
                axes[3].axis("off")

                axes[4].imshow(p_anomaly_image, cmap="hot")
                axes[4].set_title("pixel-wise")
                axes[4].axis("off")

                axes[5].imshow(f_anomaly_map_image, cmap="hot")
                axes[5].set_title("feature-wise")
                axes[5].axis("off")

                # Save the figure
                save_path = "/content/DDADC/test-image"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.savefig(f"{save_path}/comparison{i}.png")
                plt.close(fig)
                i += 1
                gt_list.append(gt)
                reconstructed_list.append(x0)

                for pred, label in zip(anomaly_map, labels):
                    print(f"pred shape :: {pred.shape}")
                    print(f"label : {label}")
                    if not torch.isnan(torch.max(pred)):
                        labels_list.append(0 if label == 'good' else 1)
                        predictions.append(torch.max(pred).item())
                        dic = {
                            "file_name": file_name, "label":0 if label == 'good' else 1, "pred": torch.max(pred).item(), "ano_score": torch.mean(anomaly_map).cpu().item()
                        }
                        ls.append(dic)
        
        metric = Metric(labels_list, predictions, anomaly_map_list, gt_list, self.config)
        threasold = metric.optimal_threshold()
        print(f"threasold value :: {threasold}")
        if self.config.metrics.auroc:
            print('AUROC: ({:.1f},{:.1f})'.format(metric.image_auroc() * 100, metric.pixel_auroc() * 100))
        if self.config.metrics.pro:
            print('PRO: {:.1f}'.format(metric.pixel_pro() * 100))
        if self.config.metrics.misclassifications:
            metric.miscalssified()
        reconstructed_list = torch.cat(reconstructed_list, dim=0)
        forward_list = torch.cat(forward_list, dim=0)
        anomaly_map_list = torch.cat(anomaly_map_list, dim=0)
        pred_mask = (anomaly_map_list > metric.threshold).float()
        gt_list = torch.cat(gt_list, dim=0)
        if not os.path.exists('results'):
                os.mkdir('results')
        if self.config.metrics.visualisation:
            visualize(forward_list, reconstructed_list, gt_list, pred_mask, anomaly_map_list, self.config.data.category)


def detection(config):
    unet = build_model(config)
    checkpoint = torch.load(os.path.join(os.getcwd(), config.model.checkpoint_dir, "3000.pt"), weights_only=False)
    unet = torch.nn.DataParallel(unet)
    if "unet" in checkpoint:
        new_state_dict = {f"module.{k}": v for k, v in checkpoint["unet"].items()}
        unet.load_state_dict(new_state_dict, strict=False)
    else:
        new_state_dict = {f"module.{k}": v for k, v in checkpoint["ema"].items()}
        unet.load_state_dict(new_state_dict, strict=False)
    # unet.load_state_dict(checkpoint)    
    unet.to(config.model.device)
    # checkpoint = torch.load(os.path.join(os.getcwd(), config.model.checkpoint_dir, str(config.model.load_chp)))
    unet.eval()
    ddad = Detection(unet, config)
    ddad()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()
    config = OmegaConf.load(args.config)
    torch.manual_seed(42)
    np.random.seed(42)
    detection(config)
