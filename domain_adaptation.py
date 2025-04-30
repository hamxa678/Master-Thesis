import logging
from omegaconf import OmegaConf
import torch
from dataset import *
from dataset import *
from helpers import build_model, parse_args
from UNet import *
from visualize import *
from resnet import *
import torchvision.transforms as T
from reconstruction import *
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

def loss_fucntion(a, b, c, d, config):
    cos_loss = torch.nn.CosineSimilarity()
    loss1 = 0
    loss2 = 0
    loss3 = 0
    for item in range(len(a)):
        loss1 += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1))) 
        loss2 += torch.mean(1-cos_loss(b[item].view(b[item].shape[0],-1),c[item].view(c[item].shape[0],-1))) * config.model.DLlambda
        loss3 += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),d[item].view(d[item].shape[0],-1))) * config.model.DLlambda
    loss = loss1+loss2+loss3
    return loss

def domain_adaptation(unet, config, fine_tune):
    if config.model.feature_extractor == 'wide_resnet101_2':
        feature_extractor = wide_resnet101_2(pretrained=True)
        frozen_feature_extractor = wide_resnet101_2(pretrained=True)
    elif config.model.feature_extractor == 'wide_resnet50_2':
        feature_extractor = wide_resnet50_2(pretrained=True)
        frozen_feature_extractor = wide_resnet50_2(pretrained=True)
    elif config.model.feature_extractor == 'resnet50': 
        feature_extractor = resnet50(pretrained=True)
        frozen_feature_extractor = resnet50(pretrained=True)
    else:
        logging.warning("Feature extractor is not correctly selected, Default: wide_resnet101_2")
        feature_extractor = wide_resnet101_2(pretrained=True)
        frozen_feature_extractor = wide_resnet101_2(pretrained=True)
    feature_extractor.to(config.model.device)  
    frozen_feature_extractor.to(config.model.device)
    frozen_feature_extractor.eval()
    feature_extractor = torch.nn.DataParallel(feature_extractor)
    frozen_feature_extractor = torch.nn.DataParallel(frozen_feature_extractor)
    train_dataset = Dataset_DA_DET(
        root= config.data.data_dir,
        category= config.data.category,
        config = config,
        is_train=True,
    )
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.DA_batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
        drop_last=True,
    ) 
    if fine_tune:      
        unet.eval()
        feature_extractor.train()
        transform = transforms.Compose([
                    # transforms.Lambda(lambda t: (t + 1) / (2)),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        optimizer = torch.optim.AdamW(feature_extractor.parameters(),lr= 1e-4)
        torch.save(frozen_feature_extractor.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir),f'feat0'))
        reconstruction = Reconstruction(unet, config)
        for epoch in range(config.model.DA_epochs):
            for step, batch in enumerate(trainloader):
                print(f"Epoch {epoch+1} | Step {step}")
                half_batch_size = batch[0].shape[0]//2
                target = batch[0][:half_batch_size].to(config.model.device)  
                input = batch[0][half_batch_size:].to(config.model.device) 

                # print(f"Input Image Shape: {input.shape}")
                # print(f"Target Image Shape: {target.shape}")
                
                x0 = reconstruction(input, target, config.model.w_DA)[-1].to(config.model.device)
                # print(f"Generated Image Shape: {x0.shape}")
                # print(f"======================")
                # Display and save the images

                x0 = x0.repeat(1, 3, 1, 1)
                target = target.repeat(1, 3, 1, 1)

                # print(f"Target Image Shape: {target.shape}")
                # print(f"Generated Image Shape: {x0.shape}")
                # return

                x0 = transform(x0)
                target = transform(target)

                reconst_fe = feature_extractor(x0)
                target_fe = feature_extractor(target)

                target_frozen_fe = frozen_feature_extractor(target)
                reconst_frozen_fe = frozen_feature_extractor(x0)

                loss = loss_fucntion(reconst_fe, target_fe, target_frozen_fe,reconst_frozen_fe, config)
                print(f"Loss: {loss.item()}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            print(f"Epoch {epoch+1} | Loss: {loss.item()}")
            if (epoch+1) % 4 == 0:
                print(f"Saving model at epoch {epoch+1}")
                torch.save(feature_extractor.state_dict(), os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir),f'feat{epoch+1}'))
    else:
        checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), f'feat{config.model.DA_chp}'))#{config.model.DA_chp}            
        feature_extractor.load_state_dict(checkpoint)  
    return feature_extractor


def finetuning(config):
    unet = build_model(config)
    checkpoint = torch.load(os.path.join(os.getcwd(), config.model.checkpoint_dir, "3000.pt"), weights_only=False)
    unet = torch.nn.DataParallel(unet)
    if "unet" in checkpoint:
        new_state_dict = {f"module.{k}": v for k, v in checkpoint["unet"].items()}
        unet.load_state_dict(new_state_dict, strict=False)
    else:
        new_state_dict = {f"module.{k}": v for k, v in checkpoint["ema"].items()}
        unet.load_state_dict(new_state_dict, strict=False)
    unet.to(config.model.device)
    unet.eval()
    domain_adaptation(unet, config, fine_tune=True)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()
    config = OmegaConf.load(args.config)
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    finetuning(config)