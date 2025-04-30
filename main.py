import torch
import numpy as np
import os
from detection import DDAD
from UNet import *
from omegaconf import OmegaConf
from domain_adaptation import * 
from detection import *
from helpers import parse_args
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"



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
    unet.to(config.model.device)
    unet.eval()
    ddad = DDAD(unet, config)
    ddad()
    

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
    # unet.load_state_dict(checkpoint)    
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
    if args.domain_adaptation:
        print('Domain Adaptation...')
        finetuning(config)
    if args.detection:
        print('Detecting Anomalies...')
        detection(config)


        