import collections
import copy
import sys
import time
from random import seed
import numpy as np
from torch import optim
from dataset import BOSCHDATASET
import dataset
import evaluating_model
from diffusion_model import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from UNet import UNetModel, update_ema_params
from tqdm import tqdm
import torchvision.transforms as transforms

torch.cuda.empty_cache()

ROOT_DIR = "./"


def train(train_loader, test_loader, args, resume):
    in_channels = args["channels"] if args["channels"] != "" else 1

    # Initialize the model
    model = UNetModel(
        args['img_size'][0],
        args['base_channels'],
        channel_mults=args['channel_mults'],
        dropout=args['dropout'],
        n_heads=args['num_heads'],
        n_head_channels=args['num_head_channels'],
        in_channels=in_channels
    )

    # Initialize beta schedule and diffusion model
    betas = get_beta_schedule(args['T'], args['beta_schedule'])
    diffusion = GaussianDiffusionModel(
        args['img_size'],
        betas,
        loss_weight=args['loss_weight'],
        loss_type=args['loss-type'],
        noise=args['noise_fn'],
        img_channels=in_channels
    )

    # Load model from checkpoint if resuming
    if resume:
        model.load_state_dict(resume.get("unet", resume.get("ema")))

        ema = UNetModel(
            args['img_size'][0],
            args['base_channels'],
            channel_mults=args['channel_mults'],
            dropout=args['dropout'],
            n_heads=args['num_heads'],
            n_head_channels=args['num_head_channels'],
            in_channels=in_channels
        )
        ema.load_state_dict(resume["ema"])
        start_epoch = resume['n_epoch']
    else:
        start_epoch = 0
        ema = copy.deepcopy(model)

    # Prepare for training
    model.to(device)
    ema.to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args['lr'],
        weight_decay=args['weight_decay'],
        betas=(0.9, 0.999)
    )

    if resume:
        optimizer.load_state_dict(resume["optimizer_state_dict"])

    del resume

    start_time = time.time()
    losses = []
    vlb_history = collections.deque(maxlen=10)

    for epoch in tqdm(range(start_epoch, args['EPOCHS'] + 1), desc="Training Progress"):
        batch_losses = []
        for i, batch in enumerate(train_loader):
            x = batch["image"].to(device)
            loss, _ = diffusion.p_loss(model, x, args)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            update_ema_params(ema, model)

            print(f"Epoch: {epoch} || Batch: {i} || Loss: {loss.item()}")
            batch_losses.append(loss.item())

        epoch_loss = np.mean(batch_losses)
        losses.append(epoch_loss)
        print(f"Epoch: {epoch} || Mean Loss: {epoch_loss}")

        if epoch % 200 == 0:
            elapsed = time.time() - start_time
            remaining = args['EPOCHS'] - epoch
            avg_time = elapsed / (epoch + 1 - start_epoch)
            hours = int((remaining * avg_time) / 3600)
            mins = int(((remaining * avg_time) % 3600) / 60)

            vlb_terms = diffusion.calc_total_vlb(x, model, args)
            vlb_history.append(vlb_terms["total_vlb"].mean(dim=-1).cpu().item())

        if epoch % 500 == 0:
            save(unet=model, args=args, optimiser=optimizer, final=False, ema=ema, epoch=epoch)

    np.save("Thesis678/loss/final_losses.npy", np.array(losses))
    save(unet=model, args=args, optimiser=optimizer, ema=ema)
    evaluating_model.testing(test_loader, diffusion, ema=ema, args=args, model=model)


def save(unet, optimiser, args, ema, loss=0, epoch=0):
    checkpoint = {
        'n_epoch': epoch,
        'model_state_dict': unet.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'args': args,
        'ema': ema.state_dict(),
        'loss': loss
    }
    save_path = f'{ROOT_DIR}model/diff-params-ARGS={args["arg_num"]}/checkpoint/diff_epoch={epoch}.pt'
    torch.save(checkpoint, save_path)

if __name__ == '__main__':

    # Set up the device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed(1)

    # Check command line arguments
    arguments = sys.argv[1:]
    if not arguments:
        raise ValueError("File argument is missing.")

    resume_mode = 0
    first_arg = arguments[0]

    if first_arg == "RESUME_RECENT":
        resume_mode = 1
        arguments = arguments[1:]
    elif first_arg == "RESUME_FINAL":
        resume_mode = 2
        arguments = arguments[1:]

    if not arguments:
        raise ValueError("File argument is missing after RESUME command.")

    # Process file argument
    config_file = arguments[0]
    if config_file.isdigit():
        config_file = f"args{config_file}.json"
    elif config_file.startswith("args") and not config_file.endswith(".json"):
        config_file = f"args{config_file[4:]}.json"
    elif not (config_file.startswith("args") and config_file.endswith(".json")):
        raise ValueError("Provided file argument is not a JSON file.")

    # Load configuration
    with open(os.path.join(ROOT_DIR, 'test_args', config_file), 'r') as f:
        args = json.load(f)

    args['arg_num'] = config_file[4:-5]
    args = defaultdict_from_json(args)

    # Create necessary directories
    required_dirs = [
        f'./model/diff-params-ARGS={args["arg_num"]}',
        f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint',
        f'./diffusion-videos/ARGS={args["arg_num"]}',
        f'./diffusion-training-images/ARGS={args["arg_num"]}'
    ]

    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)

    # Initialize dataset if channels are specified
    if args.get("channels"):
        in_channels = args["channels"]

        dataset_path = "path to your dataset"
        data_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        train_data = BOSCHDATASET(root_dir=dataset_path, train=True, transform=data_transforms, random_crop=False)
        test_data = BOSCHDATASET(root_dir=dataset_path, train=False, transform=data_transforms, random_crop=False)

        train_loader = dataset.init_dataset_loader(train_data, args)
        test_loader = dataset.init_dataset_loader(test_data, args, shuffle=False)

    # Load model if resuming
    model_checkpoint = {}
    if resume_mode:
        if resume_mode == 1:
            checkpoint_dir = f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint'
            checkpoint_files = sorted(os.listdir(checkpoint_dir), reverse=True)

            for checkpoint in checkpoint_files:
                try:
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
                    model_checkpoint = torch.load(checkpoint_path, map_location=device)
                    break
                except RuntimeError:
                    continue

        elif resume_mode == 2:
            final_checkpoint_path = f'./model/diff-params-ARGS={args["arg_num"]}/params-final.pt'
            model_checkpoint = torch.load(final_checkpoint_path, map_location=device, weights_only=False)

    # Start training
    train(train_loader, test_loader, args, model_checkpoint)
