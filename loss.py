import torch
import torch.nn as nn
import numpy as np
from simplex import Simplex_CLASS

def generate_simplex_noise(
        Simplex_instance, x, t, random_param=False, octave=6, persistence=0.8, frequency=64,
        in_channels=1
        ):

    noise = torch.empty(x.shape).to(x.device)
    # noise.shape :: noise shape:: torch.Size([1, 1200, 1600])
    for i in range(in_channels):
        Simplex_instance.newSeed()
        if random_param:
            param = random.choice(
                    [(2, 0.6, 16), (6, 0.6, 32), (7, 0.7, 32), (10, 0.8, 64), (5, 0.8, 16), (4, 0.6, 16), (1, 0.6, 64),
                     (7, 0.8, 128), (6, 0.9, 64), (2, 0.85, 128), (2, 0.85, 64), (2, 0.85, 32), (2, 0.85, 16),
                     (2, 0.85, 8),
                     (2, 0.85, 4), (2, 0.85, 2), (1, 0.85, 128), (1, 0.85, 64), (1, 0.85, 32), (1, 0.85, 16),
                     (1, 0.85, 8),
                     (1, 0.85, 4), (1, 0.85, 2), ]
                    )
            # 2D octaves seem to introduce directional artifacts in the top left
            noise[:, i, ...] = torch.unsqueeze(
                    torch.from_numpy(
                            # Simplex_instance.rand_2d_octaves(
                            #         x.shape[-2:], param[0], param[1],
                            #         param[2]
                            #         )
                            Simplex_instance.rand_3d_fixed_T_octaves(
                                    x.shape[-2:], t.detach().cpu().numpy(), param[0], param[1],
                                    param[2]
                                    )
                            ).to(x.device), 0
                    ).repeat(x.shape[0], 1, 1, 1)
        
        # lst = torch.from_numpy(
        #     Simplex_instance.rand_3d_fixed_T_octaves(
        #         x.shape[-2:], t.detach().cpu().numpy(), octave, persistence, frequency
        #     )).to(x.device).squeeze(0).shape
        # print(lst)
        # noise = torch.unsqueeze(noise,0)
        # print(f'noise shape level one :: {noise.shape}')
        # print(f'unsqueezed noise :: {noise.shape}')
        # print(Simplex_instance.rand_3d_fixed_T_octaves(x.shape[-2:], t.detach().cpu().numpy(), octave,persistence, frequency).shape)


        #unchanged
        # noise = torch.unsqueeze(
        #     # Simplex3d :: torch.Size([1, 1200, 1600])
        #         torch.from_numpy(
        #                 # Simplex_instance.rand_2d_octaves(
        #                 #         x.shape[-2:], octave,
        #                 #         persistence, frequency
        #                 #         )
        #                 Simplex_instance.rand_3d_fixed_T_octaves(
        #                         x.shape[-2:], t.detach().cpu().numpy(), octave,
        #                         persistence, frequency
        #                         )
        #                 ).to(x.device), 0
        #         ).repeat(x.shape[0], 1, 1, 1)
        
        noise = torch.from_numpy(
                # Simplex_instance.rand_2d_octaves(
                #         x.shape[-2:], octave,
                #         persistence, frequency
                #         )
                Simplex_instance.rand_3d_fixed_T_octaves(
                        x.shape[-2:], t.detach().cpu().numpy(), octave,
                        persistence, frequency
                        )
                        ).to(x.device)
        # print(f'noise shape level two :: {noise.shape}')
    # print(f"Ulambaaaa :: {noise.shape}")    
    return noise

def get_loss(model, x_0, t, config):
    # print("Loss ftnnnnnnn")
    simplex_instance = Simplex_CLASS()
    x_0 = x_0.to(config.model.device)
    
    betas = torch.linspace(
        config.model.beta_start, 
        config.model.beta_end, 
        config.model.trajectory_steps, 
        dtype=torch.float, 
        device=config.model.device
    )
    
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    at = alpha_bar.index_select(0, t).view(-1, 1, 1, 1)

    # Replace Gaussian noise with Simplex noise
    e = generate_simplex_noise(Simplex_instance=simplex_instance, x=x_0, t=t).float()
    # e = torch.randn_like(x_0, device = x_0.device)
    # print(f"printing shape e model {e.shape}")

    # e = (e - e.mean()) / e.std()


    # Apply forward diffusion process
    x_t = at.sqrt() * x_0 + (1 - at).sqrt() * e 
    # print(f"printing shape before model {x_t.shape}{t.shape}")
    # Predict noise using the model
    output = model(x_t, t.float())
    loss = (e - output).square().mean(dim=(1, 2, 3)).mean(dim=0)
    # Compute denoising loss (Normalized MSE)
    return loss


# def get_loss(model, x_0, t, config):
#     x_0 = x_0.to(config.model.device)
#     betas = np.linspace(config.model.beta_start, config.model.beta_end, config.model.trajectory_steps, dtype=np.float64)
#     b = torch.tensor(betas).type(torch.float).to(config.model.device)
#     e = torch.randn_like(x_0, device = x_0.device)
#     at = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)


#     x = at.sqrt() * x_0 + (1- at).sqrt() * e 
#     output = model(x, t.float())
#     return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)