from typing import Any
import torch
# from forward_process import *
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
from simplex import Simplex_CLASS
from PIL import Image
import torchvision.transforms as transforms


class Reconstruction:
    '''
    The reconstruction process
    :param y: the target image
    :param x: the input image
    :param seq: the sequence of denoising steps
    :param unet: the UNet model
    :param x0_t: the prediction of x0 at time step t
    '''
    def __init__(self, unet, config) -> None:
        self.unet = unet
        self.config = config

    def generate_simplex_noise(self, Simplex_instance, x, t, random_param=False, octave=6, persistence=0.8, frequency=64,in_channels=1):
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
    
    def save_image(self, tensor, path):
            # print("Image saved")
            # Normalize the tensor to [0, 1]
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
            # Convert to PIL image
            image = transforms.ToPILImage()(tensor.cpu().squeeze(0))
            # Save image
            image.save(path)
    
    def __call__(self, x, y0, w) -> Any:
        def _compute_alpha(t):
            betas = np.linspace(self.config.model.beta_start, self.config.model.beta_end, self.config.model.trajectory_steps, dtype=np.float64)
            betas = torch.tensor(betas).type(torch.float).to(self.config.model.device)
            beta = torch.cat([torch.zeros(1).to(self.config.model.device), betas], dim=0)
            beta = beta.to(self.config.model.device)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
            return a
        simplex_instance = Simplex_CLASS()
        test_trajectoy_steps = torch.Tensor([self.config.model.test_trajectoy_steps]).type(torch.int64).to(self.config.model.device).long()
        at = _compute_alpha(test_trajectoy_steps)
        # noise = torch.randn_like(x).to(self.config.model.device)
        e = self.generate_simplex_noise(Simplex_instance=simplex_instance, x=x, t=test_trajectoy_steps).float()
        xt = at.sqrt() * x + (1- at).sqrt() * e
        

        # Convert tensor to PIL image and save
        def save_image(tensor, path):
            print("Image saved")
            # Normalize the tensor to [0, 1]
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
            # Convert to PIL image
            image = transforms.ToPILImage()(tensor.cpu().squeeze(0))
            # Save image
            image.save(path)

        # Save the image
        # save_image(x[0], '/content/DDADC/noise/original_image.png')
        # save_image(xt[0], '/content/DDADC/noise/fully_noised.png')

        seq = range(0 , self.config.model.test_trajectoy_steps, self.config.model.skip)

        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            xs = [xt]
            for index, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
                t = (torch.ones(n) * i).to(self.config.model.device)
                next_t = (torch.ones(n) * j).to(self.config.model.device)
                at = _compute_alpha(t.long())
                at_next = _compute_alpha(next_t.long())
                xt = xs[-1].to(self.config.model.device)
                self.unet = self.unet.to(self.config.model.device)
                et = self.unet(xt, t)
                yt = at.sqrt() * y0 + (1- at).sqrt() *  et
                et_hat = et - (1 - at).sqrt() * w * (yt-xt)
                x0_t = (xt - et_hat * (1 - at).sqrt()) / at.sqrt()
                c1 = (
                    self.config.model.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                # xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et_hat
                xt_next = at_next.sqrt() * x0_t + c1 * self.generate_simplex_noise(Simplex_instance=simplex_instance, x=x, t=next_t.long()).float() + c2 * et_hat

                xs.append(xt_next)
        return xs

         



