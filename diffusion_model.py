import matplotlib.pyplot as plt
import numpy as np
import evaluating_model
from helpers import *
from simplex import Simplex_CLASS


class GaussianDiffusionModel:
    def __init__(self, img_size, betas, img_channels=1, loss_type="l2", loss_weight='none', noise="gauss"):
        super().__init__()

        self.img_size = img_size
        self.img_channels = img_channels
        self.loss_type = loss_type
        self.num_timesteps = len(betas)
        self.loss_weight = loss_weight


        if noise == "gauss":
            self.noise_fn = lambda x, t: torch.randn_like(x)
        else:
            self.simplex = Simplex_CLASS()
            if noise == "simplex_randParam":
                self.noise_fn = lambda x, t: generate_simplex_noise(self.simplex, x, t, True, in_channels=img_channels)
            elif noise == "random":
                self.noise_fn = lambda x, t: random_noise(self.simplex, x, t)
            else:
                self.noise_fn = lambda x, t: generate_simplex_noise(self.simplex, x, t, False, in_channels=img_channels)

        self.weights = (
            np.arange(self.num_timesteps, 0, -1) if loss_weight == 'prop-t'
            else np.ones(self.num_timesteps) if loss_weight == 'uniform'
            else None
        )

        alphas = 1 - betas
        self.betas = betas
        self.sqrt_alphas = np.sqrt(alphas)
        self.sqrt_betas = np.sqrt(betas)
        self.alphas_cumprod = np.cumprod(alphas)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))

        self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)


    def sample_t_with_weights(self, b_size, device):
        p = self.weights / self.weights.sum()
        indices = torch.from_numpy(np.random.choice(len(p), size=b_size, p=p)).long().to(device)
        weights = torch.from_numpy((1 / len(p)) * p[indices.cpu().numpy()]).float().to(device)
        return indices, weights

    def predict_x_0_from_eps(self, x_t, t, eps):
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t - \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device) * eps

    def predict_eps_from_x_0(self, x_t, t, pred_x_0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t - pred_x_0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device)

    def q_mean_variance(self, x_0, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0
        variance = extract(1.0 - self.alphas_cumprod, t, x_0.shape, x_0.device)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_0.shape, x_0.device)
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_0, x_t, t):
        return (
            extract(self.posterior_mean_coef1, t, x_t.shape, x_t.device) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape, x_t.device) * x_t,
            extract(self.posterior_variance, t, x_t.shape, x_t.device),
            extract(self.posterior_log_variance_clipped, t, x_t.shape, x_t.device)
        )

    def p_mean_variance(self, model, x_t, t, estimate_noise=None):
        estimate_noise = model(x_t, t) if estimate_noise is None else estimate_noise
        model_var = np.append(self.posterior_variance[1], self.betas[1:])
        model_logvar = np.log(model_var)
        model_var = extract(model_var, t, x_t.shape, x_t.device)
        model_logvar = extract(model_logvar, t, x_t.shape, x_t.device)
        pred_x_0 = self.predict_x_0_from_eps(x_t, t, estimate_noise).clamp(-1, 1)
        model_mean, _, _ = self.q_posterior_mean_variance(pred_x_0, x_t, t)
        return {"mean": model_mean, "variance": model_var, "log_variance": model_logvar, "pred_x_0": pred_x_0}

    def sample_p(self, model, x_t, t, denoise_fn="gauss"):
        out = self.p_mean_variance(model, x_t, t)
        noise = {
            "gauss": torch.randn_like(x_t),
            "noise_fn": self.noise_fn(x_t, t).float(),
            "random": torch.randn_like(x_t)
        }.get(denoise_fn, generate_simplex_noise(self.simplex, x_t, t, False, in_channels=self.img_channels).float())
        if not isinstance(noise, torch.Tensor):
            noise = denoise_fn(x_t, t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_x_0": out["pred_x_0"]}

    def forward_backward(self, model, x, see_whole_sequence="half", t_distance=None, denoise_fn="gauss"):
        assert see_whole_sequence in ["whole", "half", None]
        if t_distance == 0: return x.detach()
        t_distance = self.num_timesteps if t_distance is None else t_distance
        seq = [x.cpu().detach()]
        if see_whole_sequence == "whole":
            for t in range(t_distance):
                t_batch = torch.full((x.shape[0],), t, device=x.device)
                with torch.no_grad():
                    x = self.sample_q_gradual(x, t_batch, self.noise_fn(x, t_batch).float())
                seq.append(x.cpu().detach())
        else:
            t_tensor = torch.full((x.shape[0],), t_distance - 1, device=x.device)
            x = self.sample_q(x, t_tensor, self.noise_fn(x, t_tensor).float())
            if see_whole_sequence == "half": seq.append(x.cpu().detach())
        for t in range(t_distance - 1, -1, -1):
            t_batch = torch.full((x.shape[0],), t, device=x.device)
            with torch.no_grad():
                x = self.sample_p(model, x, t_batch, denoise_fn)["sample"]
            if see_whole_sequence: seq.append(x.cpu().detach())
        return seq if see_whole_sequence else x.detach()

    def sample_q(self, x_0, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape, x_0.device) * noise)

    def sample_q_gradual(self, x_t, t, noise):
        return (extract(self.sqrt_alphas, t, x_t.shape, x_t.device) * x_t +
                extract(self.sqrt_betas, t, x_t.shape, x_t.device) * noise)

    def calc_vlb_xt(self, model, x_0, x_t, t, estimate_noise=None):
        output = self.p_mean_variance(model, x_t, t, estimate_noise)
        kl = mean_flat(normal_kl(*self.q_posterior_mean_variance(x_0, x_t, t), output["mean"], output["log_variance"])) / np.log(2.0)
        decoder_nll = mean_flat(-discretised_gaussian_log_likelihood(x_0, output["mean"], 0.5 * output["log_variance"])) / np.log(2.0)
        return {"output": torch.where((t == 0), decoder_nll, kl), "pred_x_0": output["pred_x_0"]}

    def calc_loss(self, model, x_0, t):
        noise = self.noise_fn(x_0, t).float()
        x_t = self.sample_q(x_0, t, noise)
        estimate_noise = model(x_t, t)
        l2_loss = mean_flat((estimate_noise - noise).square())
        if self.loss_type == "l1":
            loss = {"loss": mean_flat((estimate_noise - noise).abs())}
        elif self.loss_type == "l2":
            loss = {"loss": l2_loss}
        elif self.loss_type == "hybrid":
            vlb = self.calc_vlb_xt(model, x_0, x_t, t, estimate_noise)["output"]
            loss = {"vlb": vlb, "loss": vlb + l2_loss}
        else:
            loss = {"loss": l2_loss}
        return loss, x_t, estimate_noise

    def p_loss(self, model, x_0, args):
        if self.loss_weight == "none":
            if args["train_start"]:
                t = torch.randint(0, min(args["sample_distance"], self.num_timesteps), (x_0.shape[0],), device=x_0.device)
            else:
                t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=x_0.device)
            weights = 1
        else:
            t, weights = self.sample_t_with_weights(x_0.shape[0], x_0.device)
        loss, x_t, eps_t = self.calc_loss(model, x_0, t)
        return ((loss["loss"] * weights).mean(), (loss, x_t, eps_t))

    def prior_vlb(self, x_0, args):
        t = torch.full((args["Batch_Size"],), self.num_timesteps - 1, device=x_0.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_0, t)
        return mean_flat(normal_kl(qt_mean, qt_log_variance, torch.tensor(0.0, device=x_0.device), torch.tensor(0.0, device=x_0.device))) / np.log(2.0)


    def calc_total_vlb(self, x_0, model, args):
        vb, x_0_mse, mse = [], [], []
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((args["Batch_Size"],), t, device=x_0.device)
            x_t = self.sample_q(x_0, t_batch, torch.randn_like(x_0))
            with torch.no_grad():
                out = self.calc_vlb_xt(model, x_0, x_t, t_batch)
            vb.append(out["output"])
            x_0_mse.append(mean_flat((out["pred_x_0"] - x_0) ** 2))
            eps = self.predict_eps_from_x_0(x_t, t_batch, out["pred_x_0"])
            mse.append(mean_flat((eps - torch.randn_like(x_0)) ** 2))
        prior = self.prior_vlb(x_0, args)
        return {
            "total_vlb": torch.stack(vb, 1).sum(1) + prior,
            "prior_vlb": prior,
            "vb": torch.stack(vb, 1),
            "x_0_mse": torch.stack(x_0_mse, 1),
            "mse": torch.stack(mse, 1)
        }

    def detection_A(self, model, x_0, args, file, mask, total_avg=2):
        for i in [f"./diffusion-videos/ARGS={args['arg_num']}/Anomalous/{file[0]}",
                  f"./diffusion-videos/ARGS={args['arg_num']}/Anomalous/{file[0]}/{file[1]}/",
                  f"./diffusion-videos/ARGS={args['arg_num']}/Anomalous/{file[0]}/{file[1]}/A"]:
            try:
                os.makedirs(i)
            except OSError:
                pass

        for i in range(7, 0, -1):
            freq = 2 ** i
            self.noise_fn = lambda x, t: generate_simplex_noise(
                    self.simplex, x, t, False, frequency=freq,
                    in_channels=self.img_channels
                    )

            for t_distance in range(50, int(args["T"] * 0.6), 50):
                output = torch.empty((total_avg, 1, *args["img_size"]), device=x_0.device)
                for avg in range(total_avg):

                    t_tensor = torch.tensor([t_distance], device=x_0.device).repeat(x_0.shape[0])
                    x = self.sample_q(
                            x_0, t_tensor,
                            self.noise_fn(x_0, t_tensor).float()
                            )

                    for t in range(int(t_distance) - 1, -1, -1):
                        t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                        with torch.no_grad():
                            out = self.sample_p(model, x, t_batch)
                            x = out["sample"]

                    output[avg, ...] = x

                output_mean = torch.mean(output, dim=0).reshape(1, 1, *args["img_size"])
                mse = ((output_mean - x_0).square() * 2) - 1
                mse_threshold = mse > 0
                mse_threshold = (mse_threshold.float() * 2) - 1
                out = torch.cat([x_0, output[:3], output_mean, mse, mse_threshold, mask])

                temp = os.listdir(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{file[0]}/{file[1]}/A')

                plt.imshow(gridify_output(out, 4), cmap='gray')
                plt.axis('off')
                plt.savefig(
                        f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{file[0]}/{file[1]}/A/freq={i}-t'
                        f'={t_distance}-{len(temp) + 1}.png'
                        )
                plt.clf()

    def detection_B(self, model, x_0, args, file, mask, denoise_fn="gauss", total_avg=5):
        assert type(file) == tuple
        for i in [f"./diffusion-videos/ARGS={args['arg_num']}/Anomalous/{file[0]}",
                  f"./diffusion-videos/ARGS={args['arg_num']}/Anomalous/{file[0]}/{file[1]}",
                  f"./diffusion-videos/ARGS={args['arg_num']}/Anomalous/{file[0]}/{file[1]}/{denoise_fn}"]:
            try:
                os.makedirs(i)
            except OSError:
                pass
        if denoise_fn == "octave":
            end = int(args["T"] * 0.6)
            self.noise_fn = lambda x, t: generate_simplex_noise(
                    self.simplex, x, t, False, frequency=64, octave=6,
                    persistence=0.8
                    ).float()
        else:
            end = int(args["T"] * 0.8)
            self.noise_fn = lambda x, t: torch.randn_like(x)
        # multiprocessing?
        dice_coeff = []
        for t_distance in range(50, end, 50):
            output = torch.empty((total_avg, 1, *args["img_size"]), device=x_0.device)
            for avg in range(total_avg):

                t_tensor = torch.tensor([t_distance], device=x_0.device).repeat(x_0.shape[0])
                x = self.sample_q(
                        x_0, t_tensor,
                        self.noise_fn(x_0, t_tensor).float()
                        )

                for t in range(int(t_distance) - 1, -1, -1):
                    t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                    with torch.no_grad():
                        out = self.sample_p(model, x, t_batch)
                        x = out["sample"]

                output[avg, ...] = x

            # save image containing initial, each final denoised image, mean & mse
            output_mean = torch.mean(output, dim=[0]).reshape(1, 1, *args["img_size"])

            temp = os.listdir(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{file[0]}/{file[1]}/{denoise_fn}')

            dice = evaluating_model.heatmap(
                    real=x_0, recon=output_mean, mask=mask,
                    filename=f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{file[0]}/{file[1]}/'
                             f'{denoise_fn}/heatmap-t={t_distance}-{len(temp) + 1}.png'
                    )

            mse = ((output_mean - x_0).square() * 2) - 1
            mse_threshold = mse > 0
            mse_threshold = (mse_threshold.float() * 2) - 1
            out = torch.cat([x_0, output[:3], output_mean, mse, mse_threshold, mask])

            plt.imshow(gridify_output(out, 4), cmap='gray')
            plt.axis('off')
            plt.savefig(
                    f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{file[0]}/{file[1]}/{denoise_fn}/t'
                    f'={t_distance}-{len(temp) + 1}.png'
                    )
            plt.clf()

            dice_coeff.append(dice)
        return dice_coeff
    
    def normalize_image(self, image):
        image = image.to(torch.float32)
        min_val, max_val = image.min(), image.max()

        # Skip normalization if already in [0,1]
        if min_val >= 0 and max_val <= 1:
            return image
    
        return (image - min_val) / (max_val - min_val)


    def detection_A_fixedT(self, model, x_0, args, mask, end_freq=6):
        t_distance = 550

        result = torch.empty((6 * end_freq, 1, *args["img_size"]), device=x_0.device)
        for i in range(1, end_freq + 1):

            freq = 2 ** i
            noise_fn = lambda x, t: generate_simplex_noise(self.simplex, x, t, False, frequency=freq).float()

            t_tensor = torch.tensor([t_distance - 1], device=x_0.device).repeat(x_0.shape[0])
            x = self.sample_q(
                    x_0, t_tensor,
                    noise_fn(x_0, t_tensor).float()
                    )
            x_noised = x.clone().detach()
            for t in range(int(t_distance) - 1, -1, -1):
                t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                with torch.no_grad():
                    out = self.sample_p(model, x, t_batch, denoise_fn=noise_fn)
                    x = out["sample"]

            image = self.normalize_image(x_0)
            output = self.normalize_image(x)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            image = image.to(device)
            output = output.to(device)


            mse = ((image - output).square() * 2) - 1
            threshold = 0.5
            mse_threshold = (mse > ((threshold * 2) - 1)).float()
            
            result[(i - 1) * 6:i * 6, ...] = torch.cat((x_0, x_noised, x, mse, mse_threshold, mask))

        return result

