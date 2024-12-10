import os

import torch
import torchvision

import pytorch_lightning as pl
import numpy as np
import torch.autograd as autograd
import sigpy as sp
from matplotlib import cm

from torchvision.models.inception import inception_v3
from utils.embeddings import WrapInception
from PIL import Image
from torch.nn import functional as F
from models.archs.inpainting.co_mod_gan import Generator, Discriminator
from evaluation_scripts.cfid.cfid_metric import CFIDMetric
from torchmetrics.functional import peak_signal_noise_ratio

class pcaGAN(pl.LightningModule):
    def __init__(self, args, exp_name, num_gpus):
        super().__init__()
        self.args = args
        self.exp_name = exp_name
        self.num_gpus = num_gpus

        self.in_chans = args.in_chans
        self.out_chans = args.out_chans

        self.generator = Generator(self.args.im_size)
        self.discriminator = Discriminator(self.args.im_size)

        self.feature_extractor = inception_v3(pretrained=True, transform_input=False)
        self.feature_extractor = WrapInception(self.feature_extractor.eval()).eval()

        self.std_mult = 1
        self.beta_pca = 1e-4
        self.lam_eps = 0

        self.cfid = CFIDMetric(None, None, None, None)

        self.is_good_model = 0
        self.resolution = self.args.im_size

        self.save_hyperparameters()  # Save passed values

    def get_noise(self, num_vectors):
        z = [torch.randn(num_vectors, 512, device=self.device)]
        return z

    def get_embed_im(self, inp, mean, std):
        embed_ims = torch.zeros(size=(inp.size(0), 3, 256, 256),
                                device=self.device)
        for i in range(inp.size(0)):
            im = inp[i, :, :, :] * std[i, :, None, None] + mean[i, :, None, None]
            im = 2 * (im - torch.min(im)) / (torch.max(im) - torch.min(im)) - 1
            embed_ims[i, :, :, :] = im

        return self.feature_extractor(embed_ims)

    def compute_gradient_penalty(self, real_samples, fake_samples, y):
        """Calculates the gradient penalty loss for WGAN GP"""
        Tensor = torch.FloatTensor
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(input=interpolates, label=y)
        # fake = Tensor(real_samples.shape[0], 1, d_interpolates.shape[-1], d_interpolates.shape[-1]).fill_(1.0).to(
        #     self.device)
        fake = Tensor(real_samples.shape[0], 1).fill_(1.0).to(
            self.device)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def forward(self, y, mask):
        noise = self.get_noise(y.size(0))
        return self.generator(y, mask, noise)

    def adversarial_loss_discriminator(self, fake_pred, real_pred):
        return fake_pred.mean() - real_pred.mean()

    def adversarial_loss_generator(self, y, gens):
        fake_pred = torch.zeros(size=(y.shape[0], self.args.num_z_train), device=self.device)
        for k in range(y.shape[0]):
            cond = torch.zeros(1, gens.shape[2], gens.shape[3], gens.shape[4], device=self.device)
            cond[0, :, :, :] = y[k, :, :, :]
            cond = cond.repeat(self.args.num_z_train, 1, 1, 1)
            temp = self.discriminator(input=gens[k], label=cond)
            fake_pred[k] = temp[:, 0]

        gen_pred_loss = torch.mean(fake_pred[0])
        for k in range(y.shape[0] - 1):
            gen_pred_loss += torch.mean(fake_pred[k + 1])

        return - self.args.adv_weight * gen_pred_loss.mean()

    def l1_std_p(self, avg_recon, gens, x):
        return F.l1_loss(avg_recon, x) - self.std_mult * np.sqrt(
            2 / (np.pi * self.args.num_z_train * (self.args.num_z_train+ 1))) * torch.std(gens, dim=1).mean()

    def gradient_penalty(self, x_hat, x, y):
        gradient_penalty = self.compute_gradient_penalty(x.data, x_hat.data, y.data)

        return self.args.gp_weight * gradient_penalty

    def drift_penalty(self, real_pred):
        return 0.001 * torch.mean(real_pred ** 2)

    def training_step(self, batch, batch_idx, optimizer_idx):
        y, x, mask, mean, std = batch[0]

        # train generator
        if optimizer_idx == 1:
            gens = torch.zeros(
                size=(y.size(0), self.args.num_z_train, 3, self.args.im_size, self.args.im_size),
                device=self.device)
            for z in range(self.args.num_z_train):
                gens[:, z, :, :, :] = self.forward(y, mask)

            avg_recon = torch.mean(gens, dim=1)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss_generator(y, gens)
            g_loss += self.l1_std_p(avg_recon, gens, x)

            if (self.global_step - 1) % self.args.pca_reg_freq == 0 and self.current_epoch >= 25:
                gens_embed = torch.zeros(
                    size=(y.size(0), self.args.num_z_pca, 196608),
                    device=self.device)
                for z in range(self.args.num_z_pca):
                    gens_embed[:, z, :] = self.forward(y, mask).view(y.size(0), -1)

                gens_zm = gens_embed - torch.mean(gens_embed, dim=1)[:, None, :].clone().detach()
                gens_zm = gens_zm.view(gens_embed.shape[0], self.args.num_z_pca, -1)

                x_zm = x.view(y.size(0), -1) - torch.mean(gens_embed, dim=1).clone().detach()
                x_zm = x_zm.view(gens_embed.shape[0], -1)

                w_loss = 0
                sig_loss = 0

                for n in range(gens_zm.shape[0]):
                    _, S, Vh = torch.linalg.svd(gens_zm[n], full_matrices=False)

                    current_x_xm = x_zm[n, :]
                    inner_product = torch.sum(Vh * current_x_xm[None, :], dim=1)

                    w_obj = inner_product ** 2
                    w_loss += 1 / (torch.norm(current_x_xm, p=2) ** 2 * (self.args.num_z_pca // 10)).detach() * w_obj[
                                                                                                                0:self.args.num_z_pca // 10].sum()  # 1e-3 for 25 iters

                    gens_zm_det = gens_zm[n].detach()
                    gens_zm_det[0, :] = x_zm[n, :].view(-1).detach()

                    if self.current_epoch >= 50:
                        inner_product_mat = 1 / self.args.num_z_pca * torch.matmul(Vh, torch.matmul(
                            torch.transpose(gens_zm_det.clone().detach(), 0, 1),
                            torch.matmul(gens_zm_det.clone().detach(), Vh.mT)))

                        # cfg 1
                        sig_diff = 1 / (torch.norm(current_x_xm, p=2) ** 2 * (self.args.num_z_pca // 10)).detach() * (
                                    1 - 1 / (S ** 2 + self.lam_eps) * torch.diag(
                                inner_product_mat.clone().detach())) ** 2

                        sig_loss += self.beta_pca * sig_diff[0:self.args.num_z_pca // 10].sum()

                w_loss_g = - self.beta_pca * w_loss
                self.log('w_loss', w_loss_g, prog_bar=True)
                self.log('sig_loss', sig_loss, prog_bar=True)
                g_loss += w_loss_g
                g_loss += sig_loss

            self.log('g_loss', g_loss, prog_bar=True)

            return g_loss

        # train discriminator
        if optimizer_idx == 0:
            x_hat = self.forward(y, mask)

            real_pred = self.discriminator(input=x, label=y)
            fake_pred = self.discriminator(input=x_hat, label=y)

            d_loss = self.adversarial_loss_discriminator(fake_pred, real_pred)
            d_loss += self.gradient_penalty(x_hat, x, y)
            d_loss += self.drift_penalty(real_pred)

            self.log('d_loss', d_loss, prog_bar=True)

            return d_loss

    def validation_step(self, batch, batch_idx, external_test=False):
        y, x, mask, mean, std = batch[0]

        fig_count = 0

        if external_test:
            num_code = self.args.num_z_test
        else:
            num_code = self.args.num_z_valid

        gens = torch.zeros(size=(y.size(0), num_code, 3, self.args.im_size, self.args.im_size),
                           device=self.device)
        for z in range(num_code):
            gens[:, z, :, :, :] = self.forward(y, mask) * std[:, :, None, None] + mean[:, :, None, None]

        avg_gen = torch.mean(gens, dim=1)
        single_gen = gens[:, 0, :, :, :]
        gt = x * std[:, :, None, None] + mean[:, :, None, None]

        psnr_8s = []
        psnr_1s = []

        for j in range(y.size(0)):
            psnr_8s.append(peak_signal_noise_ratio(avg_gen[j], gt[j]))
            psnr_1s.append(peak_signal_noise_ratio(single_gen[j], gt[j]))

        psnr_8s = torch.stack(psnr_8s)
        psnr_1s = torch.stack(psnr_1s)

        self.log('psnr_8_step', psnr_8s.mean(), on_step=True, on_epoch=False, prog_bar=True)
        self.log('psnr_1_step', psnr_1s.mean(), on_step=True, on_epoch=False, prog_bar=True)

        img_e = self.get_embed_im(gens[:, 0, :, :, :], mean, std)
        cond_e = self.get_embed_im(y, mean, std)
        true_e = self.get_embed_im(x, mean, std)

        if batch_idx == 0:
            if self.global_rank == 0 and self.current_epoch % 5 == 0 and fig_count == 0:
                fig_count += 1
                samp_1_np = gens[0, 0, :, :, :].cpu().numpy()
                samp_2_np = gens[0, 1, :, :, :].cpu().numpy()
                samp_3_np = gens[0, 2, :, :, :].cpu().numpy()
                gt_np = gt[0].cpu().numpy()
                y_np = (y * std[:, :, None, None] + mean[:, :, None, None])[0].cpu().numpy()

                plot_gt_np = gt_np

                self.logger.log_image(
                    key=f"epoch_{self.current_epoch}_img",
                    images=[
                        Image.fromarray(np.uint8(np.transpose(plot_gt_np, (1, 2, 0))*255), 'RGB'),
                        Image.fromarray(np.uint8(np.transpose(y_np, (1, 2, 0)) * 255), 'RGB'),
                        Image.fromarray(np.uint8(np.transpose(samp_1_np, (1, 2, 0))*255), 'RGB'),
                        Image.fromarray(np.uint8(np.transpose(samp_2_np, (1, 2, 0)) * 255), 'RGB'),
                        Image.fromarray(np.uint8(np.transpose(samp_3_np, (1, 2, 0)) * 255), 'RGB')
                    ],
                    caption=["x", "y", "Samp 1", "Samp 2", "Samp 3"]
                )

            self.trainer.strategy.barrier()

        return {'psnr_8': psnr_8s.mean(), 'psnr_1': psnr_1s.mean(), 'img_e': img_e, 'cond_e': cond_e, 'true_e': true_e}

    def validation_epoch_end(self, validation_step_outputs):
        avg_psnr = self.all_gather(torch.stack([x['psnr_8'] for x in validation_step_outputs]).mean()).mean()
        avg_single_psnr = self.all_gather(torch.stack([x['psnr_1'] for x in validation_step_outputs]).mean()).mean()

        true_embed = torch.cat([x['true_e'] for x in validation_step_outputs], dim=0)
        image_embed = torch.cat([x['img_e'] for x in validation_step_outputs], dim=0)
        cond_embed = torch.cat([x['cond_e'] for x in validation_step_outputs], dim=0)

        cfid = self.cfid.get_cfid_torch_pinv(y_predict=image_embed, x_true=cond_embed, y_true=true_embed)
        cfid = self.all_gather(cfid).mean()

        self.log('cfid', cfid, prog_bar=True)

        avg_psnr = avg_psnr.cpu().numpy()
        avg_single_psnr = avg_single_psnr.cpu().numpy()

        psnr_diff = (avg_single_psnr + 2.5) - avg_psnr
        psnr_diff = psnr_diff

        mu_0 = 2e-2
        self.std_mult += mu_0 * psnr_diff

        if np.abs(psnr_diff) <= self.args.psnr_gain_tol:
            self.is_good_model = 1
        else:
            self.is_good_model = 0

        self.trainer.strategy.barrier()

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))

        reduce_lr_on_plateau_mean = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_g,
            mode='min',
            factor=0.9,
            patience=10,
            min_lr=3e-5,
        )

        lr_scheduler = {"scheduler": reduce_lr_on_plateau_mean, "monitor": "cfid"}

        return [opt_d, opt_g], lr_scheduler

    def on_save_checkpoint(self, checkpoint):
        checkpoint["beta_std"] = self.std_mult
        checkpoint["is_valid"] = self.is_good_model

    def on_load_checkpoint(self, checkpoint):
        self.std_mult = checkpoint["beta_std"]
        self.is_good_model = checkpoint["is_valid"]
