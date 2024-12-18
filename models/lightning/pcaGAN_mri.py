import os

import torch
import torchvision

import pytorch_lightning as pl
import numpy as np
import torch.autograd as autograd
import sigpy as sp
from matplotlib import cm

from PIL import Image
from torch.nn import functional as F
from utils.mri.fftc import ifft2c_new, fft2c_new
from utils.mri.math import complex_abs, tensor_to_complex_np
from models.archs.mri.generator import UNetModel
from models.archs.mri.discriminator import DiscriminatorModel
from evaluation_scripts.metrics import psnr
from torchmetrics.functional import peak_signal_noise_ratio
from utils.mri.transforms import to_tensor

class pcaGAN(pl.LightningModule):
    def __init__(self, args, exp_name, num_gpus):
        super().__init__()
        self.args = args
        self.exp_name = exp_name
        self.num_gpus = num_gpus

        self.in_chans = args.in_chans + 2
        self.out_chans = args.out_chans

        self.generator = UNetModel(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
        )

        self.discriminator = DiscriminatorModel(
            in_chans=self.args.in_chans * 2,
            out_chans=self.out_chans
        )

        self.feature_extractor = vgg16(pretrained=True).eval()
        self.feature_extractor = WrapVGG(self.feature_extractor).eval()
        self.transforms = torch.nn.Sequential(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

        # self.discriminator = PatchDisc(
        #     input_nc=args.in_chans * 2
        # )

        self.std_mult = 1
        # self.std_mult_latent = 0.5
        # self.latent_weight = 1e-1
        self.beta_pca = 1e-2
        self.lam_eps = 0
        self.is_good_model = 0
        self.resolution = self.args.im_size
        self.cfid = CFIDMetric(None, None, None, None)
        self.save_hyperparameters()  # Save passed values

    def _get_embed_im(self, multi_coil_inp, mean, std, maps):
        embed_ims = torch.zeros(size=(multi_coil_inp.size(0), 3, self.args.im_size, self.args.im_size),
                                device=self.device)
        reformatted = self.reformat(multi_coil_inp * std[:, None, None, None] + mean[:, None, None, None])
        unnormal_im = reformatted

        for i in range(multi_coil_inp.size(0)):
            x_hat = torch.view_as_complex(unnormal_im[i])
            maps_complex_conj = torch.view_as_complex(maps[i]).conj()

            im = torch.sum(maps_complex_conj * x_hat, dim=0).abs()
            im = (im - torch.min(im)) / (torch.max(im) - torch.min(im))

            embed_ims[i, 0, :, :] = im
            embed_ims[i, 1, :, :] = im
            embed_ims[i, 2, :, :] = im

        return self.feature_extractor(self.transforms(embed_ims))

    def get_noise(self, num_vectors, mask):
        z = torch.randn(num_vectors, self.resolution, self.resolution, 2, device=self.device)

        return z.permute(0, 3, 1, 2)

    def reformat(self, samples):
        reformatted_tensor = torch.zeros(size=(samples.size(0), 8, self.resolution, self.resolution, 2),
                                         device=self.device)
        reformatted_tensor[:, :, :, :, 0] = samples[:, 0:8, :, :]
        reformatted_tensor[:, :, :, :, 1] = samples[:, 8:16, :, :]

        return reformatted_tensor

    def readd_measures(self, samples, measures, mask):
        reformatted_tensor = self.reformat(samples)
        measures = fft2c_new(self.reformat(measures))
        reconstructed_kspace = fft2c_new(reformatted_tensor)

        reconstructed_kspace = mask * measures + (1 - mask) * reconstructed_kspace

        image = ifft2c_new(reconstructed_kspace)

        output_im = torch.zeros(size=samples.shape, device=self.device)
        output_im[:, 0:8, :, :] = image[:, :, :, :, 0]
        output_im[:, 8:16, :, :] = image[:, :, :, :, 1]

        return output_im

    def compute_gradient_penalty(self, real_samples, fake_samples, y):
        """Calculates the gradient penalty loss for WGAN GP"""
        Tensor = torch.FloatTensor
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(input=interpolates, y=y)
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
        num_vectors = y.size(0)
        noise = self.get_noise(num_vectors, mask)
        samples = self.generator(torch.cat([y, noise], dim=1))
        samples = self.readd_measures(samples, y, mask)
        return samples

    def adversarial_loss_discriminator(self, fake_pred, real_pred):
        return fake_pred.mean() - real_pred.mean()

    def adversarial_loss_generator(self, y, gens):
        patch_out = 94
        # fake_pred = torch.zeros(size=(y.shape[0], self.args.num_z_train, patch_out, patch_out), device=self.device)
        # for k in range(y.shape[0]):
        #     cond = torch.zeros(1, gens.shape[2], gens.shape[3], gens.shape[4], device=self.device)
        #     cond[0, :, :, :] = y[k, :, :, :]
        #     cond = cond.repeat(self.args.num_z_train, 1, 1, 1)
        #     temp = self.discriminator(input=gens[k], y=cond)
        #     fake_pred[k, :, :, :] = temp[:, 0, :, :]

        fake_pred = torch.zeros(size=(y.shape[0], self.args.num_z_train), device=self.device)
        for k in range(y.shape[0]):
            cond = torch.zeros(1, gens.shape[2], gens.shape[3], gens.shape[4], device=self.device)
            cond[0, :, :, :] = y[k, :, :, :]
            cond = cond.repeat(self.args.num_z_train, 1, 1, 1)
            temp = self.discriminator(input=gens[k], y=cond)
            fake_pred[k] = temp[:, 0]

        gen_pred_loss = torch.mean(fake_pred[0])
        for k in range(y.shape[0] - 1):
            gen_pred_loss += torch.mean(fake_pred[k + 1])

        adv_weight = 1e-5
        if self.current_epoch <= 4:
            adv_weight = 1e-2
        elif self.current_epoch <= 22:
            adv_weight = 1e-4

        return - adv_weight * gen_pred_loss.mean()

    def l1_std_p(self, avg_recon, gens, x, std_mult):
        return F.l1_loss(avg_recon, x) - std_mult * np.sqrt(
            2 / (np.pi * self.args.num_z_train * (self.args.num_z_train + 1))) * torch.std(gens, dim=1).mean()

    def gradient_penalty(self, x_hat, x, y):
        gradient_penalty = self.compute_gradient_penalty(x.data, x_hat.data, y.data)

        return self.args.gp_weight * gradient_penalty

    def drift_penalty(self, real_pred):
        return 0.001 * torch.mean(real_pred ** 2)

    def training_step(self, batch, batch_idx, optimizer_idx):
        y, x, mask, mean, std, maps, _, _ = batch

        # train generator
        if optimizer_idx == 1:
            gens = torch.zeros(
                size=(y.size(0), self.args.num_z_train, self.args.in_chans, self.args.im_size, self.args.im_size),
                device=self.device)
            for z in range(self.args.num_z_train):
                gens[:, z, :, :, :] = self.forward(y, mask)

            g_loss = self.adversarial_loss_generator(y, gens)

            avg_recon_pixel = torch.mean(gens, dim=1)

            l1_std_pixel = self.l1_std_p(avg_recon_pixel, gens, x, self.std_mult)
            g_loss += l1_std_pixel
            self.log('l1_std_pixel', l1_std_pixel, prog_bar=True)

            if (self.global_step - 1) % self.args.pca_reg_freq == 0 and self.current_epoch >= 25:
                gens_embed = torch.zeros(
                    size=(y.size(0), self.args.num_z_pca, 2359296),
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

            real_pred = self.discriminator(input=x, y=y)
            fake_pred = self.discriminator(input=x_hat, y=y)

            d_loss = self.adversarial_loss_discriminator(fake_pred, real_pred)
            d_loss += self.gradient_penalty(x_hat, x, y)
            d_loss += self.drift_penalty(real_pred)

            self.log('d_loss', d_loss, prog_bar=True)

            return d_loss

    def validation_step(self, batch, batch_idx, external_test=False):
        y, x, mask, mean, std, maps, _, _ = batch

        fig_count = 0

        if external_test:
            num_code = self.args.num_z_test
        else:
            num_code = self.args.num_z_valid

        gens = torch.zeros(size=(y.size(0), 8, self.args.in_chans, self.args.im_size, self.args.im_size),
                           device=self.device)
        for z in range(num_code):
            gens[:, z, :, :, :] = self.forward(y, mask) * std[:, None, None, None] + mean[:, None, None,
                                                                                     None]  # EXPERIMENTAL UN

        avg = torch.mean(gens, dim=1)

        avg_gen = self.reformat(avg)
        gt = self.reformat(x * std[:, None, None, None] + mean[:, None, None, None])

        mag_avg_list = []
        mag_single_list = []
        mag_gt_list = []
        psnr_8s = []
        psnr_1s = []

        for j in range(y.size(0)):
            S = sp.linop.Multiply((self.args.im_size, self.args.im_size), tensor_to_complex_np(maps[j].cpu()))

            ############# EXPERIMENTAL #################
            # ON CPU
            avg_sp_out = torch.tensor(S.H * tensor_to_complex_np(avg_gen[j].cpu())).abs().unsqueeze(0).unsqueeze(0).to(
                self.device)
            single_sp_out = torch.tensor(
                S.H * tensor_to_complex_np(self.reformat(gens[:, 0])[j].cpu())).abs().unsqueeze(0).unsqueeze(0).to(
                self.device)
            gt_sp_out = torch.tensor(S.H * tensor_to_complex_np(gt[j].cpu())).abs().unsqueeze(0).unsqueeze(0).to(
                self.device)

            # ON GPU
            # avg_sp_out = complex_abs(sp.to_pytorch(S.H * sp.from_pytorch(avg_gen[j], iscomplex=True))).unsqueeze(0).unsqueeze(0)
            # single_sp_out = complex_abs(sp.to_pytorch(S.H * sp.from_pytorch(self.reformat(gens[:, 0])[j], iscomplex=True))).unsqueeze(0).unsqueeze(0)
            # gt_sp_out = complex_abs(sp.to_pytorch(S.H * sp.from_pytorch(gt[j], iscomplex=True))).unsqueeze(0).unsqueeze(0)

            psnr_8s.append(peak_signal_noise_ratio(avg_sp_out, gt_sp_out))
            psnr_1s.append(peak_signal_noise_ratio(single_sp_out, gt_sp_out))

            mag_avg_list.append(avg_sp_out)
            mag_single_list.append(single_sp_out)
            mag_gt_list.append(gt_sp_out)

        psnr_8s = torch.stack(psnr_8s)
        psnr_1s = torch.stack(psnr_1s)
        mag_avg_gen = torch.cat(mag_avg_list, dim=0)
        mag_single_gen = torch.cat(mag_single_list, dim=0)
        mag_gt = torch.cat(mag_gt_list, dim=0)

        self.log('psnr_8_step', psnr_8s.mean(), on_step=True, on_epoch=False, prog_bar=True)
        self.log('psnr_1_step', psnr_1s.mean(), on_step=True, on_epoch=False, prog_bar=True)

        img_e = self._get_embed_im(gens[:, 0, :, :, :], mean, std, maps)
        cond_e = self._get_embed_im(y, mean, std, maps)
        true_e = self._get_embed_im(x, mean, std, maps)

        ############################################

        if batch_idx == 0:
            if self.global_rank == 0 and self.current_epoch % 5 == 0 and fig_count == 0:
                fig_count += 1
                avg_gen_np = mag_avg_gen[0, 0, :, :].cpu().numpy()
                gt_np = mag_gt[0, 0, :, :].cpu().numpy()

                plot_avg_np = (avg_gen_np - np.min(avg_gen_np)) / (np.max(avg_gen_np) - np.min(avg_gen_np))
                plot_gt_np = (gt_np - np.min(gt_np)) / (np.max(gt_np) - np.min(gt_np))

                np_psnr = psnr(gt_np, avg_gen_np)

                self.logger.log_image(
                    key=f"epoch_{self.current_epoch}_img",
                    images=[Image.fromarray(np.uint8(plot_gt_np * 255), 'L'),
                            Image.fromarray(np.uint8(plot_avg_np * 255), 'L'),
                            Image.fromarray(np.uint8(cm.jet(5 * np.abs(plot_gt_np - plot_avg_np)) * 255))],
                    caption=["GT", f"Recon: PSNR (NP): {np_psnr:.2f}", "Error"]
                )

            self.trainer.strategy.barrier()

        return {'psnr_8': psnr_8s.mean(), 'psnr_1': psnr_1s.mean(), 'img_e': img_e, 'cond_e': cond_e, 'true_e': true_e}

    def validation_epoch_end(self, validation_step_outputs):
        # GATHER
        avg_psnr = self.all_gather(torch.stack([x['psnr_8'] for x in validation_step_outputs]).mean()).mean()
        avg_single_psnr = self.all_gather(torch.stack([x['psnr_1'] for x in validation_step_outputs]).mean()).mean()

        true_embed = torch.cat([x['true_e'] for x in validation_step_outputs], dim=0)
        image_embed = torch.cat([x['img_e'] for x in validation_step_outputs], dim=0)
        cond_embed = torch.cat([x['cond_e'] for x in validation_step_outputs], dim=0)

        cfid, _, _ = self.cfid.get_cfid_torch_pinv(image_embed, true_embed, cond_embed)
        cfid = self.all_gather(cfid).mean()

        self.log('cfid', cfid, prog_bar=True)

        # NO GATHER
        # avg_psnr = torch.stack([x['psnr_8'] for x in validation_step_outputs]).mean()
        # avg_single_psnr = torch.stack([x['psnr_1'] for x in validation_step_outputs]).mean()

        avg_psnr = avg_psnr.cpu().numpy()
        avg_single_psnr = avg_single_psnr.cpu().numpy()

        psnr_diff = (avg_single_psnr + 2.5) - avg_psnr
        psnr_diff = psnr_diff

        mu_0 = 2e-2
        # mu_0_latent = 2e-4
        self.std_mult += mu_0 * psnr_diff
        # self.std_mult_latent += mu_0_latent * psnr_diff

        if np.abs(psnr_diff) <= 0.25:
            self.is_good_model = 1
        else:
            self.is_good_model = 0

        if self.global_rank == 0 and self.current_epoch % 1 == 0:
            send_mail(f"EPOCH {self.current_epoch + 1} UPDATE - rcGAN",
                      f"Std. Dev. Weight: {self.std_mult:.4f}\nMetrics:\nPSNR: {avg_psnr:.2f}\nSINGLE PSNR: {avg_single_psnr:.2f}\nPSNR Diff: {psnr_diff}",
                      file_name="variation_gif.gif")

        self.trainer.strategy.barrier()

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))

        reduce_lr_on_plateau_mean = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_g,
            mode='min',
            factor=0.8,
            patience=15,
            min_lr=1e-4,
        )

        lr_scheduler = {"scheduler": reduce_lr_on_plateau_mean, "monitor": "cfid"}

        return [opt_d, opt_g], []#lr_scheduler

    def on_save_checkpoint(self, checkpoint):
        checkpoint["beta_std"] = self.std_mult
        checkpoint["is_valid"] = self.is_good_model

    def on_load_checkpoint(self, checkpoint):
        self.std_mult = checkpoint["beta_std"]
        self.is_good_model = checkpoint["is_valid"]
