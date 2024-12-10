import torch
import yaml
import types
import json
import time
import lpips

import numpy as np

from data.lightning.FFHQDataModule import FFHQDataModule

from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.pcaGAN_inpainting import pcaGAN

import matplotlib.pyplot as plt
from matplotlib import gridspec
from utils.embeddings import InceptionEmbedding
from evaluation_scripts.cfid.cfid_metric import CFIDMetric


def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    fname = 'configs/inpainting.yml'

    with open(fname, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    dm = FFHQDataModule(cfg)
    dm.setup()
    test_loader = dm.test_dataloader()
    val_loader = dm.val_dataloader()
    train_loader = dm.train_dataloader()
    inception_embedding = InceptionEmbedding()

    with torch.no_grad():
        model = pcaGAN.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint_best.ckpt')

        model.cuda()
        model.eval()

        loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
        lpips_list = []

        for i, data in enumerate(test_loader):
            y, x, mask, mean, std = data[0]
            x = x.cuda()
            y = y.cuda()
            mask = mask.cuda()
            mean = mean.cuda()
            std = std.cuda()

            sample = model(y, mask)
            lpips_val = loss_fn_vgg(sample, x)
            lpips_list.append(lpips_val.detach().cpu().numpy())

        cfid_metric = CFIDMetric(gan=model,
                                 loader=test_loader,
                                 image_embedding=inception_embedding,
                                 condition_embedding=inception_embedding,
                                 cuda=True,
                                 args=cfg,
                                 train_loader=False,
                                 num_samps=1)

        cfid_val = cfid_metric.get_cfid_torch_pinv().cpu().numpy()

        print(f'CFID: {cfid_val}')
        print(f'LPIPS: {np.mean(lpips_list)}')
