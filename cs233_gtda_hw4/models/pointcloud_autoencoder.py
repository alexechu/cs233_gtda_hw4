"""
PC-AE.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import torch
from torch import nn
from tqdm import tqdm
from ..in_out.utils import AverageMeter

if torch.cuda.is_available():
    from ..losses.chamfer import chamfer_loss
else:
    # In the unlikely case where you cannot use the JIT chamfer implementation (above) you can use the slower
    # one that is written in pure pytorch:
    from ..losses.nn_distance import chamfer_loss #(uncomment)


class PointcloudAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        """ AE constructor.
        :param encoder: nn.Module acting as a point-cloud encoder.
        :param decoder: nn.Module acting as a point-cloud decoder.
        """
        super(PointcloudAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    # @torch.no_grad()
    def embed(self, pointclouds):
        """ Extract from the input pointclouds the corresponding latent codes.
        :param pointclouds: B x N x 3
        :return: B x latent-dimension of AE
        """
        x= self.encoder(pointclouds)
        return x

    def __call__(self, pointclouds):
        """Forward pass of the AE
            :param pointclouds: B x N x 3
        """
        return self.decoder(self.embed(pointclouds))

    def train_for_one_epoch(self, loader, optimizer, device='cuda'):
        """ Train the autoencoder for one epoch based on the Chamfer loss.
        :param loader: (train) pointcloud_dataset loader
        :param optimizer: torch.optimizer
        :param device: cuda? cpu?
        :return: (float), average loss for the epoch.
        """
        self.train()
        loss_meter = AverageMeter()
        for batch in tqdm(loader):
            if optimizer is not None: optimizer.zero_grad()
            pc = batch['point_cloud'].to(device)
            # for k, v in batch.items():
            #     if isinstance(v, torch.Tensor): print(k, v.shape)
            #     else: print(k, len(v[0]))
            recons = self.reconstruct(pc, device=device)
            loss = chamfer_loss(recons, pc).mean()
            loss_meter.update(loss, pc.size(0))
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        return loss_meter.avg

    # @torch.no_grad()
    def reconstruct(self, pointclouds, device='cuda'):
        """ Reconstruct the point-clouds via the AE.
        :param loader: pointcloud_dataset loader
        :param device: cpu? cuda?
        :return: Left for students to decide
        """
        # return self.__call__(pointclouds.to(device))
        x = self.embed((pointclouds.to(device)))
        # print('embedded', x.shape)
        x= self.decoder(x.squeeze(-1))
        # print('recon', x.shape)
        return x.transpose(-1, -2).to(device)

