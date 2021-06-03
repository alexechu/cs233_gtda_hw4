"""
Part-Aware PC-AE.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import torch
from torch import nn
from ..in_out.utils import AverageMeter

if torch.cuda.is_available():
    from ..losses.chamfer import chamfer_loss
else:
    # In the unlikely case where you cannot use the JIT chamfer implementation (above) you can use the slower
    # one that is written in pure pytorch:
    from ..losses.nn_distance import chamfer_loss # (uncomment)

class PartAwarePointcloudAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, CEL):
        """ Part-aware AE initialization
        :param encoder: nn.Module acting as a point-cloud encoder.
        :param decoder: nn.Module acting as a point-cloud decoder.
        :param part_classifier: nn.Module acting as the second decoding branch that classifies the point part
        labels.
        """
        super(PartAwarePointcloudAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.part_classifier = nn.Conv1d()
        self.CEL = CEL

    def reconstruct(self, x):
        z = self.embed((pointclouds.to(device)))
        x = self.decoder(z.squeeze(-1))
        return x.transpose(-1, -2), z
    
    def embed(self, x):
        return self.encoder(x)

    def predict(self, z):
        return self.part_classifier(z)

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
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): print(k, v.shape)
                else: print(k, len(v[0]))
            if optimizer is not None: optimizer.zero_grad()
            pc = batch['point_cloud']
            recons, z = self.reconstruct(pc, device=device)
            loss = chamfer_loss(recons, pc).mean()
            pred_loss = self.CEL(self.predict(z), batch['part_mask'])
            loss += pred_loss
            loss_meter.update(loss, pc.size(0))
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        return loss_meter.avg