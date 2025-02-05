#!/usr/bin/env python
# coding: utf-8

import os
print(os.getcwd())
import torch
import numpy as np
import os.path as osp
import tqdm
import matplotlib.pylab as plt
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import defaultdict
print(torch.__version__)
print(torch.cuda.is_available())

## Imports based on our ready-to-use code (after you pip-install the cs233_gtda_hw4 package)
from cs233_gtda_hw4.in_out.utils import make_data_loaders
from cs233_gtda_hw4.in_out.utils import save_state_dicts, load_state_dicts
from cs233_gtda_hw4.in_out import pointcloud_dataset
from cs233_gtda_hw4.in_out.plotting import plot_3d_point_cloud


## Imports you might use if you follow are scaffold code (it is OK to use your own stucture of the models)
from cs233_gtda_hw4.models import PointcloudAutoencoder
from cs233_gtda_hw4.models import PartAwarePointcloudAutoencoder
from cs233_gtda_hw4.models.point_net import PointNet
from cs233_gtda_hw4.models.mlp import MLP

part_aware_model = True
#part_aware_model = False


##
## Fixed Settings (we do not expect you to change these)
## 

n_points = 1024  # number of points of each point-cloud
n_parts = 4      # max number of parts of each shape
n_train_epochs = 400

# Students: feel free to change below -ONLY- for the bonus Question:
# I.e., use THESE hyper-parameters when you train for the non-bonus questions.

part_lambda = 0.005  # for the part-aware AE you will be using (summing) two losses:
                     # chamfer + cross-entropy
                     # do it like this: chamfer + (part_lambda * cross-entropy), 
                     # i.e. scale-down the cross-entropy
init_lr = 0.009  # initial learning-rate, tested by us with ADAM optimizer (see below)


# batch-size of data loaders
batch_size = 150 # if you can keep this too as is keep it, 
                 # but if it is too big for your GPU, feel free to change it.

# which device to use: cpu or cuda?
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # Note: only the "alternative" (slower) chamfer_loss in losses/nn_distance can run in cpu.
#device = 'cpu'
#torch.cuda.set_device(0)

top_in_dir = '../data/'
top_out_dir = '../data/out/'


# In[4]:


# PREPARE DATA:

loaders = make_data_loaders(top_in_dir, batch_size)

for split, loader in loaders.items():
    print('N-examples', split, len(loader.dataset))
    
### Student on your own:

encoder = PointNet(3)
decoder = MLP(128, n_points)
part_classifier = MLP(128, n_points)


# In[5]:


if part_aware_model:
    xentropy = nn.CrossEntropyLoss()
    model = PartAwarePointcloudAutoencoder(encoder, decoder, xentropy).to(device) # Students Work here
    model_tag = 'part_pc_ae'
else:
    model = PointcloudAutoencoder(encoder, decoder).to(device)  # Students Work here
    model_tag = 'pc_ae'


# In[6]:


optimizer = optim.Adam(model.parameters(), lr=init_lr)  # Students uncomment once you have defined your model


out_file = osp.join(top_out_dir, model_tag + '_best_model.pth')
# Load model with best per-validation loss (uncomment when ready)
best_epoch = load_state_dicts(out_file, model=model)
print('per-validation optimal epoch', best_epoch)

bs = []
for b in loaders['test']:
    bs.append(b)
testset = bs[0]
testpc=testset['point_cloud'].to(device)
z = model.embed(testpc)
embeddings = z.squeeze().detach().cpu().numpy()
np.save(f'{model_tag}_testsetembeddings.npy', embeddings)
from sklearn.manifold import TSNE
tsne = TSNE().fit_transform(embeddings)
plt.close(plt.gcf())
plt.scatter(tsne[:, 0], tsne[:, 1])
plt.savefig(f'{model_tag}_tsne1.pdf')

pred = model.predict(z, testpc)
print(pred.shape)
#pred = (-pred).exp().argmax(dim=1)
pred = F.log_softmax(pred, dim=1).argmax(dim=1)
print('accuracy:')
correct = (pred==testset['part_mask'].to(device)).sum().float()
total = float(n_points * batch_size)
print(correct/total)

# MAKE your plots and analysis

# 5 examples to visualize per questions (e, f)
examples_to_visualize = ['8a67fd47001e52414c350d7ea5fe2a3a',
                         '1e0580f443a9e6d2593ebeeedbff73b',
                         'd3562f992aa405b214b1fd95dbca05',
                         '4e8d8792a3a6390b36b0f2a1430e993a',
                         '58479a7b7c157865e68f66efebc71317']

# You can (also) use the function for the reconstructions or the part-predictions 
# (for the latter check the kwargs parameter 'c' of matplotlib.
def getidx(n):
    ds = loaders['test'].dataset
    for i in range(loaders['test'].dataset.__len__()):
        if loaders['test'].dataset.model_names[i] == n:
            return i
examples_to_visualize = [getidx(n) for n in examples_to_visualize]
vis_pcs = [torch.Tensor(loaders['test'].dataset.pointclouds[ex]) for ex in examples_to_visualize]
vis_pcs = torch.stack(vis_pcs).to(device)
vis_reconsx, z = model.reconstruct(vis_pcs, device=device)
vis_recons = vis_reconsx.detach().cpu().numpy()
colors = model.predict(z, vis_pcs)
colors = F.log_softmax(colors, dim=1).argmax(dim=1).detach().cpu().numpy()
for i,ex in enumerate(examples_to_visualize):
    plot_3d_point_cloud(loaders['test'].dataset.pointclouds[ex], show=False, c=colors[i])
    plt.savefig(f'{ex}_{model_tag}_plottedpc.pdf')
    plot_3d_point_cloud(vis_recons[i], show=False)
    plt.savefig(f'{ex}_{model_tag}_plottedrecon.pdf')



# Last, save the latent codes of the test data and go to the 
# measuring_part_awareness and tsne_plot_with_latent_codes code.

model.eval()   # Do not forget this.! We are not training any more (OK, since we do not 
               # have batch-norm, drop-out etc. this is not so important, however it is good standard 
               # practice)
latent_codes = []

# Extract the latent codes and save them, so you can analyze them later.

np.savez(osp.join(top_out_dir, model_tag +'_latent_codes'), 
         latent_codes=latent_codes, 
         test_names=test_names)


