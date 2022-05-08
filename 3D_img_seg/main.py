# Cell
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as T

from torch.utils.data import DataLoader, Subset

from data.data import get_dataset, DataTuple, ImgTuple, get_test_set
from plotting import make_visual, compare
from models.unet_first_try import Unet
import random
from datetime import datetime
import os
from tqdm import tqdm

cudnn.benchmark = True

# Cell
figure_root = "/home/king_rob/Desktop/Projects/Kaggle/GI_tract_img_segmentation/2D_image_segmentation/figures"

# Cell
batch_size = 1
device = torch.device('cuda')


# Cell
data = get_dataset(128)

# Cell
"""
Split into train and val
"""
val_size = int(0.1 * len(data))
random_indices = torch.randperm(len(data))
train_indices = random_indices[:-val_size]
val_indices = random_indices[-val_size:]

train_set = Subset(data, train_indices)
val_set = Subset(data, val_indices)
# Cell
len(train_set), len(val_set)

# Cell
# while True:
# i, d, s = random.choice(train_set)
# d : DataTuple
# print(d)
# plot_sample(i, s)


# Cell
model_path = '/media/king_rob/DataDrive/models/GI_segmentation/3d_unet_fin.pt'
unet = Unet(1, 64, 4, 4).to(device)
unet.load_state_dict(torch.load(model_path, map_location=device))


# Cell
train_loader = DataLoader(train_set, batch_size,shuffle=True, num_workers=16)
val_loader = DataLoader(val_set, batch_size, shuffle=True, num_workers=16)

# Cell
# loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.7, 1, 1, 1]).to(device))
optim = torch.optim.Adam(unet.parameters(), lr=0.0002)


# Cell
def run_through(x, y, train=True):
    if train:
        optim.zero_grad()

    with torch.set_grad_enabled(train):
        out = unet(x)
        loss = loss_fn(out, y)
        if train:
            loss.backward()
            optim.step()

    return loss, out


def train():
    unet.train()
    n_epochs = 1

    train_losses = []
    val_losses = []
    print_every = 20

    iters = 0

    best_val_loss = 100000
    val_iter = iter(val_loader)

    for i in range(n_epochs):

        print('Train')


        for x, d, y in tqdm(train_loader):

            x, y = x.to(device), y.to(device)
            iters += 1
            train_loss, _ = run_through(x, y, True)
            train_losses.append(train_loss.detach().cpu().item())

            x, y = None, None
            torch.cuda.empty_cache()


            if iters % print_every == 0:
                try:
                    x_val, d_val, y_val = next(val_iter)
                except:
                    val_iter = iter(val_loader)
                    x_val, d_val, y_val = next(val_iter)
            
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_loss, _ = run_through(x_val, y_val, False)
                val_losses.append(val_loss.cpu().item())

                print(f'\tTrain loss: {train_loss}')
                print(f'\tVal loss: {val_loss}')

                # if val_loss < best_val_loss:
                #     torch.save(unet.state_dict(), model_path)
                #     best_val_loss = val_loss.detach().cpu().item()

                x_val, y_val = None, None
                torch.cuda.empty_cache()

        print(f'Epoch {i}/{n_epochs - 1}')
        print(f'\tTrain loss: {train_loss}')
        print(f'\tVal loss: {val_loss}')
        print()

# Cell
# train()

# Cell
# model_path_fin = '/media/king_rob/DataDrive/models/GI_segmentation/3d_unet_fin.pt'

# torch.save(unet.state_dict(), model_path_fin)

# Cell

# Cell
x, y = None, None
torch.cuda.empty_cache()
# Cell
ct, d, msk = random.choice(val_set)

# Cell
unet.eval()
ct = ct.to(device).unsqueeze(0)
pred = unet(ct).argmax(1)
pred.shape, msk.shape

# Cell
pred = pred.squeeze(0).squeeze(0)

# Cell
s = compare(ct.detach().cpu().squeeze(0), msk, pred.detach().cpu())

       

# Cell
















# Cell
