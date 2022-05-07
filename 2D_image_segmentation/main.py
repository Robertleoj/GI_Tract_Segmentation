# Cell
from re import A
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as T

from torch.utils.data import DataLoader, Subset

from data.data import get_dataset, plot_sample, DataTuple, plot_pair
from models.unet_first_try import Unet
import random
from datetime import datetime
import os
from tqdm import tqdm

cudnn.benchmark = True

# Cell
figure_root = "/home/king_rob/Desktop/Projects/Kaggle/GI_tract_img_segmentation/2D_image_segmentation/figures"

# Cell
batch_size = 16
device = torch.device('cuda')


# Cell
data = get_dataset()

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
unet = Unet(1, 64, 4, 4).to(device)

# Cell
train_loader = DataLoader(train_set, batch_size,shuffle=True, num_workers=16)
val_loader = DataLoader(val_set, batch_size, shuffle=True, num_workers=16)

# Cell
# loss_fn = nn.CrossEntropyLoss(ignore_index=0)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1, 1, 1]).to(device))
optim = torch.optim.Adam(unet.parameters(), lr=0.0002)

model_path = '/media/king_rob/DataDrive/models/GI_segmentation/2d_unet.pt'

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

n_epochs = 1

train_losses = []
val_losses = []
print_every = 30
plot_every = 30

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

        try:
            x_val, d_val, y_val = next(val_iter)
        except:
            val_iter = iter(val_loader)
            x_val, d_val, y_val = next(val_iter)

        x_val, y_val = x_val.to(device), y_val.to(device)
        val_loss, pred = run_through(x_val, y_val, False)
        val_losses.append(val_loss.cpu().item())

        if val_loss < best_val_loss:
            torch.save(unet.state_dict(), model_path)
            best_val_loss = val_loss.detach().cpu().item()

        if iters % print_every == 0:
            print(f'\tTrain loss: {train_loss}')
            print(f'\tVal loss: {val_loss}')

        if iters % plot_every == 0:
            for i in range(5):
                img, msk, predicted = x_val[i].cpu(), y_val[i].cpu(), pred.argmax(1)[i].cpu()
                file = os.path.join(figure_root, f'{str(datetime.now())}.jpg')
                plot_pair(img, msk, predicted, file = file)

    # print("Val")
    # for x, d, y in tqdm(val_loader):
    #     x, y = x.to(device), y.to(device)
    #     val_loss, pred = run_through(x, y, False)
    #     val_losses.append(val_loss.cpu().item())

    print(f'Epoch {i}/{n_epochs - 1}')
    print(f'\tTrain loss: {train_loss}')
    print(f'\tVal loss: {val_loss}')
    print()

    img, msk, predicted = x[0].cpu(), y[0].cpu(), pred.argmax(1)[0].cpu()
    file = os.path.join(figure_root, f'{str(datetime.now())}.jpg')
    plot_pair(img, msk, predicted, file = file)



    
       
    















# Cell
