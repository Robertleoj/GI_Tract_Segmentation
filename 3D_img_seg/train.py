# Cell
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from data.data import get_dataset, DataTuple, ImgTuple
from plotting import compare
from models.unet_first_try import Unet
import random
from tqdm import tqdm
import os

cudnn.benchmark = True
# Cell
DATA_ROOT = "/media/king_rob/DataDrive/data/GI_Tract_Image_Segment"
PICKLE_PATH = "/media/king_rob/DataDrive/data/GI_Tract_Image_Segment/3d_computed_data"
# Cell
batch_size = 1
device = torch.device('cuda')


# Cell
data = get_dataset(128, DATA_ROOT, PICKLE_PATH)

# Cell
"""
Split into train and val
"""
val_size = int(0.07 * len(data))
# random_indices = torch.randperm(len(data))
# torch.save(random_indices, os.path.join(PICKLE_PATH, 'index_shuffle.pt'))
random_indices = torch.load(os.path.join(PICKLE_PATH, 'index_shuffle.pt'))
train_indices = random_indices[:-val_size]
val_indices = random_indices[-val_size:]

train_set = Subset(data, train_indices)
val_set = Subset(data, val_indices)
# Cell
len(train_set), len(val_set)


# Cell
"""
Get the saved model
"""
model_path = '/media/king_rob/DataDrive/models/GI_segmentation/3d_unet_fin_v3.pt'
unet = Unet(1, 64, 4, 4).to(device)
unet.load_state_dict(torch.load(model_path, map_location=device))

# Cell
"""
Make dataloaders
"""
train_loader = DataLoader(train_set, batch_size,shuffle=True, num_workers=16)
val_loader = DataLoader(val_set, batch_size, shuffle=True, num_workers=16)

# Cell
"""
loss and optim
"""
# Put a smaller weight on the "none" segmentation to make the model less shy of classifying
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.64, 1, 1, 1]).to(device))
optim = torch.optim.Adam(unet.parameters(), lr=0.0002)


# Cell
"""
Training
"""
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
                    x_val, _, y_val = next(val_iter)
                except:
                    val_iter = iter(val_loader)
                    x_val, _, y_val = next(val_iter)
            
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
"""
Save the model
"""
# model_path_fin = '/media/king_rob/DataDrive/models/GI_segmentation/3d_unet_fin_v3.pt'

# torch.save(unet.state_dict(), model_path_fin)

# Cell
"""
Free some GPU memory
"""
x, y = None, None
torch.cuda.empty_cache()
# Cell
"""
Get a random scan from the validation set
"""
ct, d, msk = random.choice(val_set)

# Cell
"""
prepare the results for comparision
"""
unet.eval()
ct = ct.to(device).unsqueeze(0)
pred = unet(ct).argmax(1)
pred.shape, msk.shape
pred = pred.squeeze(0).squeeze(0)

# Cell
s = compare(ct.detach().cpu().squeeze(0), msk, pred.detach().cpu())


# Cell
