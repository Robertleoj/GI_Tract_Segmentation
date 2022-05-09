# Cell
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
# Need to import the named tuple types, otherwise error
from data.data import get_test_set, DataTuple, ImgTuple, label2class, segment2mask
from plotting import make_visual, plot_sample
from models.unet_first_try import Unet
import cv2 
import random
import pandas as pd
import multiprocessing
import numpy as np

# Cell
id_fmt = "case{case}_day{day}_slice_{slice:0>4}"
id_fmt.format(case=1, day=1, slice=1)

# Cell
num_cpu = multiprocessing.cpu_count()

# Cell
batch_size = 1
device = torch.device('cuda')

# Cell
# Cell
def get_test_loader(test_path, pickle_path):
    test_dataset = get_test_set(128, test_path, pickle_path)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        num_workers=max(num_cpu, 16),
        shuffle=False
    )

    return test_loader
# Cell
def get_model(model_path):
    unet = Unet(1, 64, 4, 4).to(device)
    unet.load_state_dict(torch.load(model_path, map_location=device))
    unet.eval()

    return unet


# Cell
"""
We need a list of 
    [id, class, segmentation]
This means that we need 
"""

def mask2rle(mask):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    mask = np.array(mask)
    pixels = mask.flatten()
    pad = np.array([0])
    pixels = np.concatenate([pad, pixels, pad])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


# Cell
def pred2rle(pred, width, height):
    nparr = pred.numpy()
    category_rle = {}

    msk = cv2.resize(nparr, (height.item(), width.item()), interpolation=cv2.INTER_NEAREST)

    for i in range(1, 4):
        category_rle[label2class[i]] = mask2rle(np.array(msk == i, dtype=np.int32))

    return category_rle


# Cell
def make_slice_list(test_loader, unet):
    slice_list = []
    slices = {}
    for scan, data in test_loader:
        case = data.case.item()
        day = data.day.item()

        print(f'case {case}, day {day}')
        with torch.no_grad():
            pred = unet(scan.to(device)).argmax(1).cpu().squeeze(0)
        # print(pred.shape)

        for i in range(pred.size(0)):
            slice = i + 1
            # print(f'slice={slice}')

            category_rle = pred2rle(pred[i], data.width, data.height)
            id_str = id_fmt.format(case=case, day=day, slice=slice)
            for key, val in category_rle.items():
                category = key
                slice_list.append((id_str, category, val))

                slices[id_str] = (scan[0][0][i], data.height, data.width)

    return slice_list, slices


# Cell
def show_sample(slice_list, slices):

    masked = [x for x in slice_list if x[2] != '']

    all_cases_masked = list({x[0] for x in masked})

    some_case = random.choice(all_cases_masked)
    print(some_case)

    img_t, height, width = slices[some_case]
    img_t = img_t.unsqueeze(0)
    print(img_t.shape, height, width)

    seg_dict = {x[1]: x[2] for x in slice_list if x[0] == some_case }
    print(seg_dict)
    seg = segment2mask(height.item(), width.item(), seg_dict)

    plot_sample(img_t, T.Resize((128, 128))(seg.unsqueeze(0)).squeeze(0))


def make_submission(model_path, test_path, pickle_path, sample_sub_path):
    unet = get_model(model_path)
    test_loader = get_test_loader(test_path, pickle_path)
    slice_list, slices= make_slice_list(test_loader, unet)

    df = pd.DataFrame(slice_list, columns=['id', 'class', 'predicted'])

    sub_df = pd.read_csv(sample_sub_path)

    del sub_df['predicted']
    sub_df = sub_df.merge(df, on=['id','class'])
    sub_df.to_csv('submission.csv',index=False)

    # return sub_df

# Cell
# import time
# model_path = '/media/king_rob/DataDrive/models/GI_segmentation/3d_unet_fin.pt'
# pickle_path = "/media/king_rob/DataDrive/data/GI_Tract_Image_Segment/3d_computed_data"
# test_path = "/media/king_rob/DataDrive/data/GI_Tract_Image_Segment/fake_test"
# sample_sub = "/media/king_rob/DataDrive/data/GI_Tract_Image_Segment/sample_submission.csv"
# unet = get_model(model_path)
# test_loader = get_test_loader(test_path, pickle_path)
# slice_list, slices = make_slice_list(test_loader, unet)

# make_submission(model_path, test_path, pickle_path, sample_sub)


# Cell
# show_sample(slice_list, slices)

# start = time.time()
# df = make_submission(model_path, test_path, pickle_path, sample_sub)
# end = time.time()
# seconds = end - start
# print(seconds)

# Cell
# sub_df = pd.read_csv(sample_sub)

# Cell
# del sub_df['predicted']
# sub_df = sub_df.merge(df, on=['id','class'])
# sub_df.to_csv('submission.csv',index=False)

# Cell
# df[df['predicted'] != '']
# 
# Cell
# df.to_csv('submission.csv', index=False)

# Cell
