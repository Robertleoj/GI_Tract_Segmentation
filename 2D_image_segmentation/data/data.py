# Cell
import torch
from torchvision import transforms as T
from torchvision.utils import make_grid
# from torch.utils.data import Dat

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pickle

from PIL import Image
import os
from glob import glob
import random
import math

from collections import namedtuple


# Cell

classes = ['large_bowel', 'stomach', 'small_bowel']

class2label = {
    c: (i + 1) for i, c in enumerate(classes)
}
label2class = {
    v: k for k, v in class2label.items()
}



data_root = "/media/king_rob/DataDrive/data/GI_Tract_Image_Segment"

csv_file = os.path.join(data_root, "train.csv")
image_files = glob(os.path.join(data_root, "train/case*/*/scans/*.png"))

df = pd.read_csv(csv_file)
df_segmentation = df
df = df.drop(labels=['class', 'segmentation'], axis=1).drop_duplicates()

df['case'] = df['id'].apply(lambda x: int(x.split('_')[0].replace("case", '')))
df['day'] = df['id'].apply(lambda x: int(x.split('_')[1].replace('day', '')))
df['slice'] = df['id'].apply(lambda x: int(x.split('_')[-1]))

paths = {}
for case_path in glob(os.path.join(data_root, "train/case*")):
    case = int(os.path.split(case_path)[1].replace('case', ''))
    # print(case)
    for day_path in glob(os.path.join(case_path, "*")):
        day = int(os.path.split(day_path)[1].split('_')[-1].replace('day', ''))
        # print(f'\t{day}')
        for img_path in glob(os.path.join(day_path, "scans/*")):
            img_nm = os.path.splitext(os.path.split(img_path)[1])[0]
            splitted = img_nm.split('_')
            slice_n = int(splitted[1])
            height = int(splitted[2])
            width = int(splitted[3])
            pxl_height = float(splitted[4])
            pxl_width = float(splitted[5])

            paths[(case, day, slice_n)] = (
                img_path, height, width, pxl_height, pxl_width
            )

# Cell
df[['img_path', 'height', 'width', 'pxl_height','pxl_width']] = df.apply(
    lambda row: paths[(row.case, row.day, row.slice)],
    axis=1,
    result_type="expand"
)


DataTuple = namedtuple("DataTuple", ['id','case', 'day', 'slice', 'img_path', 'height', 'width', 'pixel_height', 'pixel_width'])


def segment2img(height, width, segmentation):
    """
    segmentation is a dictionary {class: segmentation}
    """
    height, width = width, height

    masks = {c : torch.zeros(height, width) for c in classes}

    for c, seg in segmentation.items():
        if not isinstance(seg, str): # must be nan then
            continue
        splitted = list(map(int, seg.split(' ')))
        while splitted != []:
            pxl, len = splitted[0], splitted[1]
            splitted = splitted[2:]
            for i in range(len):
                masks[c][(pxl + i) // width][(pxl + i) % width] = 1

    return masks

def segment2mask(height, width, segmentation):
    height, width = width, height
    mask = torch.zeros(height, width, dtype=torch.long)

    for c, seg in segmentation.items():
        if not isinstance(seg, str):
            continue
        splitted = list(map(int, seg.split(' ')))

        while splitted != []:
            pxl, len = splitted[0], splitted[1]
            splitted = splitted[2:]
            for i in range(len):
                mask[(pxl + i) // width][(pxl + i) % width] = class2label[c]
    return mask


class CTData:
    def __init__(self, df):
        self.all_cases = list(df['case'].unique())

        self.pickle_folder = os.path.join(data_root, "computed_data")

        self.data_file = os.path.join(self.pickle_folder, "datadict.pt")
        self.segmentations_file = os.path.join(self.pickle_folder, "segmentations.pt")
        self.case2days_file = os.path.join(self.pickle_folder, 'case2days.pt')
        self.data_list_file = os.path.join(self.pickle_folder, 'data_list.pt')
        self.segmentation_masks_folder = os.path.join(
            self.pickle_folder, "segmentation_masks"
        )

        if all((
            os.path.exists(self.data_file), 
            os.path.exists(self.segmentations_file),
            os.path.exists(self.case2days_file)
        )):
            with open(self.data_file, 'rb') as f:
                self.data = pickle.load(f)
            with open(self.segmentations_file, 'rb') as f:
                self.segmentations = pickle.load(f)
            with open(self.case2days_file, 'rb') as f:
                self.case2days = pickle.load(f)
            with open(self.data_list_file, 'rb') as f:
                self.data_list = pickle.load(f)
            return

        # mapping case -> day -> [slices]
        self.data = {}
        self.data_list = []

        # mapping -> id -> segmentations
        self.segmentations = {}

        self.case2days = {}


        for case in self.all_cases:
            print(f'{case=}')
            self.data[case] = {}
            self.case2days[case] = list(df[df['case'] == case]['day'].unique())
            for day in self.case2days[case]:
                slices = []
                for _,row in df[(df['case']== case) & (df['day'] == day)].iterrows():
                    slices.append(
                        DataTuple(
                            id=row.id,
                            case=case,
                            day=day,
                            slice=row.slice,
                            img_path=row.img_path,
                            height=row.height,
                            width=row.width,
                            pixel_height=row.pxl_height,
                            pixel_width=row.pxl_width
                        )
                    )

                    self.segmentations[row.id] = {}
                    for _, row in df_segmentation[df_segmentation['id'] == row.id].iterrows():
                        self.segmentations[row.id][row['class']] = row['segmentation']

                self.data[case][day] = sorted(slices,key=lambda t: t.slice)

                self.data_list.extend(sorted(slices,key=lambda t: t.slice))

        with open(self.data_file, 'wb') as f:
            pickle.dump(self.data, f)
        with open(self.segmentations_file, 'wb') as f:
            pickle.dump(self.segmentations, f)
        with open(self.case2days_file, 'wb') as f:
            pickle.dump( self.case2days,f)
        return

    def __len__(self):
        return len(self.data_list)


    def get_random_day(self):
        rand_case = random.choice(self.all_cases)
        rand_day = random.choice(self.case2days[rand_case])
        return self.data[rand_case][rand_day]

    def __getitem__(self, idx):
        data : DataTuple = self.data_list[idx]

        # Get the pil img
        pil_img = Image.open(data.img_path)

        # We don't want to return the image straignt, as it is strangely encoded.
        # We thus convert it to a tensor before returning it
        img_t = torch.tensor(np.array(pil_img, dtype=np.float32)).unsqueeze(0)

        # get the segmentation tensor
        seg_path = os.path.join(self.segmentation_masks_folder, f'{data.id}.pt')
        if not os.path.exists(seg_path):
            seg = segment2mask(data.height, data.width, self.segmentations[data.id]).unsqueeze(0)
            torch.save(seg, seg_path)
        else:
            seg = torch.load(seg_path, map_location=torch.device('cpu'))

        assert img_t.shape == seg.shape

        # match img_t.shape[1], img_t.shape[2]:
            # case 266, 266:
        resz = T.Resize((256, 256))
        img_t = resz(img_t)
        seg = resz(seg).squeeze(0)

        return img_t, data, seg.long()



    def get_random_day_imgs(self):
        rand_day = self.get_random_day()

        imgs = []
        segmentations = []
        for slice in rand_day:
            img_path = slice.img_path
            img = np.expand_dims(np.array(Image.open(img_path), dtype=np.float32), -1)
            imgs.append(img)

            segmentation = self.segmentations[slice.id]
            segmentation_imgs = segment2img(slice.width, slice.height, segmentation)
            seg_img = torch.cat([segmentation_imgs[c].unsqueeze(0) for c in classes], dim=0)
            segmentations.append(seg_img)

        seg_img_block_t = torch.cat([s.unsqueeze(0) for s in segmentations])

        img_block_t = torch.tensor(np.array(imgs)).permute(0, 3, 1, 2)

        imgs_t_n = (img_block_t - img_block_t.mean()) / img_block_t.std()

        return imgs_t_n, seg_img_block_t


def get_dataset():
    return CTData(df)

# Cell
# ct = CTData(df)

# Cell
# ct[7]

# Cell
def plot_sample(img_t, seg_mask):
    fig, ax = plt.subplots(figsize=(10, 10))
    # plt.subplots_adjust(bottom=0.25)
    # idx0 = 0

    im = plt.imshow(img_t.permute(1, 2, 0).numpy(), cmap='gray')
    seg_img = np.zeros((img_t.size(1), img_t.size(2), 3))
    for i in range(img_t.size(1)):
        for j in range(img_t.size(2)):
            lbl = seg_mask[i][j]
            if lbl == 0:
                continue

            seg_img[i][j][int(lbl) - 1] = 1

    seg = plt.imshow(seg_img, cmap='jet', alpha=0.25)

    plt.show()

def plot_pair(img_t, real_mask, pred_mask, file=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    # Show scan on both axes
    for ax, msk in zip(axs, (real_mask, pred_mask)):
        ax.imshow(img_t.permute(1, 2, 0).numpy(), cmap='gray')

        seg_img = np.zeros((img_t.size(1), img_t.size(2), 3))
        for i in range(img_t.size(1)):
            for j in range(img_t.size(2)):
                lbl = msk[i][j]
                if lbl == 0:
                    continue

                seg_img[i][j][int(lbl) - 1] = 1

        ax.imshow(seg_img, cmap='jet', alpha=0.25)

    axs[0].set_title("Real")
    axs[1].set_title("Predicted")

    if file is None:
        plt.show()
    else:
        plt.savefig(file)
        plt.close()



# Cell


# Cell
# i, d, s = ct[random.randint(0, len(ct))]
# plot_sample(i, s)

# Cell
