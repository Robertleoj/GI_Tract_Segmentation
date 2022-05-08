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

# %matplotlib widget
# plt.switch_backend('QtAgg')

# plt.ion()

# Cell

data_root = "/media/king_rob/DataDrive/data/GI_Tract_Image_Segment"

csv_file = os.path.join(data_root, "train.csv")
image_files = glob(os.path.join(data_root, "train/case*/*/scans/*.png"))


# Cell
df = pd.read_csv(csv_file)
df_segmentation = df
df = df.drop(labels=['class', 'segmentation'], axis=1).drop_duplicates()

# Cell
"""
make columns for case, day, slice
"""

df['case'] = df['id'].apply(lambda x: int(x.split('_')[0].replace("case", '')))
df['day'] = df['id'].apply(lambda x: int(x.split('_')[1].replace('day', '')))
df['slice'] = df['id'].apply(lambda x: int(x.split('_')[-1]))
df



# Cell
"""
Add filename to the df
"""
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

# Cell
df

# Cell
DataTuple = namedtuple("DataTuple", ['id','case', 'day', 'slice', 'img_path', 'height', 'width', 'pixel_height', 'pixel_width'])

# all_cases =  df['case'].unique()

# Cell
classes = ['large_bowel', 'stomach', 'small_bowel']

# index for color dimension
class_colors = {
    c: i for i, c in enumerate(classes)
}


def segment2img(height, width, segmentation):
    """
    segmentation is a dictionary {class: segmentation}
    """

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

# Cell
class CTData:
    def __init__(self, df):
        self.all_cases = list(df['case'].unique())

        self.pickle_folder = os.path.join(data_root, "computed_data")

        self.data_file = os.path.join(self.pickle_folder, "datadict.pt")
        self.segmentations_file = os.path.join(self.pickle_folder, "segmentations.pt")
        self.case2days_file = os.path.join(self.pickle_folder, 'case2days.pt')

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
            return

        # mapping case -> day -> [slices]
        self.data = {}

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

        with open(self.data_file, 'wb') as f:
            pickle.dump(self.data, f)
        with open(self.segmentations_file, 'wb') as f:
            pickle.dump(self.segmentations, f)
        with open(self.case2days_file, 'wb') as f:
            pickle.dump( self.case2days,f)
        return


    def get_random_day(self):
        rand_case = random.choice(self.all_cases)
        rand_day = random.choice(self.case2days[rand_case])
        return self.data[rand_case][rand_day]

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

    
    
                    
# Cell
ct_data = CTData(df)


# Cell
len(ct_data.data), len(ct_data.data[101][20])

# Cell
    
# Cell

def update_img(idx,im, seg, fig, imgs_t, segmentations):
    im.set(data=imgs_t[idx].permute(1, 2, 0).numpy())
    seg.set(data=segmentations[idx].permute(1, 2, 0).numpy())
    fig.canvas.draw_idle()


def make_visual(imgs_t, segmentations):
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.25)
    idx0 = 0

    im = plt.imshow(imgs_t[idx0].permute(1, 2, 0).numpy(), cmap='gray')
    seg = plt.imshow(segmentations[idx0].permute(1, 2, 0).numpy(), cmap='jet', alpha=0.25)

    allowed_values = range(imgs_t.size(0))

    ax_slice = plt.axes([0.25, 0.1, 0.65, 0.03])

    slice_slider = Slider(
        ax=ax_slice, 
        label="Slice", 
        valmin=0,
        valmax=imgs_t.size(0) -1,
        valinit=idx0,
        valstep=allowed_values,
        color="blue"
    )
    slice_slider.on_changed(lambda x: update_img(x, im, seg, fig, imgs_t, segmentations))
    plt.show()
    return slice_slider

# Cell

imgs, segs = ct_data.get_random_day_imgs()
segs.shape

# Cell
slice_slider = make_visual(*ct_data.get_random_day_imgs())
    


# Cell
case_folders = glob(os.path.join(data_root, 'train/case*'))
case_folders

# Cell
random_case = case_folders[0]

# Cell


# Cell
