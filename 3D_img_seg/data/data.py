# Cell
from sklearn.datasets import make_biclusters
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

# Cell


DataTuple = namedtuple("DataTuple", ['case', 'day', 'height', 'width', 'pixel_height', 'pixel_width'])
ImgTuple = namedtuple("ImgTuple", ['path', 'id', 'slice'])


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
    def __init__(self, df, img_size, train=True):
        self.img_size= img_size
        self.all_cases = list(df['case'].unique())

        self.pickle_folder = os.path.join(data_root, "3d_computed_data")

        self.segmentations_file = os.path.join(self.pickle_folder, "3d_segmentations.pt")
        self.day_list_file = os.path.join(self.pickle_folder, '3d_day_list.pt')
        self.segmentation_masks_folder = os.path.join(
            self.pickle_folder, "segmentation_masks"
        )

        if all((
            os.path.exists(self.segmentations_file),
            os.path.exists(self.day_list_file)
        )):
            self._load_picled_data()
            return

        # list of (datatuple, [img_tuples])
        self.day_list = []

        # mapping -> id -> segmentations
        self.segmentations = {}

        for case in self.all_cases:
            print(f'{case=}')
            # self.data[case] = {}
            case_df = df[df['case'] == case]

            # for day in self.case2days[case]:
            for day in case_df['day'].unique():
                slices = []
                case_day_df = case_df[case_df['day'] == day]

                height = case_day_df['height'].unique()[0]
                width = case_day_df['width'].unique()[0]
                pxl_height = case_day_df['pxl_height'].unique()[0]
                pxl_width = case_day_df['pxl_width'].unique()[0]


                data_t = DataTuple(
                    case=case,
                    day=day,
                    height=height,
                    width=width,
                    pixel_height=pxl_height,
                    pixel_width=pxl_width
                )

                slices = []

                for _,row in case_day_df.iterrows():
                    slices.append(
                        ImgTuple(
                            id=row.id,
                            path=row.img_path,
                            slice=row.slice
                        )
                    )

                    self.segmentations[row.id] = {}
                    for _, row in df_segmentation[df_segmentation['id'] == row.id].iterrows():
                        self.segmentations[row.id][row['class']] = row['segmentation']

                self.day_list.append((data_t, sorted(slices,key=lambda t: t.slice)))

        self._pickle_data()

    def __len__(self):
        return len(self.day_list)

    def _load_picled_data(self):
        with open(self.segmentations_file, 'rb') as f:
            self.segmentations = pickle.load(f)
        with open(self.day_list_file, 'rb') as f:
            self.day_list = pickle.load(f)

    def _pickle_data(self):
        with open(self.segmentations_file, 'wb') as f:
            pickle.dump(self.segmentations, f)
        with open(self.day_list_file, 'wb') as f:
            pickle.dump(self.day_list, f)


    def __getitem__(self, idx):
        data, imgs = self.day_list[idx]

        img_tensor = torch.zeros(len(imgs), self.img_size, self.img_size)
        mask_tensor = torch.zeros(len(imgs), self.img_size, self.img_size)
        resz = T.Resize((self.img_size, self.img_size))

        for i, img in enumerate(imgs):
            # Get the pil img
            pil_img = Image.open(img.path)

            # We don't want to return the image straignt, as it is strangely encoded.
            # We thus convert it to a tensor before returning it
            img_t = torch.tensor(np.array(pil_img, dtype=np.float32)).unsqueeze(0)

            # get the segmentation tensor
            seg_path = os.path.join(self.segmentation_masks_folder, f'{img.id}.pt')
            if not os.path.exists(seg_path):
                seg = segment2mask(data.height, data.width, self.segmentations[img.id]).unsqueeze(0)
                torch.save(seg, seg_path)
            else:
                seg = torch.load(seg_path, map_location=torch.device('cpu'))

            assert img_t.shape == seg.shape

            img_t = resz(img_t).squeeze(0)
            seg = resz(seg).squeeze(0)
            # img_t = img_t.squeeze(0)
            # seg = seg.squeeze(0)



            img_tensor[i] = img_t
            mask_tensor[i] = seg

        # Normalize
        img_tensor = (img_tensor - img_tensor.mean()) / img_tensor.std()

        return img_tensor.unsqueeze(0), data, mask_tensor.round().long()



def get_dataset(sz):
    return CTData(df, sz)

# Cell




