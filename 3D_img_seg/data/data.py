# Cell
import torch
from torchvision import transforms as T

import pandas as pd
import numpy as np
import pickle

from PIL import Image
import os
from glob import glob

from collections import namedtuple

# Cell
DataTuple = namedtuple("DataTuple", ['case', 'day', 'height', 'width', 'pixel_height', 'pixel_width'])
ImgTuple = namedtuple("ImgTuple", ['path', 'id', 'slice'])


# Cell

classes = ['large_bowel', 'stomach', 'small_bowel']

class2label = {
    c: (i + 1) for i, c in enumerate(classes)
}
label2class = {
    v: k for k, v in class2label.items()
}


# Cell
def expand_df(df, cases_folder):
    """
    df only has one column: id
    ids are of the form
        case{x}_day{x}_slice_{x}
    """

    # Add case, day, and slice columns based on ids
    df['case'] = df['id'].apply(lambda x: int(x.split('_')[0].replace("case", '')))
    df['day'] = df['id'].apply(lambda x: int(x.split('_')[1].replace('day', '')))
    df['slice'] = df['id'].apply(lambda x: int(x.split('_')[-1]))

    # Get the image paths, the height, width, pxl_height, and pxl_width of the images
    paths = {}
    for case_path in glob(os.path.join(cases_folder, "case*")):
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

    df[['img_path', 'height', 'width', 'pxl_height','pxl_width']] = df.apply(
        lambda row: paths[(row.case, row.day, row.slice)],
        axis=1,
        result_type="expand"
    )

    return df
# Cell
def make_day_list(df):
    day_list = []
    all_cases = list(df['case'].unique())

    for case in all_cases:
        print(f'case={case}')
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

            day_list.append((data_t, sorted(slices,key=lambda t: t.slice)))
    return day_list

def segment2mask(height, width, segmentation):
    height, width = width, height
    mask = torch.zeros(height, width, dtype=torch.long)

    for c, seg in segmentation.items():
        if not isinstance(seg, str) or seg == '':
            continue
        splitted = list(map(int, seg.split(' ')))

        while splitted != []:
            pxl, len = splitted[0], splitted[1]
            splitted = splitted[2:]
            for i in range(len):
                mask[(pxl + i) // width][(pxl + i) % width] = class2label[c]
    return mask


class CTData:

    def __init__(self, df, img_size, train, segmentation_df, pickle_path):
        self.train = train
        self.pickle_folder = pickle_path

        self.img_size = img_size

        self._define_pickle_paths()

        if self._cache_exists():
            self._load_picled_data()
            return

        # list of (datatuple, [img_tuples])
        self.day_list = make_day_list(df)

        if self.train:
            assert segmentation_df is not None

            self._load_segmentations(segmentation_df)

        self._pickle_data()

    def _load_segmentations(self, segmentation_df):
        # mapping -> id -> segmentations
        self.segmentations = {}
        for _, day in self.day_list:
            for row in day:
                self.segmentations[row.id] = {}
                for _, row in segmentation_df[segmentation_df['id'] == row.id].iterrows():
                    self.segmentations[row.id][row['class']] = row['segmentation']

    def __len__(self):
        return len(self.day_list)

    def _define_pickle_paths(self):

        if self.train:
            self.segmentations_file = os.path.join(self.pickle_folder, "3d_segmentations.pt")
            self.day_list_file = os.path.join(self.pickle_folder, '3d_day_list.pt')
            self.segmentation_masks_folder = os.path.join(
                self.pickle_folder, "segmentation_masks"
            )

        else:
            self.day_list_file = os.path.join(self.pickle_folder, '3d_day_list_TEST.pt')


    def _cache_exists(self):
        if self.train:
            return all((
                os.path.exists(self.segmentations_file),
                os.path.exists(self.day_list_file)
            ))
        else:
            return os.path.exists(self.day_list_file)


    def _load_picled_data(self):
        if self.train:
            with open(self.segmentations_file, 'rb') as f:
                self.segmentations = pickle.load(f)

        with open(self.day_list_file, 'rb') as f:
            self.day_list = pickle.load(f)

    def _pickle_data(self):
        if self.train:
            with open(self.segmentations_file, 'wb') as f:
                pickle.dump(self.segmentations, f)

        with open(self.day_list_file, 'wb') as f:
            pickle.dump(self.day_list, f)


    def __getitem__(self, idx):
        data, imgs = self.day_list[idx]

        # img_tensor = torch.zeros(len(imgs), self.img_size, self.img_size)
        # mask_tensor = torch.zeros(len(imgs), self.img_size, self.img_size)
        resz = T.Resize((self.img_size, self.img_size))
        img_t = self._get_scan_tensor(resz, imgs)
        if self.train:
            mask_t = self._get_mask_tensor(data, resz, imgs)
            assert img_t.shape == mask_t.shape
            return img_t.unsqueeze(0), data, mask_t
        
        return img_t.unsqueeze(0), data

    def _get_mask_tensor(self, data, resize, imgs):
        mask_tensor = torch.zeros(len(imgs), self.img_size, self.img_size)

        for i, img in enumerate(imgs):
            # get the segmentation tensor
            seg_path = os.path.join(self.segmentation_masks_folder, f'{img.id}_{self.img_size}.pt')

            # Check if we have it in cache
            if not os.path.exists(seg_path):
                seg = segment2mask(data.height, data.width, self.segmentations[img.id]).unsqueeze(0)
                torch.save(seg, seg_path)
            else:
                seg = torch.load(seg_path, map_location=torch.device('cpu'))

            seg = resize(seg).squeeze(0)
            mask_tensor[i] = seg
        return mask_tensor.round().long()

    def _get_slice_tensor(self, resize, img):
        pil_img = Image.open(img.path)

        # We don't want to return the image straigt, as it is strangely encoded.
        # We thus convert it to a tensor before returning it
        img_t = torch.tensor(np.array(pil_img, dtype=np.float32)).unsqueeze(0)

        img_t = resize(img_t).squeeze(0)
        return img_t


    def _get_scan_tensor(self, resize, imgs):
        img_tensor = torch.zeros(len(imgs), self.img_size, self.img_size)

        for i, img in enumerate(imgs):
            img_tensor[i] = self._get_slice_tensor(resize, img)

        # Normalize
        img_tensor = (img_tensor - img_tensor.mean()) / img_tensor.std()

        return img_tensor


def get_dataset(sz, data_root, pickle_path):
    df = pd.read_csv(os.path.join(data_root, "train.csv"))
    df_segmentation = df
    df = df.drop(labels=['class', 'segmentation'], axis=1).drop_duplicates()
    df = expand_df(df, os.path.join(data_root, 'train'))

    return CTData(df, sz, True, df_segmentation, pickle_path)

# Cell
def get_test_df(cases_root):
    ids = []

    for case_path in glob(os.path.join(cases_root, 'case*')):
        case = os.path.split(case_path)[1].replace('case', '')
        for day_path in glob(os.path.join(case_path, '*')):
            day = os.path.split(day_path)[1].split('_')[1].replace('day', '')

            for scan_path in glob(os.path.join(day_path, 'scans/*.png')):
                slice_n = os.path.split(scan_path)[1].split('_')[1]
                ids.append(f'case{case}_day{day}_slice_{slice_n}')

    return pd.DataFrame(ids, columns=['id'])


def get_test_set(sz, data_path, pickle_path):
    df = get_test_df(data_path)
    df = expand_df(df, data_path)
    return CTData(df, sz, False, None, pickle_path)

# get_test_set(128)


# Cell
