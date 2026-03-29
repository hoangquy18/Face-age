import torch.utils.data as tordata
import os.path as osp
import numpy as np
from torchvision.datasets.folder import pil_loader
import pandas as pd
import random

from common.ops import age2group


class BaseImageDataset(tordata.Dataset):
    def __init__(self, dataset_name, transforms=None, data_root=None, list_path=None):
        self.transforms = transforms
        default_pack_root = osp.join(osp.dirname(osp.dirname(__file__)), "dataset")

        if list_path is not None:
            list_file = osp.abspath(list_path)
        else:
            list_base = osp.abspath(data_root) if data_root is not None else default_pack_root
            list_file = osp.join(list_base, "{}.txt".format(dataset_name))

        if data_root is not None:
            self.root = osp.abspath(data_root)
        else:
            # Ảnh cùng thư mục với file .txt (repo / layout cũ)
            self.root = osp.dirname(list_file)

        df = pd.read_csv(list_file, header=None, index_col=False, sep=" ")
        self.data = df.values
        self.image_list = np.array([osp.join(self.root, x) for x in self.data[:, 1]])

    def __len__(self):
        return len(self.image_list)


class EvaluationImageDataset(BaseImageDataset):
    def __init__(self, dataset_name, transforms=None, data_root=None, list_path=None):
        super(EvaluationImageDataset, self).__init__(
            dataset_name,
            transforms=transforms,
            data_root=data_root,
            list_path=list_path,
        )

    def __getitem__(self, index):
        img = pil_loader(self.image_list[index])
        if self.transforms is not None:
            img = self.transforms(img)
        return img


class TrainImageDataset(BaseImageDataset):
    def __init__(self, dataset_name, transforms=None, data_root=None, list_path=None):
        super(TrainImageDataset, self).__init__(
            dataset_name,
            transforms=transforms,
            data_root=data_root,
            list_path=list_path,
        )
        self.ids = self.data[:, 0].astype(int)
        self.classes = np.unique(self.ids)
        self.ages = self.data[:, 2].astype(np.float32)
        self.genders = self.data[:, 3].astype(int)

    def __getitem__(self, index):
        img = pil_loader(self.image_list[index])
        if self.transforms is not None:
            img = self.transforms(img)
        age = self.ages[index]
        gender = self.genders[index]
        label = self.ids[index]
        return img, label, age, gender


class AgingDataset(BaseImageDataset):
    def __init__(
        self,
        dataset_name,
        age_group,
        total_pairs,
        transforms=None,
        data_root=None,
        list_path=None,
    ):
        super(AgingDataset, self).__init__(
            dataset_name,
            transforms=transforms,
            data_root=data_root,
            list_path=list_path,
        )
        self.ids = self.data[:, 0].astype(int)
        self.classes = np.unique(self.ids)
        self.ages = self.data[:, 2].astype(np.float32)
        self.genders = self.data[:, 3].astype(int)
        self.groups = age2group(self.ages, age_group=age_group).astype(int)
        self.label_group_images = []
        for i in range(age_group):
            self.label_group_images.append(
                self.image_list[self.groups == i].tolist())
        np.random.seed(0)
        self.target_labels = np.random.randint(0, age_group, (total_pairs,))
        self.total_pairs = total_pairs

    def __getitem__(self, index):
        target_label = self.target_labels[index]
        target_img = pil_loader(random.choice(self.label_group_images[target_label]))
        if self.transforms is not None:
            target_img = self.transforms(target_img)
        return target_img, target_label

    def __len__(self):
        return self.total_pairs
