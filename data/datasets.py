import os
import logging
import random

import torch
from torchvision import datasets
from sklearn.model_selection import train_test_split
from PIL import Image

log = logging.getLogger('main')
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_name: str, data_root: str, train: bool, transform, num_subsample=-1):
        self.data_name = data_name
        self.data_root = data_root
        self.train = train
        self.transform = transform
        self.image_list = list()

        for path,subdirs,files in os.walk(self.data_root):
            for name in files:
                if ('.jpg' in name) or ('.png' in name):
                    self.image_list.append(os.path.join(path,name))
        
        self.train_list, self.val_list = train_test_split(self.image_list, train_size=0.8, random_state=42, shuffle=True)
        if train:
            self.image_list = self.train_list
        else:
            self.image_list = self.val_list

    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,idx):
        img = Image.open(self.image_list[idx])
        self.img = self.transform(img)
        return self.img


def get_dataset(data_name: str, data_root: str, train: bool, 
                transform, num_subsample=-1, class_num=None):
    # dataset
    if data_name == 'imagenet':
        dataset_cls = get_dataset_cls(data_name)
        split = 'train' if train else 'val'
        dataset = dataset_cls(root=data_root, split=split, transform=transform)
        num_classes = len(dataset.classes)
    else:
        dataset = CustomDataset(data_name, data_root, train, transform)
        print(f"DataSet {data_name} is prepared")
        return dataset

    log.info(f"[Dataset] {data_name}(train={'True' if train else 'False'}) / "
             f"{dataset.__len__()} images are available.")

    # use smaller subset when debugging
    if num_subsample > 0:
        log.info(f"[Dataset] sample {num_subsample} images from the dataset in "
                 f"{'train' if train else 'test'} set.")
        num_subsample = min(num_subsample, len(dataset))
        indices = random.choices(range(len(dataset)), k=num_subsample)
        dataset = torch.utils.data.Subset(dataset, list(indices))
        
    return dataset, num_classes


def get_dataset_cls(name):
    try:
        return {
            'imagenet': datasets.ImageNet,
            # add your custom dataset class here
        }[name]
    except KeyError:
        raise KeyError(f"Unexpected dataset name: {name}")
