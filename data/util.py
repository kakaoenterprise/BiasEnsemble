'''Modified from https://github.com/alinlab/LfF/blob/master/data/util.py'''

import os
import torch
from torch.utils.data.dataset import Dataset, Subset
from torchvision import transforms as T
from glob import glob
from PIL import Image

class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


class ZippedDataset(Dataset):
    def __init__(self, datasets):
        super(ZippedDataset, self).__init__()
        self.dataset_sizes = [len(d) for d in datasets]
        self.datasets = datasets

    def __len__(self):
        return max(self.dataset_sizes)

    def __getitem__(self, idx):
        items = []
        for dataset_idx, dataset_size in enumerate(self.dataset_sizes):
            items.append(self.datasets[dataset_idx][idx % dataset_size])

        item = [torch.stack(tensors, dim=0) for tensors in zip(*items)]

        return item

class CMNISTDataset(Dataset):
    def __init__(self,root,split,transform=None, image_path_list=None):
        super(CMNISTDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split=='train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            self.data = self.align + self.conflict
        elif split=='valid':
            self.data = glob(os.path.join(root,split,"*"))            
        elif split=='test':
            self.data = glob(os.path.join(root, '../test',"*","*"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.data[index].split('_')[-2]),int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        
        return image, attr, self.data[index]


class bFFHQDataset(Dataset):
    def __init__(self, root, split, transform=None, image_path_list=None):
        super(bFFHQDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split=='train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            self.data = self.align + self.conflict

        elif split=='valid':
            self.data = glob(os.path.join(os.path.dirname(root), split, "*"))

        elif split=='test':
            self.data = glob(os.path.join(os.path.dirname(root), split, "*"))
            data_conflict = []
            for path in self.data:
                target_label = path.split('/')[-1].split('.')[0].split('_')[1]
                bias_label = path.split('/')[-1].split('.')[0].split('_')[2]
                if target_label != bias_label:
                    data_conflict.append(path)
            self.data = data_conflict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)  
        return image, attr, self.data[index]

class BARDataset(Dataset):
    def __init__(self, root, split, transform=None, percent=None, image_path_list=None):
        super(BARDataset, self).__init__()
        self.transform = transform
        self.percent = percent
        self.split = split
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        self.train_align = glob(os.path.join(root,'train/align',"*/*"))
        self.train_conflict = glob(os.path.join(root,'train/conflict',f"{self.percent}/*/*"))
        self.valid = glob(os.path.join(root,'valid',"*/*"))
        self.test = glob(os.path.join(root,'test',"*/*"))

        if self.split=='train':
            self.data = self.train_align + self.train_conflict
        elif self.split=='valid':
            self.data = self.valid
        elif self.split=='test':
            self.data = self.test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor(
            [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')
        image_path = self.data[index]

        if 'bar/train/conflict' in image_path:
            attr[1] = (attr[0] + 1) % 6
        elif 'bar/train/align' in image_path:
            attr[1] = attr[0]

        if self.transform is not None:
            image = self.transform(image)  
        return image, attr, (image_path, index)
    
class DogCatDataset(Dataset):
    def __init__(self, root, split, transform=None, image_path_list=None):
        super(DogCatDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image_path_list = image_path_list

        if split == "train":
            self.align = glob(os.path.join(root, "align", "*", "*"))
            self.conflict = glob(os.path.join(root, "conflict", "*", "*"))
            self.data = self.align + self.conflict
        elif split == "valid":
            self.data = glob(os.path.join(root, split, "*"))
        elif split == "test":
            self.data = glob(os.path.join(root, "../test", "*", "*"))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor(
            [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)  
        return image, attr, self.data[index]


transforms = {
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "valid": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
        },
    "bar": {
        "train": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "valid": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "test": T.Compose([T.Resize((224, 224)), T.ToTensor()])
    },
    "bffhq": {
        "train": T.Compose([T.Resize((224,224)), T.ToTensor()]),
        "valid": T.Compose([T.Resize((224,224)), T.ToTensor()]),
        "test": T.Compose([T.Resize((224,224)), T.ToTensor()])
        },
    "dogs_and_cats": {
        "train": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "valid": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
        "test": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
    },
    }


transforms_preprcs = {
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "valid": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()])
        },
    "bar": {
        "train": T.Compose([
            T.Resize((224, 224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        ),
        "valid": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        ),
        "test": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        )
    },
    "bffhq": {
        "train": T.Compose([
            T.Resize((224,224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "valid": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "test": T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        },
    "dogs_and_cats": {
            "train": T.Compose(
                [
                    T.Resize((224, 224)),
                    T.RandomCrop(224, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            ),
            "valid": T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            ),
            "test": T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            ),
        },
}

def get_dataset(dataset, data_dir, dataset_split, transform_split, percent, use_preprocess=None, image_path_list=None):

    dataset_category = dataset.split("-")[0]
    if use_preprocess:
        transform = transforms_preprcs[dataset_category][transform_split]
    else:
        transform = transforms[dataset_category][transform_split]

    dataset_split = "valid" if (dataset_split == "eval") else dataset_split

    if dataset == 'cmnist':
        root = data_dir + f"/cmnist/{percent}"
        dataset = CMNISTDataset(root=root,split=dataset_split,transform=transform, image_path_list=image_path_list)

    elif dataset == "bffhq":
        root = data_dir + f"/bffhq/{percent}"
        dataset = bFFHQDataset(root=root, split=dataset_split, transform=transform, image_path_list=image_path_list)

    elif dataset == "bar":
        root = data_dir + f"/bar"
        dataset = BARDataset(root=root, split=dataset_split, transform=transform, percent=percent, image_path_list=image_path_list)

    elif dataset == "dogs_and_cats":
        root = data_dir + f"/dogs_and_cats/{percent}"
        print(root)
        dataset = DogCatDataset(root=root, split=dataset_split, transform=transform, image_path_list=image_path_list)

    else:
        print('wrong dataset ...')
        import sys
        sys.exit(0)

    return dataset