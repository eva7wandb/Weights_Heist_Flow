from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class CIFAR10_dataset():
    """
    a utility class to get data loader for CIFAR10.
    transforms are hardcoded for now.
    USAGE:
        train_loader = CIFAR10_dataset(
            train=True, cude=torch.cuda.is_available()
        ).get_loader()
    """
    def __init__(
        self,
        train, cuda,
        root='./data',
        normalize=True,
    ):
        self.train = train
        self.cuda = cuda
        self.root = root
        
        self.mean = (0.491, 0.482, 0.447)
        self.std = (0.247, 0.243, 0.262)
        
        if normalize:
            self.train_transforms = A.Compose([
                A.Normalize(
                    mean=self.mean, 
                    std=self.std,
                    always_apply=True
                ),
                A.Sequential([
                    A.PadIfNeeded(40,40),
                    A.RandomCrop(32,32)],
                    p=0.5
                ),
                #A.CropAndPad(px=4,keep_size=False, p=0.5,),
                #A.RandomCrop(32, 32, always_apply=False, p=1),
                A.HorizontalFlip(p=0.5),
                #A.Cutout (num_holes=8, max_h_size=8, fill_value=(0.491, 0.482, 0.447), always_apply=False, p=0.5),
                A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
#                 A.CoarseDropout(
#                     max_holes=3, max_height=16, max_width=16, min_holes=None, min_height=None, min_width=None, 
#                     fill_value=(0.491, 0.482, 0.447), mask_fill_value=None, always_apply=False, p=0.25
#                 ),
#                 A.Rotate(limit=5, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
                ToTensorV2()
            ])
            self.test_transforms = A.Compose([
                A.Normalize(
                    mean=self.mean, 
                    std=self.std,
                    always_apply=True
                ),
                ToTensorV2()
            ])
        else:
            self.train_transforms = A.Compose([
                A.RandomCrop(32, 32, always_apply=False, p=0.5),
                A.CoarseDropout(
                    max_holes=3, max_height=16, max_width=16, min_holes=None, min_height=None, min_width=None, 
                    fill_value=(0.491, 0.482, 0.447), mask_fill_value=None, always_apply=False, p=0.25
                ),
                A.Rotate(limit=5, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
                ToTensorV2()
            ])
            self.test_transforms = A.Compose([
                ToTensorV2()
            ])
        
        if self.train:
            self.transforms = self.train_transforms
        else:
            self.transforms = self.test_transforms
            
        self.shuffle = True if self.train else False
        self.classes = None
        
        
    def get_data(self):
        data = datasets.CIFAR10(
            self.root,
            train=self.train,
            transform=lambda img:self.transforms(image=np.array(img))["image"],
            download=True
        )
        self.classes = data.classes
        return data
            
    def get_loader(self, batch_size=128):
        data = self.get_data()

        dataloader_args = dict(
            shuffle=self.shuffle, 
            batch_size=batch_size, 
            num_workers=2, 
            pin_memory=True
        ) if self.cuda else dict(
            shuffle=self.shuffle, 
            batch_size=64
        )
        data_loader = DataLoader(data, **dataloader_args)
        print(
            f"""[INFO] {'train' if self.train else 'test'} dataset of size {len(data)} loaded..."""
        )
        return data_loader