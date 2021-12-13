from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import matplotlib.pyplot as plt

def albumentations_transforms(p=1.0, is_train=False, normalize=True):
    # Mean and standard deviation of train dataset
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    transforms_list = []
    # Use data aug only for train data
    if is_train:
        transforms_list.extend([
            A.PadIfNeeded(min_height=72, min_width=72, p=1.0),
            A.RandomCrop(height=64, width=64, p=1.0),
            A.HorizontalFlip(p=0.25),
            A.Rotate(limit=15, p=0.25),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.25),
            A.CoarseDropout(max_holes=1, max_height=32, max_width=32, min_height=8,
                            min_width=8, fill_value=mean*255.0, p=0.5),
        ])
    if normalize:
        transforms_list.extend([
            A.Normalize(
                mean=mean,
                std=std,
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()
        ])
    else:
        transforms_list.extend([ToTensorV2()])

    data_transforms = A.Compose(transforms_list, p=p)
    return lambda img: data_transforms(image=np.array(img))["image"]

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
                A.CropAndPad(px=4,keep_size=False, p=0.5,),
                A.RandomCrop(32, 32, always_apply=False, p=1),
                A.HorizontalFlip(p=0.5),
                A.Cutout (num_holes=8, max_h_size=8, fill_value=(0.491, 0.482, 0.447), always_apply=False, p=0.5),
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


class TinyImagenet200_dataset():

    def __init__(self,
                 train,
                 cuda,
                 root='./data',
                 normalize=True,
                 batch_size_cpu=64,
                 batch_size_cuda=128,
                 num_workers = 3,
                 ):
        self.classes = ["%s" % i for i in range(200)]
        self.train = train
        self.cuda = cuda
        self.root = root
        self.batch_size_cuda = batch_size_cuda
        self.num_workers = num_workers
        self.batch_size_cpu = batch_size_cpu
        self.train_data_path = root + '/tiny-imagenet-200/new_train'
        self.test_data_path = root + '/tiny-imagenet-200/new_test'

    def _transforms(self):
        # Data Transformations
        train_transform = albumentations_transforms(p=1.0, is_train=True)
        test_transform = albumentations_transforms(p=1.0, is_train=False)
        return train_transform, test_transform

    def _dataset(self):
        # Get data transforms
        train_transform, test_transform = self._transforms()

        # Dataset and Creating Train/Test Split
        train_set = datasets.ImageFolder(root=self.train_data_path,	transform=train_transform)
        test_set = datasets.ImageFolder(root=self.test_data_path, transform=test_transform)
        return train_set, test_set

    def get_loader(self):
        # Get Train and Test Data
        train_set, test_set = self._dataset()

        # Dataloader Arguments & Test/Train Dataloaders
        dataloader_args = dict(
            shuffle= False,
            batch_size= self.batch_size_cpu)
        if self.cuda:
            dataloader_args.update(
                batch_size= self.batch_size_cuda,
                num_workers= self.num_workers,
                pin_memory= True)
        if self.train:
            dataloader_args.update(shuffle=True)
            return DataLoader(train_set, **dataloader_args)
        else:
            return DataLoader(test_set, **dataloader_args)

        # self.train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)
        # self.test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)

    def show_samples(self):
        # get some random training images
        dataiter = iter(self.train_loader)
        images, labels = dataiter.next()
        index = []
        num_img = min(len(self.classes), 10)
        for i in range(num_img):
            for j in range(len(labels)):
                if labels[j] == i:
                    index.append(j)
                    break
        if len(index) < num_img:
            for j in range(len(labels)):
                if len(index) == num_img:
                    break
                if j not in index:
                    index.append(j)
        imshow(make_grid(images[index], nrow=num_img, scale_each=True), "Sample train data")

def imshow(img, title):
    img = denormalize(img)
    npimg = img.numpy()
    fig = plt.figure(figsize=(15,7))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)

def denormalize(tensor, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
    single_img = False
    if tensor.ndimension() == 3:
        single_img = True
        tensor = tensor[None,:,:,:]

    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    ret = tensor.mul(std).add(mean)
    return ret[0] if single_img else ret