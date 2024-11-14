# Copyright 2023 NNAISENSE SA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import pathlib
import pickle
import zipfile
from typing import Union, Tuple, List

import numpy as np
import requests
import torch
import torchvision
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.utils import make_grid

from utils_model import quantize
from torch.utils.data import Dataset, DataLoader
TEXT8_CHARS = list("_abcdefghijklmnopqrstuvwxyz")
import random
from ffhqdataset import *
# from visualize.visualize import Visualizer

class CustomTensorDatasetfor2(Dataset):
    def __init__(self, data_tensor, labels,transform=None, noisy=False, color=False):
        self.data_tensor = data_tensor
        self.transform = transform
        self.labels = labels
        self.indices = range(len(self))
        self.noisy = noisy
        self.color = color
        if self.color:
            self.color_tensor = np.random.uniform(0, 1, [data_tensor.size(0), 3, 1, 1])

    def __getitem__(self, index1):

        if self.noisy:

            index2 = random.choice(self.indices)

            img1 = self.data_tensor[index1]
            color1 = np.random.uniform(0, 1, [3, 64, 64])
            img2 = self.data_tensor[index2]
            color2 = np.random.uniform(0, 1, [3, 64, 64])
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return np.minimum(img1.repeat(3, 1, 1) + color1, 1).float(), np.minimum(img2.repeat(3, 1, 1) + color2,
                                                                                    1).float()
        elif self.color:

            index2 = random.choice(self.indices)

            img1 = self.data_tensor[index1]
            img2 = self.data_tensor[index2]
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return (img1.repeat(3, 1, 1) * self.color_tensor[index1]).float(), (
                        img2.repeat(3, 1, 1) * self.color_tensor[index2]).float()
        else:

            index2 = random.choice(self.indices)

            img1 = self.data_tensor[index1]
            img2 = self.data_tensor[index2]
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img1, img2, self.labels[index1], self.labels[index2]

    def __len__(self):
        return self.data_tensor.size(0)

def bin_mnist_transform(x):
    return torch.bernoulli(x.permute(1, 2, 0).contiguous()).int()


def bin_mnist_cts_transform(x):
    return torch.bernoulli(x.permute(1, 2, 0).contiguous()) - 0.5


def rgb_image_transform(x, num_bins=256):
    return quantize((x * 2) - 1, num_bins).permute(1, 2, 0).contiguous()

def bfnpermute(x, num_bins=256):
    return x.permute(1, 2, 0)


class MyLambda(torchvision.transforms.Lambda):
    def __init__(self, lambd, arg1=None):
        super().__init__(lambd)
        self.arg1 = arg1

    def __call__(self, x):
        return self.lambd(x, self.arg1)

# class MyLambda(torchvision.transforms.Lambda):
#     def __init__(self, lambd):
#         super().__init__(lambd)
#
#     def __call__(self, x):
#         return self.lambd(x)


class CIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, idx):
        return super().__getitem__(idx)[0],super().__getitem__(idx)[1]


class MNIST(torchvision.datasets.MNIST):
    def __getitem__(self, idx):
        return super().__getitem__(idx)[0],super().__getitem__(idx)[1]
class Crop:
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return torchvision.transforms.crop(img, self.x1, self.y1, self.x2 - self.x1,
                                           self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2)
def d2c_crop():
    # from D2C paper for CelebA dataset.
    cx = 89
    cy = 121
    x1 = cy - 64
    x2 = cy + 64
    y1 = cx - 64
    y2 = cx + 64
    return Crop(x1, x2, y1, y2)

import h5py
class CelebADataset(Dataset):
    def __init__(self, file, transform,type='train'):
        self.file_object = h5py.File(file, 'r')  # Open the file
        self.dataset = self.file_object['images']  # Access the group
        self.target = self.file_object['labels']
        self.transform = transform
        self.type = type

    def __len__(self):
        return len(self.dataset["img_align_celeba"].keys())  # Total number of items in the dataset

    def __getitem__(self, index):  # Retrieve an item
        if index >= len(self.dataset["img_align_celeba"].keys()):  # Check for out-of-bounds index
            raise IndexError("Index out of range")
        if self.type == 'train':
            index +=1
        elif self.type == 'valid':
            index += 162771
        elif self.type == 'test':
            index += 182638
        # Format the index as a string with leading zeros (assuming a maximum of 6 digits in the index)
        formatted_index = f"{index:06d}.jpg"  # Example: index 8978 becomes "008978.jpg"

        # Access the image by its formatted name
        # img = np.array(self.dataset["img_align_celeba/"+formatted_index])
        img = self.dataset["img_align_celeba/"+formatted_index][:]

        # Apply the transformation to the image if one is provided
        # if self.transform:
        #     img = self.transform(img)

        return img, self.target["img_align_celeba/"+formatted_index][:]

def make_datasets(cfg: DictConfig) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Mandatory keys: dataset (must be cifar10, mnist, bin_mnist, bin_mnist_cts or text8), data_dir
    Optional for vision: num_bins (default 256), val_frac (default 0.01), horizontal_flip (default: False)
    Mandatory for text: seq_len
    """
    num_bins = cfg.get("num_bins", 256)
    #print(cfg.dataset)
    if cfg.dataset == "cifar10":
        train_transform_list = [transforms.ToTensor()]
        if cfg.get("horizontal_flip", False):
            train_transform_list.append(transforms.RandomHorizontalFlip())
        train_transform_list.append(MyLambda(rgb_image_transform, num_bins))
        train_transform = transforms.Compose(train_transform_list)
        test_transform = transforms.Compose([transforms.ToTensor(), MyLambda(rgb_image_transform, num_bins)])
        train_set = CIFAR10(root=cfg.data_dir, train=True, download=True, transform=train_transform)
        val_set = CIFAR10(root=cfg.data_dir, train=True, download=True, transform=test_transform)
        test_set = CIFAR10(root=cfg.data_dir, train=False, download=True, transform=test_transform)
        # for i in range(len(train_set)):
        #     print("img", train_set[i].shape) # img torch.Size([32, 32, 3])
        #     assert False
    elif cfg.dataset == "shapes3d":
        pass
        # visl = Visualizer(name="shapes3d")
        # images, labels = visl.dataset.images, visl.dataset.labels
        # data = torch.from_numpy(images).float()
        # train_kwargs = {'data_tensor': data,'labels': labels,'noisy':False, 'color':False}
        # dset = CustomTensorDatasetfor2
        # train_set = dset(**train_kwargs)
        # val_set = None
        # test_set = None

    elif cfg.dataset == "bfashion_mnist":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Lambda(lambda t: (t * 2) - 1),
            transforms.Lambda(bin_mnist_transform)
        ])

        dataset = torchvision.datasets.FashionMNIST(root="/etc/disks/omniai/data/zhangkai/dataset", train=True, download=True, transform=transform)
        train_set = dataset
        val_set = torchvision.datasets.FashionMNIST(root="/etc/disks/omniai/data/zhangkai/dataset", train=False, download=True, transform=transform)
        test_set = None
    elif cfg.dataset == "dsprites":
        root = os.path.join("/etc/disks/omniai/data/zhangkai/dataset" + '/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        file = np.load(root, encoding='latin1')
        data = file['imgs'][:, np.newaxis, :, :]
        latents_values = file['latents_values']
        latents_classes = file['latents_classes']
        train_kwargs = {'data': data, 'latents_values': latents_values, 'latents_classes': latents_classes}
        dset = CustomTensorDataset
        train_set = dset(**train_kwargs)
        test_set = None
        val_set = None
    elif cfg.dataset == "celebah5PY":
        print("********************Loading from h5py********************")
        data_dir = "/home/zhanwu/Data/dataset/celeba/"
        #data_dir = "/scratch/xs71/xf4858/zhangkai/dataset/celeba/"
        if cfg.get("infodiffusionTransform", True):
            print("--------transforming from infoDiffusion")
            as_tensor = True,
            do_augment = True,
            do_normalize = True,
            crop_d2c = False
            input_size = 64
            if crop_d2c:
                transform = [
                    d2c_crop(),
                    torchvision.transforms.Resize(input_size),
                ]
            else:
                transform = [
                    torchvision.transforms.Resize(input_size),
                    torchvision.transforms.CenterCrop(input_size),
                ]

            if do_augment:
                transform.append(torchvision.transforms.RandomHorizontalFlip())
            if as_tensor:
                transform.append(torchvision.transforms.ToTensor())
            if do_normalize:
                transform.append(
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            transform.append(MyLambda(bfnpermute))

            transform = torchvision.transforms.Compose(transform)

            train_set = CelebADataset(data_dir+"train_transformed.h5py",transform,'train')
            val_set = CelebADataset(data_dir+"valid_transformed.h5py",transform,'valid')
            test_set = CelebADataset(data_dir+"test_transformed.h5py", transform, 'test')
        else:
            print("--------transforming from BFN--------")
            train_transform_list = [transforms.ToTensor(), torchvision.transforms.Resize(64),
                                    torchvision.transforms.CenterCrop(64)]
            if cfg.get("horizontal_flip", False):
                print("yes we use horizontal_flip")
                train_transform_list.append(transforms.RandomHorizontalFlip())
            train_transform_list.append(MyLambda(rgb_image_transform, num_bins))
            train_transform = transforms.Compose(train_transform_list)
            test_transform = transforms.Compose([transforms.ToTensor(), torchvision.transforms.Resize(64),
                                                 torchvision.transforms.CenterCrop(64),
                                                 MyLambda(rgb_image_transform, num_bins)])
            train_set = CelebADataset(data_dir+"train_transformed.h5py", train_transform, 'train')
            val_set = CelebADataset(data_dir+"valid_transformed.h5py", test_transform, 'valid')
            test_set = CelebADataset(data_dir+"test_transformed.h5py", test_transform, 'test')

    elif cfg.dataset == "celeba":
        print("********************Loading from torchvision********************")
        if cfg.get("infodiffusionTransform", True):
            print("--------transforming from infoDiffusion")
            as_tensor = True,
            do_augment = True,
            do_normalize = True,
            crop_d2c = False
            input_size = 64
            if crop_d2c:
                transform = [
                    d2c_crop(),
                    torchvision.transforms.Resize(input_size),
                ]
            else:
                transform = [
                    torchvision.transforms.Resize(input_size),
                    torchvision.transforms.CenterCrop(input_size),
                ]

            if do_augment:
                transform.append(torchvision.transforms.RandomHorizontalFlip())
            if as_tensor:
                transform.append(torchvision.transforms.ToTensor())
            if do_normalize:
                transform.append(
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            transform.append(MyLambda(bfnpermute))

            transform = torchvision.transforms.Compose(transform)

            train_set = torchvision.datasets.CelebA(root="/etc/disks/omniai/data/zhangkai/datasets", split="train", download=False,
                                                    transform=transform)
            val_set = torchvision.datasets.CelebA(root="/etc/disks/omniai/data/zhangkai/datasets", split="valid", download=False,
                                                  transform=transform)
            test_set = torchvision.datasets.CelebA(root="/etc/disks/omniai/data/zhangkai/datasets", split="test", download=False,
                                                   transform=transform)
        else:
            print("------------------transforming from BFN------------------")
            train_transform_list = [transforms.ToTensor(), torchvision.transforms.Resize(64),
                                    torchvision.transforms.CenterCrop(64)]
            if cfg.get("horizontal_flip", True):
                print("yes we use horizontal_flip")
                train_transform_list.append(transforms.RandomHorizontalFlip())
            train_transform_list.append(MyLambda(rgb_image_transform, num_bins))
            train_transform = transforms.Compose(train_transform_list)
            print(train_transform_list)
            test_transform = transforms.Compose([transforms.ToTensor(), torchvision.transforms.Resize(64),
                                    torchvision.transforms.CenterCrop(64), MyLambda(rgb_image_transform, num_bins)])
            train_set = torchvision.datasets.CelebA(root="/etc/disks/omniai/data/zhangkai/datasets", split="train", download=False,transform=train_transform)
            val_set = torchvision.datasets.CelebA(root="/etc/disks/omniai/data/zhangkai/datasets", split="valid", download=False,
                                                  transform=test_transform)
            test_set = torchvision.datasets.CelebA(root="/etc/disks/omniai/data/zhangkai/datasets", split="test", download=False,
                                                   transform=test_transform)

            # from PIL import Image
            #
            # # Assuming train_set has already been defined as shown in your code snippet
            # image, label = train_set[0]  # This gets the first image and its corresponding label from the dataset
            # image = image.permute(1,2,0)  # Rearrange the tensor dimensions from CxHxW to HxWxC
            # image = (image * 255).byte().numpy()  # Convert from tensor to numpy array and scale to 0-255
            # image = Image.fromarray(image)  # Convert numpy array to PIL Image for saving

            # # Save the image
            # image.save('celebaNoquan.png')
            # assert False

            # num_samples = len(train_set)
            # print(f"训练集中的样本数量为: {num_samples}") #  162770
            # assert False
        # for i,j in train_set:
        #     print("img", i.shape) # infoDiffusion: torch.Size([3, 64, 64]), BFN: torch.Size([64, 64, 3])
        #     assert False
    elif cfg.dataset == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                MyLambda(rgb_image_transform, num_bins),
            ]
        )
        train_set = MNIST(root=cfg.data_dir, train=True, download=True, transform=transform)
        val_set = MNIST(root=cfg.data_dir, train=True, download=True, transform=transform)
        test_set = MNIST(root=cfg.data_dir, train=False, download=True, transform=transform)

    elif cfg.dataset == "bin_mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(bin_mnist_transform)])
        train_set = MNIST(root=cfg.data_dir, train=True, download=True, transform=transform)
        val_set = MNIST(root=cfg.data_dir, train=True, download=True, transform=transform)
        test_set = MNIST(root=cfg.data_dir, train=False, download=True, transform=transform)

    elif cfg.dataset == "bin_mnist_cts":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(bin_mnist_cts_transform)])
        train_set = MNIST(root=cfg.data_dir, train=True, download=True, transform=transform)
        val_set = MNIST(root=cfg.data_dir, train=True, download=True, transform=transform)
        test_set = MNIST(root=cfg.data_dir, train=False, download=True, transform=transform)

    elif cfg.dataset == "text8":
        train_set = Text8Dataset(cfg.data_dir, "train", download=True, seq_len=cfg.seq_len)
        val_set = Text8Dataset(cfg.data_dir, "val", download=True, seq_len=cfg.seq_len)
        test_set = Text8Dataset(cfg.data_dir, "test", download=True, seq_len=cfg.seq_len)

    elif cfg.dataset == "ffhq":
        train_set = FFHQlmdb(path='../datasets/ffhq256_lmdb',image_size=128, split="train")
        val_set = None
        test_set = FFHQlmdb(path='../datasets/ffhq256_lmdb',image_size=128,split="test")
    else:
        raise NotImplementedError(cfg.dataset)

    if cfg.dataset in ["bin_mnist", "bin_mnist_cts","mnist"]:
        # For vision datasets we split the train set into train and val
        val_frac = cfg.get("val_frac", 0.01)
        train_val_split = [1.0 - val_frac, val_frac]
        seed = 2147483647
        train_size = len(train_set)
        val_size = len(val_set)
        train_set = random_split(train_set, [int(train_size*(1.0 - val_frac)), int(train_size*(val_frac))], generator=torch.Generator().manual_seed(seed))[0]
        val_set = random_split(val_set, [int(val_size*(1.0 - val_frac)), int(val_size*(val_frac))], generator=torch.Generator().manual_seed(seed))[1]

    #print(train_set[0][0].shape)

    return train_set, val_set, test_set


def prepare_text8(data_dir: pathlib.Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    data_url = "http://mattmahoney.net/dc/text8.zip"
    with open(data_dir / "text8.zip", "wb") as f:
        print("Downloading text8")
        f.write(requests.get(data_url).content)
        print("Done")
    with zipfile.ZipFile(data_dir / "text8.zip") as f:
        f.extractall(data_dir)
    os.remove(data_dir / "text8.zip")
    data = (data_dir / "text8").read_text()

    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", "".join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]  # encoder: take a string, output a list of integers

    # encode both to integers
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) : int(n * 0.95)]
    test_data = data[int(n * 0.95) :]
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    test_ids = encode(test_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")
    print(f"test has {len(test_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    test_ids = np.array(test_ids, dtype=np.uint16)
    train_ids.tofile(data_dir / "train.bin")
    val_ids.tofile(data_dir / "val.bin")
    test_ids.tofile(data_dir / "test.bin")
    print(f"Saved to {data_dir / 'train.bin'}, {data_dir / 'val.bin'}, {data_dir / 'test.bin'}")

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(os.path.join(data_dir / "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"text8 dataset downloaded and prepared in dir {data_dir}")


class Text8Dataset(Dataset):
    def __init__(self, data_dir: Union[str, pathlib.Path], split: str, download: bool, seq_len: int):
        """
        seq_len should include context length. Example: seq_len=512 for modeling 256 chars with 256 char of context.
        context is only used for correct preparation of val/test sets.
        """
        self.root_dir = pathlib.Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        fname = {"train": "train.bin", "val": "val.bin", "test": "test.bin"}[self.split]
        assert self.split in ["train", "val", "test"]
        data_dir = self.root_dir / "text8"
        if not os.path.exists(data_dir):
            if download:
                prepare_text8(data_dir)
            else:
                raise NotADirectoryError(f"dir {data_dir} does not exist and download is False")
        self.data = np.memmap(data_dir / fname, np.uint16, "r")

    def __getitem__(self, index) -> torch.Tensor:
        seq = torch.from_numpy(self.data[index : index + self.seq_len].astype(np.int64))
        return seq

    def __len__(self):
        return self.data.size - self.seq_len


def char_ids_to_str(char_ids: Union[List[int], np.array, torch.Tensor]) -> str:
    """Decode a 1D sequence of character IDs to a string."""
    return "".join([TEXT8_CHARS[i] for i in char_ids])


def batch_to_str(text_batch: Union[List[list], np.array, torch.Tensor]) -> List[str]:
    """Decode a batch of character IDs to a list of strings."""
    return [char_ids_to_str(row_char_ids) for row_char_ids in text_batch]


def batch_to_images(image_batch: torch.Tensor, ncols: int = None) -> plt.Figure:
    if ncols is None:
        ncols = math.ceil(math.sqrt(len(image_batch)))
    if image_batch.size(-1) == 3:  # for color images (CIFAR-10)
        image_batch = (image_batch + 1) / 2
    grid = make_grid(image_batch.permute(0, 3, 1, 2), ncols, pad_value=1).permute(1, 2, 0)
    fig = plt.figure(figsize=(grid.size(1) / 30, grid.size(0) / 30))
    plt.imshow(grid.cpu().clip(min=0, max=1), interpolation="nearest")
    plt.grid(False)
    plt.axis("off")
    return fig

def get_fmnist(args):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.input_size, args.input_size)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda t: (t * 2) - 1),
    ])

    dataset = torchvision.datasets.FashionMNIST(root = args.data_dir, train=True, download=True, transform = transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, drop_last = True, num_workers = 4)

    return dataloader



class CustomTensorDataset(Dataset):
    def __init__(self, data, latents_values, latents_classes):
        self.data = data
        self.latents_values = latents_values
        self.latents_classes = latents_classes

    def __getitem__(self, index):
        return (torch.from_numpy(self.data[index]).float().permute(1,2,0),
                torch.from_numpy(self.latents_values[index]).float(),
                torch.from_numpy(self.latents_classes[index]).int())

    def __len__(self):
        return self.data.shape[0]

# def get_dsprites(args):
#
#
#     dataloader = DataLoader(dataset,
#                             batch_size=args.batch_size,
#                             shuffle=True,
#                             num_workers=4,
#                             pin_memory=True,
#                             drop_last=True)
#
#     return dataloader