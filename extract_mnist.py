# import torchvision
# import torchvision.transforms as transforms
# from tqdm import tqdm  # 导入tqdm
# import os
#
# # 设置数据集下载和保存的路径
# dataset_path = './data'
# # 设置图像保存的目标文件夹
# target_folder = './discrete_mnist_images/5e4'
# os.makedirs(target_folder, exist_ok=True)
#
# # 定义一个转换，先将数据转换为Tensor，然后应用一个离散化函数
# transform = transforms.Compose([
#     transforms.Resize((28, 28)),  # 确保图像缩放到28x28
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: (x > 0.5).float())  # 离散化：二值化处理，大于0.5的为1，否则为0
# ])
#
# # 加载MNIST训练数据集
# train_dataset = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)
#
# # 使用tqdm显示进度条
# for i in tqdm(range(10000), desc="Saving images"):  # 假设我们只保存前10个图像
#     image, label = train_dataset[i]
#     # 将Tensor转换为PIL图像
#     image_pil = transforms.ToPILImage()(image).convert("L")  # 转换为灰度图
#     # 保存图像
#     image_pil.save(os.path.join(target_folder, f'image2_{i}_label_{label}.png'))
import os
import random
from math import inf

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_ema
import torchvision
import torchvision.transforms as transforms
from torch.nn.functional import relu
from torch.optim import AdamW
from torchsummary import summary
from tqdm.auto import tqdm


DATAPATH = "../datasets/"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODEL = True

print(f"Using device: {DEVICE}")
if SEED:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)


def get_sample(dataloader: torch.utils.data.DataLoader) -> tuple[torch.Tensor, int]:
    """
    Returns a random sample from a data loader

    Args:
        dataloader (torch.utils.data.DataLoader): data loader storing the images.

    Returns:
        tuple[torch.Tensor, int]: (image, label)
    """
    for sample in dataloader:
        return sample[0], sample[1][0].numpy()
    raise IndexError("Could not sample an empty data loader.")


def show_samples(dataloader: torch.utils.data.DataLoader, n: int, title: str = None) -> None:
    """
    Displays some random samples from a data loader.

    Args:
        dataloader (torch.utils.data.DataLoader): data loader storing the images.
        n (int): number of samples to display.
    """
    fig, ax = plt.subplots(1, n, figsize=(3 * n, 3))
    for i in range(n):
        img, label = get_sample(dataloader)
        ax[i].imshow(img[0][0], cmap='Greys_r', interpolation='nearest')
        ax[i].set_title(label)
        ax[i].axis("off")
    title = title if title else f"{n} random samples"
    fig.suptitle(title, position=(0.5, 1.1))


def moving_average(data: list[float], window_size: int = 20) -> list[float]:
    """
    Computes the moving average of a list of values.

    Args:
        data (list[float]): list of values.
        window_size (int, optional): length of the window over which the values will be averaged out. Defaults to 20.

    Returns:
        list[float]: list of averaged values.
    """
    moving_avg = []
    for i in range(len(data) - window_size + 1):
        window = data[i: i + window_size]
        avg = sum(window) / window_size
        moving_avg.append(avg)
    return moving_avg

# Define the data transformation
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,)),  # Normalize the pixel values to [-1, 1]
    #transforms.Lambda(lambda x: torch.round((x + 1) / 2).to(torch.int64)) # Discretize data
])

# Download and load the MNIST dataset
train_dataset = torchvision.datasets.MNIST(root=DATAPATH, train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root=DATAPATH, train=False, transform=transform, download=True)

# Create data loaders
batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class DynamicallyBinarizedMNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(DynamicallyBinarizedMNIST, self).__init__(root, train=train, transform=transform,
                                                        target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

def collate_dynamic_binarize(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function that samples a binarization probability for each batch.

    Args:
        batch (list[tuple[torch.Tensor, int]]): list of samples to collate.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: resulting batch.
    """
    images, targets = zip(*batch)
    binarization_probs = torch.rand(len(images))
    binarized_images = []
    for img, prob in zip(images, binarization_probs):
        binarized_img = (img > prob).float()
        binarized_images.append(binarized_img)
    return torch.stack(binarized_images)[:, None, ...].to(torch.int64), torch.tensor(targets)

# Create the dynamically binarized MNIST dataset
train_dataset = DynamicallyBinarizedMNIST(root=DATAPATH, train=True, download=True) # transform=transform
test_dataset = DynamicallyBinarizedMNIST(root=DATAPATH, train=False, download=True) # transform=transform

# Create data loaders with the collate function
batch_size = 512
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_dynamic_binarize)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_dynamic_binarize)


from torchvision.utils import save_image
from tqdm import tqdm
import os


def save_images_directly(dataloader, n, save_path):
    """
    Saves images directly from a dataloader as PNG files without plotting,
    with progress tracking via tqdm.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader storing the images.
        n (int): Number of samples to save.
        save_path (str): Path where the images will be saved.
    """
    # Ensure the save path exists
    import shutil
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    # Initialize a progress bar
    pbar = tqdm(total=n, desc="Saving images")

    # Keep track of how many images have been saved
    saved_images = 0

    # Iterate through the dataloader
    for images, labels in dataloader:
        if saved_images >= n:
            break  # Exit the loop if we've saved enough images

        # Ensure images are on CPU and convert them to float for saving
        images = images.cpu().float()

        for j, label in enumerate(labels):
            if saved_images >= n:
                break  # Check again to stop saving when enough images have been saved

            # Save each image, note that `save_image` adjusts the range itself
            file_name = os.path.join(save_path, f"sample_{saved_images}_label_{label.item()}.png")
            save_image(images[j], file_name)

            saved_images += 1
            pbar.update(1)  # Update the progress bar

    pbar.close()  # Close the progress bar


# Example usage
save_images_directly(train_loader, 50000, "../CacheDisFID/mnist/True/5e4")