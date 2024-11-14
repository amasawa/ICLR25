from data import make_datasets
from torch.utils.data import DataLoader
import argparse
from omegaconf import OmegaConf
from utils_train import (
    seed_everything,  make_config, make_dataloaders,
)
from torchvision.utils import save_image
from tqdm import tqdm
import os

def save_images_directly(dataloader, n, save_path, cfg):
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
    for images in dataloader:
        if saved_images >= n:
            break  # Exit the loop if we've saved enough images

        image = images.float()
        image = image.permute(2,0,1).float()

        file_name = os.path.join(save_path, f"sample_{saved_images}.png")
        save_image(image, file_name)

        saved_images += 1
        pbar.update(1)  # Update the progress bar

    pbar.close()  # Close the progress bar
 
def main(cfg):

    seed_everything(cfg.training.seed)



    trainset,_,_ = make_datasets(cfg.data)
    print("dataloader  done!")
    # Example usage
    save_images_directly(trainset, 40, "../CacheConFID/" + cfg.data.dataset + "/True/40", cfg)


if __name__ == "__main__":
    cfg_file = "configs/ffhq/ffhq.yaml" #OmegaConf.from_cli()['config_file']

    print("cfg load done!")

    main(make_config(cfg_file))
