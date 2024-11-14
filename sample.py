import torch
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm  # 导入 tqdm
import shutil  # 导入 shutil 模块

from utils_train import seed_everything, make_config, make_bfn
from data import batch_to_images

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
import os
from PIL import Image
import numpy as np
import shutil
from tqdm import tqdm

def save_individual_images(samples, save_dir, batch_index):
    """Save individual images from a batch."""
    samples = samples.cpu()
    for i in range(samples.size(0)):

        # print(img_tensor.shape)

        if samples.shape[3] == 1:
            img_tensor = np.squeeze(samples[i])  # Remove the channel dimension if it's 1
            img = Image.fromarray((img_tensor.numpy()*255).astype('uint8'), 'L')
        else:
            img_tensor = (samples[i] + 1) / 2
            img = Image.fromarray((img_tensor.detach().numpy()*255).astype('uint8'))
            #img =  (img + 1) / 2

        img.save(os.path.join(save_dir, f'sample_{batch_index}_{i+1}.png'))

def main(cfg: DictConfig) -> torch.Tensor:
    """
    Config entries:
        seed (int): Optional
        config_file (str): Name of config file containing model and data config for a saved checkpoint
        load_model (str): Path to a saved checkpoint to be tested
        sample_shape (list): Shape of sample batch, e.g.:
            (3, 256) for sampling 3 sequences of length 256 from the text8 model.
            (2, 32, 32, 3) for sampling 2 images from the CIFAR10 model.
            (4, 28, 28, 1) for sampling 4 images from the MNIST model.
        n_steps (int): Number of sampling steps (positive integer).
        save_file (str): File path to save the generated sample tensor. Skip saving if None.
    """
    cfg["seed"] = 1
    seed_everything(cfg.seed)
    print(f"Seeded everything with seed {cfg.seed}")

    cfg["n_steps"] = 1000
    N = 10

    cfg['samples_shape'] = [4, 128,128,3]

    # Get model config from the training config file
    train_cfg = make_config(cfg.config_file)
    bfn = make_bfn(train_cfg.model)

    #cfg['load_model'] = "/etc/disks/omniai/data/zhangkai/oBFN/checkpoints/CEL-180/last/65000_emamodel.pt"
    cfg['load_model'] = "/etc/disks/omniai/data/zhangkai/paramrel/ConCkpt/ffhq100e/last/140000_emamodel.pt"
    save_dir = '../CacheConFID/' + train_cfg.data.dataset + '/Gen/40/'  # 'Recon/UnetZ'

    import shutil
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving images to {save_dir}")

    bfn.load_state_dict(torch.load(cfg.load_model, weights_only=True, map_location="cpu"))
    if torch.cuda.is_available():
        bfn.to("cuda")

    num_batches = N // cfg['samples_shape'][0]  # Number of batches needed

    # 使用 tqdm 添加进度条
    for batch_index in tqdm(range(num_batches), desc='Generating samples'):
        samples = bfn.sample(cfg.samples_shape, cfg.n_steps, train_cfg.model.encoder.parameters.a_dim)

        # Save each image individually
        save_individual_images(samples, save_dir, batch_index)

    # return samples

if __name__ == "__main__":
    main(OmegaConf.from_cli())
