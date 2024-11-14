import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cleanfid import fid
import numpy as np


# 假设 'path_to_real_images' 和 'path_to_fake_images' 是包含你的真实和生成图像的目录
path_to_real_images = './DisFID/Fashion/True/1e4'
path_to_fake_images = './DisFID/Fashion/Gen/1e4'


# 计算FID得分
fid_score = fid.compute_fid(path_to_real_images, path_to_fake_images,mode="legacy_pytorch")
print(fid_score)