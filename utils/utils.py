import jittor as jt
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import os
import math
import jittor.nn as nn
from skimage import measure
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def seed_jittor(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 and classname.find('SplAtConv2d') == -1:
        jt.init.xavier_normal_(m.weight)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        jt.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        jt.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        jt.init.gauss_(m.weight, mean=1.0, std=0.02)
        jt.init.constant_(m.bias, 0.0)


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = jt.array(kernel_h).unsqueeze(0).unsqueeze(0).float32()
        kernel_v = jt.array(kernel_v).unsqueeze(0).unsqueeze(0).float32()
        self.weight_h = kernel_h
        self.weight_v = kernel_v

    def execute(self, x):
        x0 = x[:, 0]
        x0_v = nn.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = nn.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x0 = jt.sqrt(jt.pow(x0_v, 2) + jt.pow(x0_h, 2) + 1e-6)

        return x0


def random_crop(img, mask, patch_size):
    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        mask = np.pad(mask, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        h, w = img.shape
    h_start = random.randint(0, h - patch_size)
    h_end = h_start + patch_size
    w_start = random.randint(0, w - patch_size)
    w_end = w_start + patch_size

    img_patch = img[h_start:h_end, w_start:w_end]
    mask_patch = mask[h_start:h_end, w_start:w_end]

    return img_patch, mask_patch


def Normalized(img, img_norm_cfg):
    return (img - img_norm_cfg['mean']) / img_norm_cfg['std']


def Denormalization(img, img_norm_cfg):
    return img * img_norm_cfg['std'] + img_norm_cfg['mean']


def get_img_norm_cfg(dataset_name, dataset_dir):
    if dataset_name == 'NUAA-SIRST':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'NUDT-SIRST':
        img_norm_cfg = dict(mean=107.80905151367188, std=33.02274703979492)
    elif dataset_name == 'IRSTD-1K':
        img_norm_cfg = dict(mean=87.4661865234375, std=39.71953201293945)
    elif dataset_name == 'NUDT-SIRST-Sea':
        img_norm_cfg = dict(mean=43.62403869628906, std=18.91838264465332)
    elif dataset_name == 'SIRST4':
        img_norm_cfg = dict(mean=62.10432052612305, std=23.96998405456543)
    elif dataset_name == 'IRDST-real':
        img_norm_cfg = dict(mean=101.54053497314453, std=56.49856185913086)
    else:
        with open(dataset_dir + '/' + dataset_name + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            train_list = f.read().splitlines()
        with open(dataset_dir + '/' + dataset_name + '/img_idx/test_' + dataset_name + '.txt', 'r') as f:
            test_list = f.read().splitlines()
        img_list = train_list + test_list
        img_dir = dataset_dir + '/' + dataset_name + '/images/'
        mean_list = []
        std_list = []
        for img_pth in img_list:
            try:
                img = Image.open((img_dir + img_pth).replace('//', '/') + '.jpg').convert('I')
            except:
                try:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.png').convert('I')
                except:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.bmp').convert('I')
            img = np.array(img, dtype=np.float32)
            mean_list.append(img.mean())
            std_list.append(img.std())
        img_norm_cfg = dict(mean=float(np.array(mean_list).mean()), std=float(np.array(std_list).mean()))
    return img_norm_cfg


def get_optimizer(net, optimizer_name, scheduler_name, optimizer_settings, scheduler_settings):
    if optimizer_name == 'Adam':
        optimizer = jt.optim.Adam(net.parameters(), lr=optimizer_settings['lr'])
    elif optimizer_name == 'Adagrad':
        optimizer = jt.optim.Adagrad(net.parameters(), lr=optimizer_settings['lr'])
    elif optimizer_name == 'SGD':
        optimizer = jt.optim.SGD(net.parameters(), lr=optimizer_settings['lr'])

    # Jittor doesn't have built-in schedulers, so we'll implement basic ones
    scheduler = None
    if scheduler_name == 'MultiStepLR':
        # You may need to implement MultiStepLR manually or use a custom scheduler
        scheduler = {'type': 'MultiStepLR', 'milestones': scheduler_settings['step'], 'gamma': scheduler_settings['gamma']}
    elif scheduler_name == 'CosineAnnealingLR':
        # You may need to implement CosineAnnealingLR manually or use a custom scheduler
        scheduler = {'type': 'CosineAnnealingLR', 'T_max': scheduler_settings['epochs'], 'eta_min': scheduler_settings['min_lr']}

    return optimizer, scheduler


def PadImg(img, times=32):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w)), mode='constant')
    return img
