import argparse
import cv2
import jittor as jt
import scipy.io as scio
import os
import random

from net import Net
from utils.utils import seed_jittor, get_optimizer
from utils.datasets import NUDTSIRSTSetLoader
from utils.datasets import IRSTD1KSetLoader

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
            
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = []
            for idx in batch_indices:
                batch_data.append(self.dataset[idx])
            
            if len(batch_data) == 1:
                # 单个样本的情况，直接返回（已经有正确的维度）
                item = batch_data[0]
                if len(item) == 2:  # 训练数据 (img, mask)
                    img, mask = item[0], item[1]
                    # 添加batch维度: (1, H, W) -> (1, 1, H, W)
                    img = img.unsqueeze(0) 
                    mask = mask.unsqueeze(0)
                    yield (img, mask)
                elif len(item) == 4:  # 测试数据 (img, mask, size, name)
                    img, mask, size, name = item[0], item[1], item[2], item[3]
                    # 添加batch维度: (1, H, W) -> (1, 1, H, W)
                    img = img.unsqueeze(0)
                    mask = mask.unsqueeze(0)
                    yield (img, mask, size, name)
            else:
                # 多个样本的情况
                if len(batch_data[0]) == 2:  # 训练数据
                    batch_imgs = []
                    batch_masks = []
                    for item in batch_data:
                        batch_imgs.append(item[0])  # Jittor张量 (1, H, W)
                        batch_masks.append(item[1]) # Jittor张量 (1, H, W)
                    
                    # 使用Jittor的stack: (batch_size, 1, H, W)
                    batch_imgs = jt.stack(batch_imgs, dim=0)
                    batch_masks = jt.stack(batch_masks, dim=0)
                    yield (batch_imgs, batch_masks)
                    
                elif len(batch_data[0]) == 4:  # 测试数据
                    # 测试时通常batch_size=1，但为了完整性也处理
                    batch_imgs = []
                    batch_masks = []
                    sizes = []
                    names = []
                    
                    for item in batch_data:
                        batch_imgs.append(item[0])
                        batch_masks.append(item[1])
                        sizes.append(item[2])
                        names.append(item[3])
                    
                    batch_imgs = jt.stack(batch_imgs, dim=0)
                    batch_masks = jt.stack(batch_masks, dim=0)
                    yield (batch_imgs, batch_masks, sizes[0], names[0])  # 测试时通常只关心第一个
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

# 设置jittor使用GPU
jt.flags.use_cuda = 1

parser = argparse.ArgumentParser(description="Jittor L2SKNet test")

parser.add_argument("--model_names", default='L2SKNet_UNet', type=str, nargs='+',
                    help="model_name: 'L2SKNet_UNet', 'L2SKNet_FPN', "
                         "'L2SKNet_1D_UNet', 'L2SKNet_1D_FPN'")
parser.add_argument("--dataset_names", default='NUDT-SIRST', type=str, nargs='+',
                    help="dataset_name: 'NUDT-SIRST', 'IRSTD-1K','SIRST','NUAA-SIRST'")
parser.add_argument("--dataset_dir", default='./data', type=str, help="train_dataset_dir")
parser.add_argument("--save", default='./log', type=str, help="Save path of checkpoints")
parser.add_argument("--seed", type=int, default=42, help="Threshold for test")
parser.add_argument("--test_epo", type=str, default='200', help="Number of epoch for test")

global opt
opt = parser.parse_args()
seed_jittor(opt.seed)



def test():
    if (opt.dataset_name == "NUDT-SIRST"):
        dataset_dir = r'./data/NUDT-SIRST/'
        test_set = NUDTSIRSTSetLoader(base_dir=dataset_dir, mode='test')
    elif (opt.dataset_name == "IRSTD-1K"):
        dataset_dir = r'./data/IRSTD-1K/'
        test_set = IRSTD1KSetLoader(base_dir=dataset_dir, mode='test')
    else:
        raise NotImplementedError

    param_path = "log/" + opt.dataset_name + "/" + opt.model_name + '/' + opt.test_epo + '.pth.tar'

    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    net = Net(model_name=opt.model_name)

    net.load_state_dict(jt.load(param_path)['state_dict'])
    net.eval()
    
    print('testing data=' + opt.dataset_name + ', model=' + opt.model_name + ', epoch=' + opt.test_epo)

    imgDir = "./result/" + opt.dataset_name + "/img/" + opt.model_name + "/"
    if not os.path.exists(imgDir):
        os.makedirs(imgDir)
    matDir = "./result/" + opt.dataset_name + "/mat/" + opt.model_name + "/"
    if not os.path.exists(matDir):
        os.makedirs(matDir)
        
    for idx_iter, (img, gt_mask, size, iname) in enumerate(test_loader):
        name = iname[0]
        pngname = name + ".png"
        matname = name + '.mat'
        with jt.no_grad():
            img = jt.array(img)
            pred = net.execute(img)
            pred = pred[:, :, :size[0], :size[1]]
            pred_out = pred.numpy().squeeze()
            pred_out_png = pred_out * 255

        cv2.imwrite(imgDir + pngname, pred_out_png.astype('uint8'))
        scio.savemat(matDir + matname, {'T': pred_out})


if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            test()
            print('\n')
