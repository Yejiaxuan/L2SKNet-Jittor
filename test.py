import argparse
import cv2
import jittor as jt
import scipy.io as scio
import os
import random

from net import Net
from utils.utils import seed_jittor, get_optimizer
from utils.datasets import NUDTSIRSTSetLoader
from utils.dataloader import DataLoader

# 设置jittor使用GPU
jt.flags.use_cuda = 1

parser = argparse.ArgumentParser(description="Jittor L2SKNet test")

parser.add_argument("--model_names", default='L2SKNet_UNet', type=str, nargs='+',
                    help="model_name: 'L2SKNet_UNet', 'L2SKNet_FPN', "
                         "'L2SKNet_1D_UNet', 'L2SKNet_1D_FPN'")
parser.add_argument("--dataset_names", default='NUDT-SIRST', type=str, nargs='+',
                    help="dataset_name: 'NUDT-SIRST'")
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
        name = iname  # 由于batch_size=1，DataLoader直接返回字符串
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
