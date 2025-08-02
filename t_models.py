import argparse
from net import Net
import os
import time
import jittor as jt

# 设置jittor使用GPU
jt.flags.use_cuda = 1

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
model_name = 'L2SKNet_UNet'
input_img = jt.rand(1,1,256,256)
net = Net(model_name)

# 计算参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

params = count_parameters(net)
print(model_name)
print('Params: %.2fM' % (params/1e6))

# 注意：jittor没有直接的FLOPs计算工具，这里只显示参数量
print('Note: FLOPs calculation not available in Jittor version')