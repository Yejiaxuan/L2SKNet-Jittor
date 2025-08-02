import numpy as np
import jittor as jt
import tqdm
import os
import time

# 设置jittor使用GPU
jt.flags.use_cuda = 1

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from net import Net

model_name = 'L2SKNet_UNet'
input_img = jt.rand(1,1,256,256)
net = Net(model_name)
net.eval()

repetitions = 300

print('warm up ...\n')
with jt.no_grad():
    for _ in range(100):
        _ = net.execute(input_img)

# Jittor会自动同步，不需要手动同步
jt.sync_all()

timings = np.zeros((repetitions, 1))

print('testing ...\n')
with jt.no_grad():
    for rep in tqdm.tqdm(range(repetitions)):
        start_time = time.time()
        _ = net.execute(input_img)
        jt.sync_all()  # 确保计算完成
        end_time = time.time()
        curr_time = (end_time - start_time) * 1000  # 转换为毫秒
        timings[rep] = curr_time

avg = timings.sum()/repetitions
print('\navg={:.3f}ms\n'.format(avg))