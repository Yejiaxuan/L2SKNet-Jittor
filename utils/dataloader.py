import random
import jittor as jt


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
                    # 保持 (1, H, W) 的通道维，并添加 batch 维度 -> (1, 1, H, W)
                    img = img.unsqueeze(0)
                    mask = mask.unsqueeze(0)
                    yield (img, mask)
                elif len(item) == 4:  # 测试数据 (img, mask, size, name)
                    img, mask, size, name = item[0], item[1], item[2], item[3]
                    # 添加batch维度: (1, H, W) -> (1, 1, H, W)
                    # 保持 (1, H, W) 的通道维，并添加 batch 维度 -> (1, 1, H, W)
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