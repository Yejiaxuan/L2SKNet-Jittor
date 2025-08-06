import random
import jittor as jt


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        # num_workers在Jittor中暂不支持，保留接口兼容性
  
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
            
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]
            
            # 根据数据格式处理
            if len(batch_data[0]) == 2:  # 训练数据 (img, mask)
                if self.batch_size == 1:
                    yield batch_data[0]
                else:
                    imgs = jt.stack([item[0] for item in batch_data], dim=0)
                    masks = jt.stack([item[1] for item in batch_data], dim=0)
                    yield (imgs, masks)
                    
            elif len(batch_data[0]) == 4:  # 测试数据 (img, mask, size, name)
                if self.batch_size == 1:
                    yield batch_data[0]
                else:
                    # 测试时通常batch_size=1，但为完整性处理多batch情况
                    imgs = jt.stack([item[0] for item in batch_data], dim=0)
                    masks = jt.stack([item[1] for item in batch_data], dim=0)
                    yield (imgs, masks, batch_data[0][2], batch_data[0][3])
            else:
                raise ValueError(f"Unsupported data format with {len(batch_data[0])} elements")
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size