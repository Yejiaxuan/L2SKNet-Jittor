import numpy as np
import jittor as jt

class mIoU():
    
    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return float(pixAcc), mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

def batch_pix_accuracy(output, target):   
    # 统一转换为numpy数组进行处理
    if isinstance(output, jt.Var):
        output = output.data
    if isinstance(target, jt.Var):
        target = target.data
        
    if len(target.shape) == 3:
        target = np.expand_dims(target.astype(np.float32), axis=1)
    elif len(target.shape) == 4:
        target = target.astype(np.float32)
    else:
        raise ValueError("Unknown target dimension")

    output = output.astype(np.float32)
    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).astype(np.float32)
    pixel_labeled = (target > 0).astype(np.float32).sum()
    pixel_correct = (((predict == target).astype(np.float32))*((target > 0)).astype(np.float32)).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled
    
def batch_intersection_union(output, target):
    mini = 1
    maxi = 1
    nbins = 1
    
    # 统一转换为numpy数组进行处理
    if isinstance(output, jt.Var):
        output = output.data
    if isinstance(target, jt.Var):
        target = target.data
        
    predict = (output > 0).astype(np.float32)
    if len(target.shape) == 3:
        target = np.expand_dims(target.astype(np.float32), axis=1)
    elif len(target.shape) == 4:
        target = target.astype(np.float32)
    else:
        raise ValueError("Unknown target dimension")
    
    intersection = predict * ((predict == target).astype(np.float32))

    area_inter, _  = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union