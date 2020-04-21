import torch
from torch import nn

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean=(0.5, 0.5, 0.5), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        print(self.weight.data.size())
        self.bias.data = sign * torch.Tensor(rgb_mean)
        self.requires_grad = False
    #python test.py --weights_path YHSR_epoch_199.pth --image_path 0810.png --outputs_dir ./
    #python train.py --images_dir ../DIV2K_train_HR --outputs_dir ./ --jpeg_quality 40
