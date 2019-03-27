import argparse
import os

from dataset import Dataset
from ar_0hyenet import AR_0hyeNet

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import transforms
from torchvision.models.vgg import vgg16

from utils import AverageMeter
from tqdm import tqdm

if __name__ == '__main__':
    '''
    It enables benchmark mode in cudnn.
    benchmark mode is good whenever your input sizes for your network do not vary. 
    This way, cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time). 
    This usually leads to faster runtime.
    But if your input sizes changes at each iteration, 
    then cudnn will benchmark every time a new size appears, 
    possibly leading to worse runtime performances.
    '''
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--jpeg_quality', type=int, default=40)
    parser.add_argument('--patch_size', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    torch.manual_seed(opt.seed)

    transforms_train = transforms.Compose([transforms.ToTensor()])
    model = AR_0hyeNet().to(device)

    if opt.resume:
        if os.path.isfile(opt.resume):
            state_dict = model.state_dict()
            for n, p in torch.load(opt.resume, map_location=lambda storage, loc: storage).items():
                if n in state_dict.keys():
                    state_dict[n].copy_(p)
                else:
                    raise KeyError(n)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    dataset = Dataset(opt.images_dir, opt.patch_size, opt.jpeg_quality,transforms=transforms_train)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=True)

    print(device)

    vgg = vgg16(pretrained=True).cuda()
    loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
    for param in loss_network.parameters():
        param.requires_grad = False

    for epoch in range(opt.num_epochs):
        epoch_losses = AverageMeter()

        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, opt.num_epochs))
            for data in dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)

                loss = criterion(outs, labels)
                #perception_loss = criterion(loss_network(outs), loss_network(labels))

                #loss = loss + perception_loss*0.0001

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(inputs))

        torch.save(model.state_dict(), os.path.join(opt.outputs_dir, '{}_epoch_{}.pth'.format("AR_0hyeNet", epoch)))
