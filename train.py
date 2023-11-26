import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models import BasicBlock, EDDNet
from data import Dataset,Dataset1
from utils import AverageMeter, batch_PSNR, batch_ssim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.cudnn as cudnn

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="REDNet")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument('--image_path', type=str, default='/data/DIV2K_valid_HR/DIV2K_valid_HR')
parser.add_argument('--images_dir', type=str, default='/data/DIV2K_CROP_256_256')
parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=50, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=50, help='noise level used on validation set')
opt = parser.parse_args()


def main():
    # Load dataset
    print('Loading dataset ...\n')

    images_dir = os.getcwd() + opt.images_dir
    image_path = os.getcwd() + opt.image_path
    dataset_train = Dataset(images_dir)  # , opt.patch_size, opt.noiseL)
    dataset_val = Dataset1(image_path)
    loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batchSize, shuffle=True,
                              drop_last=True)
    loader_val = DataLoader(dataset=dataset_val, num_workers=0, batch_size=4, shuffle=True, drop_last=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model

    model = EDDNet(BasicBlock, 64)
    criterion = nn.MSELoss(reduction='sum')
    model = model.to(device)
    criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    for param_group in optimizer.param_groups:
        param_group["lr"] = opt.lr

    step = 0

    sgdr = CosineAnnealingLR(optimizer, opt.epochs * len(loader_train), eta_min=0.0, last_epoch=-1)
    for epoch in range(opt.epochs):

        epoch_losses = AverageMeter()
        print('learning rate %f' % param_group["lr"])
        # train
        for i, data in enumerate(loader_train):

            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data

            noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL / 255.)
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())

            out_train = model(imgn_train)

            loss = criterion(out_train, img_train) / (imgn_train.size()[0] * 2)
            epoch_losses.update(loss.item(), len(img_train))
            loss.backward()
            optimizer.step()
            sgdr.step()

            # results
            model.eval()
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            ssim_train = batch_ssim(out_train, img_train, 1.)
            if i % 20 == 0:
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f SSIM_train: %.4f" %
                      (epoch + 1, i + 1, len(loader_train), epoch_losses.avg, psnr_train, ssim_train))  # loss.item()

            step += 1
        psnr_val = 0
        ssim_val = 0
        val_losses = AverageMeter()
        with torch.no_grad():
            for j, data_val in enumerate(loader_val):
                img_val = data_val
                noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL / 255.)
                imgn_val = img_val + noise
                img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
                out_val = model(imgn_val)
                out_val = torch.clamp(out_val, 0., 1.)
                val_loss = criterion(out_val, img_val) / (imgn_val.size()[0] * 2)
                val_losses.update(val_loss.item(), len(img_val))
                psnr_val += batch_PSNR(out_val, img_val, 1.)
                ssim_val += batch_ssim(out_val, img_val, 1.)
            psnr_val /= len(loader_val)
            ssim_val /= len(loader_val)

        print("[epoch %d] Validation Loss: %.4f PSNR_val: %.4f SSIM_val: %.4f" %
              (epoch + 1, val_losses.avg, psnr_val, ssim_val))

        if (epoch + 1)%5 == 0:
            model_name = "model_epoch{}.pth".format(epoch + 1)
            torch.save(model, os.path.join(opt.outf, model_name))


if __name__ == "__main__":
    main()
