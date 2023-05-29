import os
import random

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from icnet import ICNet
from config import cfg
from loading_data import loading_data
from utils2 import *
from timer import Timer
import pdb
from loss import ICNetLoss

exp_name = cfg.TRAIN.EXP_NAME
log_txt = cfg.TRAIN.EXP_LOG_PATH + '/' + exp_name + '.txt'
writer = SummaryWriter(cfg.TRAIN.EXP_PATH+ '/' + exp_name)

pil_to_tensor = standard_transforms.ToTensor()
train_loader, val_loader, restore_transform = loading_data()

def main():

    cfg_file = open('/content/drive/MyDrive/project-WasteSemSeg-main_2/config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_txt, 'a') as f:
            f.write(''.join(cfg_lines) + '\n\n\n\n')
    if len(cfg.TRAIN.GPU_ID)==1:
        torch.cuda.set_device(cfg.TRAIN.GPU_ID[0])
    torch.backends.cudnn.benchmark = True

    net = []   
    net = ICNet()

    if len(cfg.TRAIN.GPU_ID)>1:
        net = torch.nn.DataParallel(net, device_ids=cfg.TRAIN.GPU_ID).cuda()
    else:
        net=net.cuda()

    net.train()
    criterion = ICNetLoss(cfg.DATA.NUM_CLASSES, ignore_index=-1)
    criterion.cuda()
    optimizer = optim.Adam(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.NUM_EPOCH_LR_DECAY, gamma=cfg.TRAIN.LR_DECAY)
    _t = {'train time' : Timer(),'val time' : Timer()} 
    validate(val_loader, net, criterion, optimizer, -1, restore_transform)
    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        _t['train time'].tic()
        train(train_loader, net, criterion, optimizer, epoch)
        _t['train time'].toc(average=False)
        print('training time of one epoch: {:.2f}s'.format(_t['train time'].diff))
        _t['val time'].tic()
        validate(val_loader, net, criterion, optimizer, epoch, restore_transform)
        _t['val time'].toc(average=False)
        print('val time of one epoch: {:.2f}s'.format(_t['val time'].diff))


def train(train_loader, net, criterion, optimizer, epoch):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
   
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        optimizer.step()



def validate(val_loader, net, criterion, optimizer, epoch, restore):
    net.eval()
    criterion.cpu()
    input_batches = []
    output_batches = []
    label_batches = []
    mean_classes = []
    iou_ = 0.0
    mean_classe0 = 0
    mean_classe1 = 0
    mean_classe2 = 0
    mean_classe3 = 0
    mean_classe4 = 0
    mean_tot = 0
    
    for vi, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()
        outputs = net(inputs)
        out0, out1, out2, out3 = outputs

        out0 = F.softmax(out0, dim=1)  # Apply softmax activation function along the channel dimension
        # For each pixel, determine the class with highest probability
        max_value, predicted = torch.max(out0.data, 1) 
        
        input_batches.append(inputs)
        output_batches.append(predicted)
        label_batches.append(labels)

        mean0, mean1, mean2, mean3, mean4, mean5 = calculate_mean_iu([predicted.unsqueeze_(1).data.cpu().numpy()], 
                                        [labels.unsqueeze_(1).data.cpu().numpy()], cfg.DATA.NUM_CLASSES)
        mean_classe0 += mean0
        mean_classe1 += mean1
        mean_classe2 += mean2
        mean_classe3 += mean3
        mean_classe4 += mean4
        mean_tot += mean5
        
    print(f"Class 0: {mean_classe0 / len(val_loader):.4f}")
    print(f"Class 1: {mean_classe1 / len(val_loader):.4f}")
    print(f"Class 2: {mean_classe2 / len(val_loader):.4f}")
    print(f"Class 3: {mean_classe3 / len(val_loader):.4f}")
    print(f"Class 4: {mean_classe4 / len(val_loader):.4f}")
    print(f"Class tot: {mean_tot / len(val_loader):.4f}")
  
  
    # Calculate average IoU score over all classes
    #mean_classes[5] =float (iou_ / cfg.DATA.NUM_CLASSES)
    #print(f"Mean IoU: {mean_iou:.4f}")

    net.train()
    criterion.cuda()

if __name__ == '__main__':
    main()








