import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
from tqdm import tqdm
from time import time

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder


ROOT = 'dataset3/train/'
NAME = 'my_dink34_2'
mylog = open('logs/'+NAME+'.log','w')

SHAPE = (256,256)
batchsize = 2
solver = MyFrame(DinkNet34, dice_bce_loss, 2e-4)
no_optim = 0
total_epoch = 100
train_epoch_best_loss = 100.

# imagelist = filter(lambda x: x.find('sat')!=-1, os.listdir(ROOT))
imagelist = filter(lambda x: x.find('sat')!=-1, os.listdir(ROOT))
trainlist = map(lambda x: x[:-8], imagelist)
trainlist = list(trainlist)
dataset = ImageFolder(trainlist, ROOT)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=4)

tic = time()
for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    print('---------- Epoch:'+str(epoch)+ ' ----------')
    print('Train:')
    for img, mask in tqdm(data_loader_iter,ncols=20,total=len(data_loader_iter)):
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader_iter)

    mylog.write('********'+'\n')
    mylog.write('epoch:'+str(epoch)+'    time:'+str(int(time()-tic))+'\n')
    mylog.write('train_loss:'+str(train_epoch_loss)+'\n')
    mylog.write('SHAPE:'+str(SHAPE)+'\n')
    print('********')
    print('epoch:',epoch,'    time:',int(time()-tic))
    print('train_loss:',train_epoch_loss)
    print('SHAPE:',SHAPE)

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('weights/'+NAME+'.th')
    if no_optim > 6:
        print (mylog, 'early stop at %d epoch' % epoch)
        print ('early stop at %d epoch' % epoch)
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load('weights/'+NAME+'.th')
        solver.update_lr(5.0, factor = True, mylog = mylog)
    mylog.flush()

mylog.write('Finish!')
print('Finish!')
mylog.close()
