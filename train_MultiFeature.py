import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from data import *
from model import Generator
import sys
import copy
from torch.utils.data import DataLoader
from DBM_features import *
import math

seed = 10
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
random.seed(seed)
np.random.seed(seed)
batchsize = 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 100
lr = 1e-4
upscale = 4
height = 32
width = 32
slope_weight = 0
cutdepth_weight = 0
#multerrain
# trainlr_folders = [r'./dataset_new/60s/train', r'./dataset_new/60s_terrain/slope/train']
# trainhr_folder = r'./dataset_new/15s/train'
#
# vallr_folders = [r'./dataset_new/60s/val', r'./dataset_new/60s_terrain/slope/val']
# valhr_folder = r'./dataset_new/15s/val'

#singleterrain
trainlr_folders = [r'./dataset_new/60s/train']
trainhr_folder = r'./dataset_new/15s/train'

vallr_folders = [r'./dataset_new/60s/val']
valhr_folder = r'./dataset_new/15s/val'

traindataset = Dataset(trainlr_folders, trainhr_folder, lr_transform=lr_transform)
valdataset = Dataset(vallr_folders, valhr_folder, lr_transform=lr_transform)

traindata_loader = DataLoader(traindataset, batch_size=batchsize, shuffle=True, drop_last=True)
valdata_loader = DataLoader(valdataset, batch_size=1)
#
model = Generator(16, upscale).to(device)


content_criterion = nn.L1Loss().to(device)
mse_criterion = nn.MSELoss().to(device)
optimiser = optim.Adam(model.parameters(), lr=lr)

logfilpath ='./'+'_eval_rmse.txt'

print(device)
if torch.cuda.device_count() > 1:
    print("You have", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

best_weights = copy.deepcopy(model.state_dict())
best_epoch = 0
best_rmse = float('inf')

for epoch in range(1, epochs + 1):
    mean_dbm_loss = 0.0
    mean_generator_content_loss = 0.0
    mean_generator_slope_loss = 0.0
    mean_generator_cutdepth_loss = 0.0
    mean_generator_total_loss = 0.0
    model.train()
    for lr_images, hr_image, lr_min_vals, lr_max_vals in traindata_loader:
        lr_images = lr_images.to(device)
        # lr_dbm, lr_BPI = torch.split(lr_images, 1, dim=1)
        lr_dbm = lr_images
        srdbm = model(lr_dbm)
        hrdbm = hr_image.to(device)

        optimiser.zero_grad()
        loss_depth = content_criterion(srdbm, hrdbm)
        #slope loss
        hr_slope = Slope_net(hrdbm)
        sr_slope = Slope_net(srdbm)
        loss_slope = content_criterion(hr_slope, sr_slope)
        #cutdepth loss
        hr_cutdepth = Cutdepth_net(hrdbm)
        sr_cutdepth = Cutdepth_net(srdbm)
        loss_cutdepth = content_criterion(hr_cutdepth, sr_cutdepth)
        #total loss
        loss_total = loss_depth + slope_weight * loss_slope + cutdepth_weight * loss_cutdepth
        loss_total.backward()
        optimiser.step()

        dbm_loss = 0
        for j in range(batchsize):
            hr_normalize = min_max_denormalize(hrdbm[j], lr_min_vals[j], lr_max_vals[j])
            sr_normalize = min_max_denormalize(srdbm[j], lr_min_vals[j], lr_max_vals[j])
            dbm_loss += math.sqrt(mse_criterion(hr_normalize, sr_normalize))
        dbm_loss = dbm_loss / batchsize
        sys.stdout.write(
            '\r[%d/%d] Generator_Loss (Content/slope/cutdepth/totalloss/dbmloss): %.8f - %.8f - %.8f - %.8f - %8f' % (
                epoch, epochs,
                loss_depth, loss_slope, loss_cutdepth,
                loss_total,
                dbm_loss))
        # sys.stdout.write(
        #     '\r[%d/%d] Generator_Loss (Content/dbmloss): %.8f - %.8f ' % (
        #         epoch, epochs,
        #         generator_content_loss,
        #         dbm_loss))

        mean_generator_content_loss += loss_depth
        mean_generator_slope_loss += loss_slope
        mean_generator_cutdepth_loss += loss_cutdepth
        mean_generator_total_loss += loss_total
        mean_dbm_loss += dbm_loss
    mean_generator_content_loss = mean_generator_content_loss / len(traindataset)
    mean_generator_slope_loss = mean_generator_slope_loss / len(traindataset)
    mean_generator_cutdepth_loss = mean_generator_cutdepth_loss / len(traindataset)
    mean_generator_total_loss = mean_generator_total_loss / len(traindataset)
    mean_dbm_loss = mean_dbm_loss / len(traindataset) * batchsize
    sys.stdout.write(
        '\r[%d/%d] Generator_Loss (MeanContent/Meanslope/Meancutdepth/Meantotalloss/Meandbmloss): %.8f - %.8f - %.8f - %.8f - %8f\n' % (
            epoch, epochs,
            mean_generator_content_loss, mean_generator_slope_loss, mean_generator_cutdepth_loss,
            mean_generator_total_loss,
            mean_dbm_loss))

    model.eval()
    eval_rmse = 0
    with torch.no_grad():
        mean_dbm_loss2 = 0.0
        for vallr_images, valhr_image, vallrmin, vallrmax in valdata_loader:
            vallr_images = vallr_images.to(device)
            vallr_dbm = vallr_images
            # vallr_dbm, vallr_BPI = torch.split(vallr_images, 1, dim=1)
            valsrdbm = model(vallr_dbm)
            valhrdbm = valhr_image.to(device)
            vallrmin, vallrmax = vallrmin.to(device), vallrmax.to(device)

            valhr_normalize = min_max_denormalize(valhrdbm, vallrmin, vallrmax)
            valsr_normalize = min_max_denormalize(valsrdbm, vallrmin, vallrmax)
            eval_rmse += math.sqrt(mse_criterion(valhr_normalize, valsr_normalize))

        eval_rmse = eval_rmse/len(valdataset)
        eval_ave_errlog = open(logfilpath, 'a')
        eval_ave_errlog.write(
            '\r[%d/%d] eval_RMSE: %.4f' % (
                epoch, epochs, eval_rmse))
        eval_ave_errlog.close()
        print('Epoch: [%d/%d], eval_rmse: %.4f' % (epoch, epochs, eval_rmse))

    # Do checkpointing
    torch.save(model.state_dict(), 'epochsX4/netG_%d.pth' % (epoch))

    if eval_rmse < best_rmse:
        best_epoch = epoch
        best_rmse = eval_rmse
        best_weights = copy.deepcopy(model.state_dict())
    print('best epoch: {}, rmse: {:.4f}'.format(best_epoch, best_rmse))
    torch.save(best_weights, 'epochsX4/' + 'best.pth')

