import argparse
import time
from osgeo import gdal
from torch.utils.data import DataLoader
import numpy as np
from data import *
from model import Generator
import os
import itertools

os.environ['PROJ_LIB'] = r'D:\Anaconda\Lib\site-packages\osgeo\data\proj'

def readTif_att(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset

def readTif(fileName):
    dataset = gdal.Open(fileName)  # 打开tif
    # 获取行数列数和地理信息
    width = dataset.RasterXSize  # 宽度
    height = dataset.RasterYSize  # 高度
    geo_information = dataset.GetGeoTransform()
    proj = dataset.GetProjection()  # 地图投影
    filevalue = np.array(dataset.ReadAsArray(0, 0, width, height), dtype=float)  # 将数据写成数组，对应栅格矩阵
    return filevalue

def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

# Multerrain
# testlr_folders = [r'./dataset/60s/test', r'./dataset_new/60s_terrain/slope/test']

# Sinterrain
testlr_folders = [r'./dataset_new/60s/test']

testhr_folders = r'./dataset_new/15s/test' #选择个空文件夹
testdataset = Dataset(testlr_folders, testhr_folders, lr_transform=lr_transform)
testdata_loader = DataLoader(testdataset, batch_size=1, shuffle=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--attdir_name', type=str, default=r'./dataset_new/15s/test')
parser.add_argument('--model_name', default='best.pth', type=str, help='generator model epoch name')
parser.add_argument('--outpath', default=r'./result_new/StfeTX4/', type=str, help='generator model epoch name')


opt = parser.parse_args(args=[])

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name
ATT_NAME = opt.attdir_name
OUT_PATH = opt.outpath
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH, exist_ok=True)
    print(f"文件夹已创建: {OUT_PATH}")
else:
    print(f"文件夹已存在: {OUT_PATH}")

upscale = 4
height = 32
width = 32
window_size = 8
model = Generator(16, upscale).to(device)
# model = CRAFT(upscale=upscale, img_size=(height, width), in_chans=1,
#                  window_size=window_size, depths=[6, 6, 6, 6, 6, 6],
#                  embed_dim=96, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2).to(device)
model.load_state_dict(torch.load('epochsX4/' + MODEL_NAME))
model.to(device)
model.eval()

# 列表保存生成的超分辨率图像
generated_images = []

with torch.no_grad():
    for idx, (lr_images, hr_image, lr_min_val, lr_max_val) in enumerate(testdata_loader):
        lr_images = lr_images.to(device)
        # lr_dbm, lr_BPI = torch.split(lr_images, 1, dim=1)
        outputs = model(lr_images)
        denormalized_outputs = min_max_denormalize(outputs, lr_min_val.to(device), lr_max_val.to(device))

        # 将生成的超分辨率图像保存到列表中
        generated_images.append(denormalized_outputs.cpu().numpy())


path_list = os.listdir(ATT_NAME)
path_list.sort() #对读取的路径进行排序
i = 0

# 创建一个循环迭代器
list_cycle = itertools.cycle(generated_images)

for image_path in path_list:
    head_tail = os.path.split(image_path)
    fname, ext = os.path.splitext(head_tail[1])
    base_name = os.path.basename(fname)
    # print(base_name)
    # 获取列表中的下一个矩阵
    image = next(list_cycle)
    # image = pil_image.open(args.image_file).convert('RGB')
    img_att = readTif_att(ATT_NAME + '/' + image_path)
    proj = img_att.GetProjection()
    geotrans = img_att.GetGeoTransform()
    # image = readTif(opt.image_name + '/' +image_path)

    # image = image.transpose(1, 2, 0)

    out_img = image[0, 0, :, :]
    writeTiff(out_img, geotrans, proj, OUT_PATH + '%s' % (base_name) + '.tif')
