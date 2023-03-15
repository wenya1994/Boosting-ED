import torch
import torchvision.models
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import glob
from matplotlib import pyplot as plt
from model import MobileNetV2
from dataset import dataset
from torch import nn
import torch.optim as optim
import os
import PIL
from torch.autograd import Variable
import cv2 as cv

# testing transformation
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 给定pth路径
pth_name = 'Epoch_149_TrainLOSS_0.027_TestACC_0.814.pth'


def check_mkdir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)


all_sample_paths = glob.glob(r'D:\ywy\3.Dada&Code\Dataset\PASCAL\aug_data\0.0_0\*.jpg')

if __name__ == '__main__':
    idx2label = {0: 'animal', 1: 'building', 2: 'people', 3: 'plant', 4: 'scenery', 5: 'vehicles'}
    count = 0
    # 新建存放简单，复杂样本的文件夹
    p = 0.20
    easy_dir = 'easy_' + str(p)
    hard_dir = 'hard_' + str(p)
    check_mkdir(easy_dir)
    check_mkdir(hard_dir)
    # 加载模型
    if torch.cuda.is_available():
        net = MobileNetV2(num_classes=6).cuda()
        net.load_state_dict(torch.load(os.path.join('pth', pth_name), 'cuda'))
    else:
        net = MobileNetV2(num_classes=6)
        net.load_state_dict(torch.load(os.path.join('pth', pth_name), 'cpu'))
    net.eval()
    # 预测，根据最大预测概率阈值筛选出样本，提取文件名，读取，另存
    with torch.no_grad():
        for idx, image_path in enumerate(all_sample_paths):
            print('---[%s/%s] image processing---' % (idx + 1, len(all_sample_paths)))
            file_name = image_path.split('\\')[-1]
            img = Image.open(image_path)
            if torch.cuda.is_available():
                img_var = Variable(transform_test(img).unsqueeze(0)).cuda()
            else:
                img_var = Variable(transform_test(img).unsqueeze(0))
            out = net(img_var)
            # 提取预测值
            out = out.data.cpu()
            # 提取下标
            tensor_idx = torch.argmax(out, dim=1)
            # 转换类型
            idx = int(np.array(tensor_idx)[0])
            label = idx2label[idx]
            # 根据下标提取最大的概率值
            image = cv.imread(image_path)
            max_prob = float(np.array(out[0][idx]))
            if max_prob >= p:
                count += 1
                new_dir = os.path.join(easy_dir, file_name)
            else:
                new_dir = os.path.join(hard_dir, file_name)
            cv.imwrite(new_dir, image)
        # print("easy ratio: %s" % (count/len(all_sample_paths)))
