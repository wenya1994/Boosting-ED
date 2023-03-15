import torch
import cv2
import numpy as np
import os.path as osp


# ----------------数据增大和Tensor化------------------------

# 创建transform
class BSDS_Dataset(torch.utils.data.Dataset):
    def __init__(self, root='D:/ywy/3.Dada&Code/Dataset/BSDS', split='train', transform=False):
        super(BSDS_Dataset, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.root2 = 'data/PASCAL'  # 我加的
        if self.split == 'train':
            self.file_list = osp.join(self.root, 'train_pair.lst')
            """以下加"""
            # self.pascal_file_list = osp.join(self.root2, 'train_pair.lst')  # 我加
            # with open(self.file_list, 'r') as f:
            #     self.file_list = f.readlines()
            # with open(self.pascal_file_list, 'r') as f:
            #     self.pascal_file_list = f.readlines()
            # self.file_list += self.pascal_file_list
            """以上加"""
        elif self.split == 'test':
            self.file_list = osp.join(self.root, 'test.lst')
            """以下加"""
            # with open(self.file_list, 'r') as f:
            #     self.file_list = f.readlines()
            """以上加"""
        else:
            raise ValueError('Invalid split type!')
        with open(self.file_list, 'r') as f:
            self.file_list = f.readlines()
        self.mean = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)

    def __len__(self):
        return len(self.file_list)  # 原
        # return len(self.file_list) + len(self.pascal_file_list) # 改

    def __getitem__(self, index):
        if self.split == 'train':
            img_file, label_file = self.file_list[index].split()

            label = cv2.imread(osp.join(self.root, label_file), 0)
            label = cv2.resize(label, (416,416))
            label = np.array(label, dtype=np.float32)

            label = label[np.newaxis, :, :]
            label[label == 0] = 0
            label[np.logical_and(label > 0, label < 127.5)] = 2
            label[label >= 127.5] = 1
            # -----------------主要修改该地方-------------------
            label[np.logical_and(label > 0, label < 127.5)] = 2
            label[label >= 127.5] = 1
        else:
            img_file = self.file_list[index].rstrip()

        img = cv2.imread(osp.join(self.root, img_file))
        img = cv2.resize(img, (416,416))
        img = np.array(img, dtype=np.float32)
        img = (img - self.mean).transpose((2, 0, 1))
        # resize
        if self.split == 'train':
            return img, label
        else:
            return img
