import glob
import os
import numpy as np
import os.path as osp
import cv2
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision
from dataset import BSDS_Dataset
# from models import RCF
from model.SDTR import mit_b1
from model.SDTR import mit_b2

encoder = "m1"


def single_scale_test(model, test_loader, test_list, save_dir):
    model.eval()
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    for idx, image in enumerate(test_loader):
        image = image.cuda()
        _, _, H, W = image.shape
        results = model(image)
        all_res = torch.zeros((len(results), 1, H, W))
        for i in range(len(results)):
            all_res[i, 0, :, :] = results[i]
        filename = osp.splitext(test_list[idx])[0]
        torchvision.utils.save_image(1 - all_res, osp.join(save_dir, '%s.jpg' % filename))
        result = torch.sigmoid(results[-1])
        fuse_res = torch.squeeze(result.detach()).cpu().numpy()
        # fuse_res = torch.squeeze(result[-1].detach()).cpu().numpy() # 原
        # fuse_res = ((1 - fuse_res) * 255).astype(np.uint8)
        fuse_res = (fuse_res * 255).astype(np.uint8)
        cv2.imwrite(osp.join(save_dir, '%s.png' % filename), fuse_res)
        # print('\rRunning single-scale test [%d/%d]' % (idx + 1, len(test_loader)), end='')
    print('Running single-scale test done')


def multi_scale_test(model, test_loader, test_list, save_dir):
    model.eval()
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    scale = [0.5, 1, 1.5]
    for idx, image in enumerate(test_loader):
        in_ = image[0].numpy().transpose((1, 2, 0))
        _, _, H, W = image.shape
        ms_fuse = np.zeros((H, W), np.float32)
        for k in range(len(scale)):
            im_ = cv2.resize(in_, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2, 0, 1))
            results = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
            fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse_res = cv2.resize(fuse_res, (W, H), interpolation=cv2.INTER_LINEAR)
            ms_fuse += fuse_res
        ms_fuse = ms_fuse / len(scale)
        ### rescale trick
        # ms_fuse = (ms_fuse - ms_fuse.min()) / (ms_fuse.max() - ms_fuse.min())
        filename = osp.splitext(test_list[idx])[0]
        ms_fuse = ((1 - ms_fuse) * 255).astype(np.uint8)
        cv2.imwrite(osp.join(save_dir, '%s_ms.png' % filename), ms_fuse)
        # print('\rRunning multi-scale test [%d/%d]' % (idx + 1, len(test_loader)), end='')
    print('Running multi-scale test done')


if __name__ == "__main__":
    Batch_size = 1
    test_dataset = BSDS_Dataset(root="D:/ywy/3.Dada&Code/Dataset/BSDS", split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False, shuffle=False)
    test_list = [osp.split(i.rstrip())[1] for i in test_dataset.file_list]
    assert len(test_list) == len(test_loader)

    pth_path = "MyPth/m1_2_-7.06.pth"
    print(pth_path)
    if torch.cuda.is_available():
        # ---------------------------------------模型初始化--------------------------------------------
        if encoder == "m1":
            model = mit_b1().cuda()
        else:
            model = mit_b2().cuda()
        model.load_state_dict(torch.load(pth_path))  # 使用GPU测试
        print("-----------Loading checkpoint to GPU-----------")
    else:
        if encoder == "m1":
            model = mit_b1()
        else:
            model = mit_b2()
        model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))  # 使用CPU测试
        print("-----------Loading checkpoint to CPU-----------")

    print("-----------Single-scale Testing Start-----------")
    # test(model, test_dl, test_list, pre_path)
    # ber, acc = BER(pre_path, label_path)
    # print("BER::%.2f, Acc:%.2f" % (ber, acc))

    print('Performing the testing...')
    single_scale_test(model, test_loader, test_list, "./result")
    # multi_scale_test(model, test_loader, test_list, args.save_dir)
