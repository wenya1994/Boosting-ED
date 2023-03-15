import os
import os.path as osp
import torch
import glob
import time
from tqdm import tqdm
from utils import loss_fn, Cross_entropy_loss
from evaluate import BER
from torch.utils import data
from torch.utils.data import DataLoader
from dataset import BSDS_Dataset
from config import train_images_dir, train_annos_dir, test_images_dir, encoder, Batch_size
from model.SDTR import mit_b1
from model.SDTR import mit_b2


def train(model, train_loader, epoch):
    train_loss = 0
    model.train()
    # 训练
    for x, y in tqdm(train_loader):  # i表示i-th batch
        x, y = x.cuda(), y.cuda()
        _y_pred = model(x)  # _y_pred接收的类型为list
        loss = torch.zeros(1).cuda()
        # 单监督
        # y, y_pred = torch.squeeze(y), torch.squeeze(_y_pred[-1])
        # loss = loss_fn(5, y_pred, y)
        # 深监督
        for j, y_pred in enumerate(_y_pred):
            y, y_pred = torch.squeeze(y), torch.squeeze(y_pred)
            loss = loss + Cross_entropy_loss(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item()
    train_loss = train_loss / len(train_loader.dataset)
    # # 计算测试集BER
    # with torch.no_grad():
    #     test(model, test_dl, test_images, pre_path)
    #     ber, acc = BER(pre_path, label_path)

    # -----------------------保存模型-----------------------------------
    PATH = basic_path_dir + encoder + '_' + str(epoch + 1) + '_' + str(round(train_loss, 2)) + '.pth'
    torch.save(model.state_dict(), PATH)
    spl = "	"  # excell 分隔符
    log = str(epoch + 1) + encoder + spl + str(round(train_loss, 2))
    print(log)
    with open("log.txt", "a") as f:
        f.write(log + "\n")


if __name__ == "__main__":
    # 提取训练和测试数据的路径
    train_dataset = BSDS_Dataset(root="D:/ywy/3.Dada&Code/Dataset/BSDS", split='train')
    test_dataset = BSDS_Dataset(root="D:/ywy/3.Dada&Code/Dataset/BSDS", split='test')
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, drop_last=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False, shuffle=False)
    test_list = [osp.split(i.rstrip())[1] for i in test_dataset.file_list]
    assert len(test_list) == len(test_loader)
    # ---------------------------------------模型初始化--------------------------------------------
    if encoder == "m1":
        model = mit_b1()
    else:
        model = mit_b2()

    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # -----------------------------------训练开始--------------------------
    basic_path_dir = './MyPth/'
    if not os.path.exists(basic_path_dir):
        os.makedirs(basic_path_dir)

    epochs = 60
    time_start = time.time()
    for epoch in range(epochs):
        train(model, train_loader, epoch)
    time_end = time.time()
    print('totally cost', time_end - time_start)
