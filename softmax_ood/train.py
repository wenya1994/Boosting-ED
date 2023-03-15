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


def fit(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0

    model.train()
    for x, y in trainloader:
        if torch.cuda.is_available():
            y = torch.tensor(y, dtype=torch.long)
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()

    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / total

    test_correct = 0
    test_total = 0
    test_running_loss = 0

    model.eval()  # 关闭dropout   关闭BN
    with torch.no_grad():
        for x, y in testloader:
            y = torch.tensor(y, dtype=torch.long)
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total

    print('epoch: ', epoch,
          'loss： ', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss： ', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
          )
    pth_name = 'Epoch_{:0>3}_TrainLOSS_{:.3f}_TestACC_{:.3f}.pth'.format(epoch, epoch_loss, epoch_test_acc)
    torch.save(model.state_dict(), os.path.join(save_dir, pth_name))
    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc


if __name__ == '__main__':
    save_dir = 'pth'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    # 读取所有图像的路径，生成one-hot编码
    all_imgs_path = sorted(glob.glob(r'Data/*/*.jpg'))
    all_labels_name = []
    for path in all_imgs_path:
        all_labels_name.append(path.strip().split('\\')[-2])

    # 进行one-hot编码
    unique_labels = np.unique(all_labels_name)

    label_to_index = dict((v, k) for k, v in enumerate(unique_labels))
    index_to_label = dict((v, k) for k, v in
                          label_to_index.items())  # {0: 'animal', 1: 'building', 2: 'people', 3: 'plant', 4: 'scenery', 5: 'vehicles'}
    print(index_to_label)


    all_labels = [label_to_index.get(name) for name in all_labels_name]

    # 乱序
    # np.random.seed(2022)
    index = np.random.permutation(len(all_imgs_path))
    all_imgs_path = np.array(all_imgs_path)[index]
    all_labels = np.array(all_labels)[index]

    # 切分训练集和测试集
    s = int(len(all_imgs_path) * 0.8)
    train_imgs = all_imgs_path[:s]
    train_labels = all_labels[:s]

    test_imgs = all_imgs_path[s:]
    test_labels = all_labels[s:]

    train_dataset = dataset(train_imgs, train_labels)
    test_dataset = dataset(test_imgs, test_labels)

    train_dl = data.DataLoader(train_dataset, batch_size=48, shuffle=True)
    test_dl = data.DataLoader(test_dataset, batch_size=1)

    # 数据可视化
    # img_batch, label_batch = next(iter(train_dl))
    # plt.figure(figsize=(12,8))
    # for i, (img, label) in enumerate(zip(img_batch[:6], label_batch[:6])):
    #     img = img.permute(1,2,0).numpy()
    #     plt.subplot(2,3,i+1)
    #     plt.title(index_to_label.get(label.item()))
    #     plt.imshow(img)
    # plt.show()

    # 加载预训练模型
    net = MobileNetV2(num_classes=6)
    model_weight_path = "mobilenet_v2.pth"
    pre_weights = torch.load(model_weight_path, map_location='cuda')
    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    # strict = False 表示仅读取可以匹配的权重
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)
    # freeze features weights
    for param in net.features.parameters():
        param.requires_grad = False
    net.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 150

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for epoch in range(epochs):
        epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                     net,
                                                                     train_dl,
                                                                     test_dl)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)
