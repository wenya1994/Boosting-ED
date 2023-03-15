import torch
import torchvision.models
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import glob
from matplotlib import pyplot as plt


# training transformation
p = 0.5
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224,224)),
    transforms.RandomHorizontalFlip(p=p),
    transforms.RandomRotation(45),
    # transforms.ColorJitter(brightness=1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# training dataset类
class dataset(data.Dataset):
    def __init__(self, img_paths, labels):
        self.imgs = img_paths
        self.labels = labels

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]

        pil_img = Image.open(img)
        data = transform(pil_img)
        return data, label

    def __len__(self):
        return len(self.imgs)







