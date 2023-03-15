import torch.nn as nn
import torch
import torch.nn.functional as F


class SFF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SFF, self).__init__()
        self.features = []
        for i in range(2):
            self.features.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
        self.sig = nn.Sigmoid()

    def forword(self, glob_feature, local_feature):
        cat = torch.cat([glob_feature, local_feature], 1)
        for f in self.features:
            cat = f(cat)
        cat = self.sig(cat)
        return cat


if __name__ == "__main__":
    sff = SFF(512, 512)
    glob = torch.rand(2, 512, 13, 13)
    lobal = torch.rand(2, 512, 13, 13)
    cat = sff(glob, lobal)
