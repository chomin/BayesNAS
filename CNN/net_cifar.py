import torch
import torch.nn as nn
import torch.nn.functional as F
from CNN.cell import *

DEVICE = torch.device('cuda')


class CReduct(nn.Module):
    def __init__(self, in_channels, out_channels, size_reduct=False, save_device=torch.device('cpu')):
        super(CReduct, self).__init__()
        self.save_device = save_device

        self.size_reduct = size_reduct
        if size_reduct:
            self.size_reduct = nn.Conv2d(in_channels, in_channels, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        self.data = None

    def forward(self, x):
        x.retain_grad()
        self.data = {'input': x}
        if self.size_reduct:
            x = self.size_reduct(x)
            self.data['reducted'] = x.detach().to(self.save_device)
        x = self.bn(x)
        self.data['bn'] = x.detach().to(self.save_device)
        x = F.relu(x)
        x.retain_grad()
        self.data['relu'] = x
        x = self.conv(x)
        self.data['conv'] = x.detach().to(self.save_device)

        return x


class NetWork(nn.Module):
    def __init__(self, num_classes, save_device=torch.device('cpu'), training=True, drop_prob=0.2):
        super(NetWork, self).__init__()
        self.save_device = save_device
        self.drop_prob = drop_prob

        cell_channels = 36
        stem_channels = cell_channels * 3

        self.conv = nn.Conv2d(3, stem_channels, 3, padding=1, bias=False)

        self.reduct_append_loc = [5, 11]
        c_prev, c_prev_prev = stem_channels, stem_channels
        for i in range(18):
            if i - 1 in self.reduct_append_loc:
                size_reduct = True
            else:
                size_reduct = False
            setattr(self, "reduct_" + str(2 * i), CReduct(c_prev, cell_channels, save_device=save_device))
            setattr(self, "reduct_" + str(2 * i + 1),
                    CReduct(c_prev_prev, cell_channels, save_device=save_device, size_reduct=size_reduct))

            setattr(self, "norm_cell_" + str(i),
                    NormalCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob))
            c_prev, c_prev_prev = 4 * cell_channels, c_prev
            if i in self.reduct_append_loc:
                cell_channels *= 2
                idx = self.reduct_append_loc.index(i)
                setattr(self, "_reduct_" + str(2 * idx), CReduct(c_prev, cell_channels, save_device=save_device))
                setattr(self, "_reduct_" + str(2 * idx + 1),
                        CReduct(c_prev_prev, cell_channels, save_device=save_device))
                setattr(self, "reduct_cell_" + str(idx),
                        ReductCell(cell_channels, save_device=save_device, training=training, drop_prob=drop_prob))
                c_prev, c_prev_prev = 4 * cell_channels, c_prev

        self.bn = nn.BatchNorm2d(c_prev)
        self.pool = nn.AvgPool2d(8)
        self.linear = nn.Linear(c_prev, num_classes)

    def forward(self, x):
        x_prev_prev = self.conv(x)
        x_prev = x_prev_prev

        for i in range(18):
            reduct_prev = getattr(self, "reduct_" + str(2 * i))
            reduct_prev_prev = getattr(self, "reduct_" + str(2 * i + 1))
            norm_cell = getattr(self, "norm_cell_" + str(i))
            x_prev_, x_prev_prev_ = reduct_prev(x_prev), reduct_prev_prev(x_prev_prev)
            x_prev, x_prev_prev = norm_cell(x_prev_, x_prev_prev_), x_prev
            if i in self.reduct_append_loc:
                idx = self.reduct_append_loc.index(i)
                reduct_prev = getattr(self, "_reduct_" + str(2 * idx))
                reduct_prev_prev = getattr(self, "_reduct_" + str(2 * idx + 1))
                reduct_cell = getattr(self, "reduct_cell_" + str(idx))

                x_prev_, x_prev_prev_ = reduct_prev(x_prev), reduct_prev_prev(x_prev_prev)
                x_prev, x_prev_prev = reduct_cell(x_prev_, x_prev_prev_), x_prev

        x = self.bn(x_prev)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)

        return x

    def no_drop(self):
        for idx in range(18):
            norm_cell = getattr(self, 'norm_cell_' + str(idx))
            norm_training = False
        for idx in range(2):
            reduct_cell = getattr(self, 'reduct_cell_' + str(idx))
            reduct_training = False

    def drop(self):
        for idx in range(18):
            norm_cell = getattr(self, 'norm_cell_' + str(idx))
            norm_training = True
        for idx in range(2):
            reduct_cell = getattr(self, 'reduct_cell_' + str(idx))
            reduct_training = True

    def drop_prob_update(self, epoch, total_epoch):
        drop_prob = self.drop_prob * epoch / total_epoch
        for idx in range(18):
            norm_cell = getattr(self, 'norm_cell_' + str(idx))
            norm_drop_prob = drop_prob
        for idx in range(2):
            reduct_cell = getattr(self, 'reduct_cell_' + str(idx))
            reduct_drop_prob = drop_prob
