from torch import nn
import torch.nn.functional as F
from models.map_modules import get_padded_mask_and_weight


class MapConv(nn.Module):

    def __init__(self, cfg):
        super(MapConv, self).__init__()
        input_size = cfg.INPUT_SIZE  # 512
        hidden_sizes = cfg.HIDDEN_SIZES  # [512, 512, 512, 512, 512, 512, 512, 512]
        kernel_sizes = cfg.KERNEL_SIZES  # [5, 5, 5, 5, 5, 5, 5, 5]
        strides = cfg.STRIDES  # [1, 1, 1, 1, 1, 1, 1, 1]
        paddings = cfg.PADDINGS  # [16, 0, 0, 0, 0, 0, 0, 0]
        dilations = cfg.DILATIONS  # [1, 1, 1, 1, 1, 1, 1, 1]
        self.convs = nn.ModuleList()
        assert len(hidden_sizes) == len(kernel_sizes) \
               and len(hidden_sizes) == len(strides) \
               and len(hidden_sizes) == len(paddings) \
               and len(hidden_sizes) == len(dilations)
        channel_sizes = [input_size] + hidden_sizes
        for i, (k, s, p, d) in enumerate(zip(kernel_sizes, strides, paddings, dilations)):
            self.convs.append(nn.Conv2d(channel_sizes[i], channel_sizes[i + 1], k, s, p, d))

    def forward(self, x, mask):
        padded_mask = mask
        for i, pred in enumerate(self.convs):
            x = F.relu(pred(x))
            padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, pred)
            x = x * masked_weight
        return x  # batchsize * 512 * 16 * 16


class ResMapConv(nn.Module):

    def __init__(self, cfg):
        super(ResMapConv, self).__init__()
        input_size = cfg.INPUT_SIZE  # 512
        hidden_sizes = cfg.HIDDEN_SIZES  # [512, 512, 512, 512, 512, 512, 512, 512]
        kernel_sizes = cfg.KERNEL_SIZES  # [5, 5, 5, 5, 5, 5, 5, 5]
        strides = cfg.STRIDES  # [1, 1, 1, 1, 1, 1, 1, 1]
        paddings = cfg.PADDINGS  # [16, 0, 0, 0, 0, 0, 0, 0]
        dilations = cfg.DILATIONS  # [1, 1, 1, 1, 1, 1, 1, 1]
        self.convs = nn.ModuleList()
        assert len(hidden_sizes) == len(kernel_sizes) \
               and len(hidden_sizes) == len(strides) \
               and len(hidden_sizes) == len(paddings) \
               and len(hidden_sizes) == len(dilations)
        channel_sizes = [input_size] + hidden_sizes
        for i, (k, s, p, d) in enumerate(zip(kernel_sizes, strides, paddings, dilations)):
            self.convs.append(nn.Conv2d(channel_sizes[i], channel_sizes[i + 1], k, s, p, d))

        if 'NORM' not in cfg or cfg.NORM:
            self.bn_layers = nn.ModuleList(
                [nn.BatchNorm2d(hidden_sizes[i]) for i in range(0, len(hidden_sizes))])
            self.bn_layers.append(nn.BatchNorm2d(hidden_sizes[-1]))
        else:
            self.bn_layers = None

    def forward(self, x, mask):
        padded_mask = mask
        if self.bn_layers is not None:
            x = self.bn_layers[0](x)
        for i, pred in enumerate(self.convs):
            x = pred(x) + x
            if self.bn_layers is not None:
                x = self.bn_layers[i + 1](x)
            x = F.relu(x)
            padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, pred)
            x = x * masked_weight
        return x  # batchsize * 512 * 16 * 16