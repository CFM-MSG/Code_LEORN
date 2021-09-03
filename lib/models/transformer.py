import copy
import torch
import math
from torch import nn
import torch.nn.functional as F
from core.config import config

from models.temporal_shift import Temporal_Shift

class TransformerEncoderLayerWithBN(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.05, activation="relu", temporal_shift=0,
                 norm='bn', shift_sentinel_vector=False, use_multi_att=True):
        super(TransformerEncoderLayerWithBN, self).__init__()
        self.norm = norm
        self.shift_sentinel_vector = shift_sentinel_vector
        self.use_multi_att = use_multi_att

        if use_multi_att:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.temporal_shift = Temporal_Shift(inplace=False, shift_length=temporal_shift) if temporal_shift > 0 else None
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear3 = nn.Linear(dim_feedforward,
                                 dim_feedforward) if shift_sentinel_vector and temporal_shift > 0 else None

        if self.norm == 'bn':
            self.norm1 = nn.BatchNorm2d(d_model)
            self.norm2 = nn.BatchNorm2d(d_model)
        elif self.norm == 'ln':
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        assert activation == 'relu' or activation == 'gelu'
        self.activation = F.relu if activation == 'relu' else F.gelu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayerWithBN, self).__setstate__(state)

    def forward(self, src, mask=None):
        r"""Pass the x through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required). batch * frame(temporal) * object * hidden_size
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        batch, temporal, object_num, hidden_size = src.shape
        src = src.view(-1, object_num, hidden_size).transpose(0, 1)
        if mask is not None:
            key_padding_mask = mask.view(-1, object_num)
        if config.DEBUG:
            assert ~torch.isnan(src).any()
        if self.use_multi_att:
            src2 = self.self_attn(src, src, src, key_padding_mask=~key_padding_mask)[0]
        else:
            src2 = src
        if config.DEBUG:
            if torch.isnan(src2).any():
                torch.save([src, src2, key_padding_mask, self.self_attn],
                           'error{}.pt'.format(torch.distributed.get_rank()))
            assert ~torch.isnan(src2).any()
        src = src + self.dropout1(src2)
        src = src.transpose(0, 1).view(batch, temporal, object_num, -1)
        if mask is not None:
            src = src * mask[:, :, :, None]
        if self.norm == 'bn':
            src = self.norm1(src.transpose(1, 3)).transpose(1, 3)
        elif self.norm == 'ln':
            src = self.norm1(src)
        if config.DEBUG:
            assert ~torch.isnan(src).any()

        src2 = self.activation(self.linear1(src))
        if self.temporal_shift is not None:
            if self.shift_sentinel_vector:
                sentinel_vector = src2[:, :, 0, :]
                sentinel_vector = self.temporal_shift(sentinel_vector)
                src2[:, :, 0, :] = self.activation(self.linear3(sentinel_vector))
                # src2[:, :, 0, :] = sentinel_vector
            else:
                src2 = self.temporal_shift(src2)
        src2 = self.linear2(self.dropout(src2))
        src = src + self.dropout2(src2)
        if mask is not None:
            src = src * mask[:, :, :, None]
        if self.norm == 'bn':
            src = self.norm2(src.transpose(1, 3)).transpose(1, 3)
        elif self.norm == 'ln':
            src = self.norm2(src)
        if config.DEBUG:
            assert ~torch.isnan(src).any()
        return src


if __name__ == '__main__':
    q = torch.rand(2, 512)
    c3d = torch.rand(2, 512)
    rcnn = torch.rand(2, 36, 512)

    encoder = CrossModalEncoder(512)  # 实现有问题
    encoder2 = CrossModalEncoder(512)

    out = encoder(q, rcnn, rcnn)
    out2 = encoder2(c3d, out, out)

    print(out2.shape)
