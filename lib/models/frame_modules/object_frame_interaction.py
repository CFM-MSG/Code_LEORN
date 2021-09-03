import math

import torch
import torch.nn as nn
from models.transformer import  TransformerEncoderLayerWithBN
from models.utils import clones

from core.config import config


class JustPool(nn.Module):
    def __init__(self, cfg):
        super(JustPool, self).__init__()

    def forward(self, rcnn, query, rcnn_mask, rcnn_bbox):
        '''

        :param rcnn: batch * frame * object * hidden_size
        :param query: batch * hidden_size
        :param rcnn_mask: batch * frame * object
        :param rcnn_bbox: batch * frame * object * 4
        :return:
        '''
        object_num = torch.sum(rcnn_mask.int(), dim=2)  # batch * frame
        feature = torch.sum(rcnn, dim=2) / object_num[:, :, None]  # batch * frame * hidden_size
        return feature


class ObjectTextAtteneion(nn.Module):
    def __init__(self, hidden_size):
        super(ObjectTextAtteneion, self).__init__()
        self.hidden_size = hidden_size
        self.rcnn_linear = nn.Linear(hidden_size, hidden_size)
        self.text_linear = nn.Linear(hidden_size, hidden_size)
        self.att_linear = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            # nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        self.norm = nn.BatchNorm2d(hidden_size)

    def forward(self, rcnn, query):
        '''

        :param rcnn: batch * frame * object * OBJECT_INPUT_SIZE
        :param query: batch * hidden_size
        :return:
        '''
        if config.DEBUG:
            assert ~torch.isnan(rcnn).any()
        batch, frame, object_num, _ = rcnn.shape
        rcnn = rcnn.view(batch, -1, self.hidden_size)  # batch * frame 36 * 512
        rcnn = torch.relu(self.rcnn_linear(rcnn))  # batch * frame 36 * 512
        text = torch.relu(self.text_linear(query))  # batch * 512
        text = text[:, None, :].expand(batch, rcnn.shape[1], self.hidden_size)
        if config.DEBUG:
            assert ~torch.isnan(rcnn).any()
            assert ~torch.isnan(text).any()
        fused = torch.cat([rcnn, text], dim=2)  # batch * frame 36 * 1024
        if config.DEBUG:
            assert ~torch.isnan(fused).any()
        att = self.att_linear(fused)  # batch * frame 36 * 512
        # att = torch.softmax(self.att_linear(fused), dim=2)  # batch * frame 36 * 512
        if config.DEBUG:
            assert ~torch.isnan(att).any()
        enhanced_rcnn = (rcnn * att).view(batch, frame, object_num, -1)  # batch * frame 36 * 512
        enhanced_rcnn = self.norm(enhanced_rcnn.transpose(1, 3)).transpose(1, 3).contiguous()
        if config.DEBUG:
            assert ~torch.isnan(enhanced_rcnn).any()
        return enhanced_rcnn


class Temporal_Conv(nn.Module):
    def __init__(self, hidden_size, kernel_size, use_sentinel_vector=False):
        super(Temporal_Conv, self).__init__()
        self.use_sentinel_vector = use_sentinel_vector
        if use_sentinel_vector:
            self.temporal_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, stride=1, padding=kernel_size // 2)
        else:
            self.temporal_conv = nn.Conv2d(hidden_size, hidden_size, (1, kernel_size), stride=1,
                                           padding=(0, kernel_size // 2))

    def forward(self, x):
        '''

        :param x: batch * frame * object * hidden_size
        :return: batch * frame * object * hidden_size
        '''
        if self.use_sentinel_vector:
            sentinel_vector = x[:, :, 0, :]  # batch * frame * hidden_size
            sentinel_output = torch.relu(
                self.temporal_conv(sentinel_vector.transpose(1, 2)).transpose(1, 2) + sentinel_vector)
            x[:, :, 0, :] = sentinel_output
            return x
        else:
            return torch.relu(self.temporal_conv(x.transpose(1, 3)).transpose(1, 3) + x)


class ObjectTextInteractionBlock(nn.Module):
    def __init__(self, hidden_size, temporal_shift=0, norm='bn', temporal_conv=0, use_sentinel_vector=False,
                 text_attention=True, multi_attention=True, shift_sentinel_vector=True):
        self.hidden_size = hidden_size
        super(ObjectTextInteractionBlock, self).__init__()
        self.use_text_attention = text_attention
        if text_attention:
            self.object_text_attention = ObjectTextAtteneion(hidden_size)
        self.use_multi_attention = multi_attention
        self.transformer = TransformerEncoderLayerWithBN(hidden_size, 4, dim_feedforward=1024,
                                                         temporal_shift=temporal_shift, norm=norm,
                                                         shift_sentinel_vector=shift_sentinel_vector,
                                                         use_multi_att=multi_attention)
        self.temporal_conv = Temporal_Conv(hidden_size, temporal_conv,
                                           use_sentinel_vector=shift_sentinel_vector) if temporal_conv > 0 else None

    def forward(self, rcnn, query, rcnn_mask=None):
        '''
        TODO add mask
        :param rcnn: batch * frame * object * OBJECT_INPUT_SIZE
        :param query: batch * hidden_size
        :param rcnn_mask: batch * frame * object
        :return:
        '''
        batch, frame, object_num, _ = rcnn.shape
        feature = rcnn
        if self.use_text_attention:
            feature = self.object_text_attention(feature, query)  # batch * frame * 36 * 512
        if config.DEBUG:
            assert ~torch.isnan(feature).any()
        if rcnn_mask is not None:
            feature = feature * rcnn_mask[:, :, :, None]
        feature = self.transformer(feature, rcnn_mask).contiguous()
        if config.DEBUG:
            assert ~torch.isnan(feature).any()
        if self.temporal_conv is not None:
            feature = self.temporal_conv(feature).contiguous()
        return feature


class AttFusionBlock(nn.Module):
    def __init__(self, hidden_size, use_sentinel_vector=False):
        super(AttFusionBlock, self).__init__()
        self.text_linear = nn.Linear(hidden_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, object, query, mask, norm=True, feature_extraction=False):
        '''
        TODO add mask
        :param object: batch * frame * object * hidden_size
        :param query: batch * hidden_size
        :param mask: batch * frame * object
        :return:
        '''
        batch, frame, object_size, hidden_size = object.shape
        # object_num = torch.sum(mask.int(), dim=2)  # batch * frame
        rcnn = object.view(batch, -1, hidden_size)  # batch * frame 36 * 512
        text = self.text_linear(query)  # batch * hidden_size
        att = torch.matmul(rcnn, text[:, :, None])
        if norm:
            att = att / (torch.linalg.norm(rcnn, dim=2, keepdim=True))
            att = att / (torch.linalg.norm(text, dim=1, keepdim=True)[:, :, None])
        att = att.view(batch, frame, object_size, 1)  # batch * frame * 36 * 1
        att = att.masked_fill(~mask[:, :, :, None], float('-inf'))
        att = torch.softmax(att, dim=2)  # batch * frame * 36 * 1
        feature = torch.sum(object * att, dim=2)  # batch * frame * hidden_size
        feature = self.batch_norm(feature.transpose(1, 2)).transpose(1, 2)
        if feature_extraction:
            re_att = torch.sum(object * att, dim=3)
            return feature, re_att
        else:
            return feature


class ObjectTextInteraction(nn.Module):
    def __init__(self, cfg):
        super(ObjectTextInteraction, self).__init__()
        hidden_size = cfg.HIDDEN_SIZE
        if cfg.BBOX_ENCODE:
            self.position_encoding = nn.Sequential(
                nn.Linear(4, hidden_size // 4),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, hidden_size)
            )
            self.rcnn_linear = nn.Linear(cfg.OBJECT_INPUT_SIZE + hidden_size, hidden_size, bias=True)
        else:
            self.position_encoding = None
            self.rcnn_linear = nn.Linear(cfg.OBJECT_INPUT_SIZE, hidden_size, bias=True)

        self.use_sentinel_vector = cfg.SENTINEL_VECTOR if 'SENTINEL_VECTOR' in cfg else False
        self.use_frame_as_sentinel = cfg.FRAME_AS_SENTINEL if 'FRAME_AS_SENTINEL' in cfg else False
        if self.use_sentinel_vector:
            self.register_buffer('true_tensor', torch.Tensor([True]).bool())
            if self.use_frame_as_sentinel:
                self.frame_linear = nn.Linear(cfg.VIS_INPUT_SIZE, hidden_size, bias=True)
            else:
                self.sentinel_vector = nn.Parameter(torch.Tensor(hidden_size))
                bound = 1 / math.sqrt(hidden_size)
                nn.init.uniform_(self.sentinel_vector, -bound, bound)

        self.norm_type = cfg.NORM if 'NORM' in cfg else 'bn'
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(hidden_size)
        elif self.norm_type == 'ln':
            self.norm = nn.LayerNorm(hidden_size)

        temporal_conv = cfg.TEMPORAL_CONV if 'TEMPORAL_CONV' in cfg else 0

        if 'REASONING' not in cfg or cfg.REASONING:
            self.interaction_layers = clones(ObjectTextInteractionBlock(hidden_size, temporal_shift=cfg.TEMPORAL_SHIFT,
                                                                        norm=self.norm_type,
                                                                        temporal_conv=temporal_conv,
                                                                        use_sentinel_vector=self.use_sentinel_vector,
                                                                        text_attention=cfg.TEXT_ATT if 'TEXT_ATT' in cfg else True,
                                                                        multi_attention=cfg.MULTI_ATT if 'MULTI_ATT' in cfg else True,
                                                                        shift_sentinel_vector=cfg.SHIFT_SENTINEL_VECTOR if 'SHIFT_SENTINEL_VECTOR' in cfg else True),
                                             cfg.LAYER_NUM)
        else:
            self.interaction_layers = None
        self.use_attentive_pool = cfg.ATT_POOL if 'ATT_POOL' in cfg else True
        if self.use_attentive_pool:
            self.attendtive_pool = AttFusionBlock(hidden_size, use_sentinel_vector=self.use_sentinel_vector)

    def forward(self, rcnn, query, rcnn_mask, rcnn_bbox, feature_extraction=False):
        '''

        :param rcnn: batch * frame * object * hidden_size
        :param query: batch * hidden_size
        :param rcnn_mask: batch * frame * object
        :param rcnn_bbox: batch * frame * object * 4
        :return:
        '''
        if isinstance(rcnn, list):
            frame_feature = rcnn[0]
            rcnn = rcnn[1]
        batch, frame, object_num, _ = rcnn.shape
        is_query_list = isinstance(query, list)
        if is_query_list:
            assert len(query) >= len(self.interaction_layers) + 1

        if config.DEBUG:
            assert torch.sum(torch.isnan(rcnn).int()) == 0

        if self.use_frame_as_sentinel and self.use_sentinel_vector:
            frame_feature = self.frame_linear(frame_feature)  # batch * frame * 512

        if self.position_encoding is not None:
            bbox_encode = self.position_encoding(rcnn_bbox)
            rcnn = torch.cat([rcnn, bbox_encode], dim=3)
        feature = self.rcnn_linear(rcnn)
        feature = feature * rcnn_mask[:, :, :, None]
        if config.DEBUG:
            assert torch.sum(torch.isnan(feature).int()) == 0

        if self.norm_type == 'bn':
            feature = self.norm(feature.transpose(1, 3)).transpose(1, 3).contiguous()
        elif self.norm_type == 'ln':
            feature = self.norm(feature)

        if self.use_sentinel_vector:
            if self.use_frame_as_sentinel:
                feature = torch.cat([frame_feature[:, :, None, :], feature], dim=2)
                rcnn_mask = torch.cat([self.true_tensor.expand(batch, frame, 1), rcnn_mask], dim=2)
            else:
                hidden_size = self.sentinel_vector.shape[0]
                feature = torch.cat([self.sentinel_vector.expand(batch, frame, 1, hidden_size), feature], dim=2)
                rcnn_mask = torch.cat([self.true_tensor.expand(batch, frame, 1), rcnn_mask], dim=2)

        if config.DEBUG:
            assert torch.sum(torch.isnan(feature).int()) == 0

        if self.interaction_layers is not None:
            for i, i_layer in enumerate(self.interaction_layers):
                feature = i_layer(feature, query[1] if is_query_list else query, rcnn_mask)
                if config.DEBUG:
                    assert ~torch.isnan(feature).any()

        if self.use_attentive_pool:
            if feature_extraction:
                feature, att = self.attendtive_pool(feature, query[0] if is_query_list else query, rcnn_mask,
                                                    feature_extraction=True)
            else:
                feature = self.attendtive_pool(feature, query[0] if is_query_list else query, rcnn_mask)
        else:
            feature = feature * rcnn_mask[:, :, :, None]
            feature = torch.sum(feature, dim=2) / (torch.sum(rcnn_mask, dim=2)[:, :, None])
        if config.DEBUG:
            assert torch.sum(torch.isnan(feature).int()) == 0

        if feature_extraction:
            return feature, att  # batch * frame * 36 * 1
        else:
            return feature


if __name__ == '__main__':
    import torch.autograd.profiler as profiler
    from easydict import EasyDict as edict

    cfg = edict()
    cfg.OBJECT_INPUT_SIZE = 512
    cfg.OBJECT_NUM = 36
    cfg.VIS_INPUT_SIZE = 512
    cfg.HIDDEN_SIZE = 512
    cfg.LAYER_NUM = 4
    cfg.TEMPORAL_SHIFT = False

    q = torch.rand(4, 512)
    c3d = torch.rand(2, 16, 512)
    rcnn = torch.rand(4, 128, 36, 512)
    rcnn_mask = torch.randint(0, 2, (4, 128, 36))
    rcnn_bbox = torch.rand(4, 128, 36, 4)

    interaction = ObjectTextInteraction(cfg)

    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        out = interaction(rcnn, q, rcnn_mask, rcnn_bbox)

    # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
    print(out.shape)
