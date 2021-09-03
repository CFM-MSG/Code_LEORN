import torch
from torch import nn
from core.config import config
import models.frame_modules as frame_modules
import models.prop_modules as prop_modules
import models.map_modules as map_modules
import models.fusion_modules as fusion_modules
import models.textual_modules as textual_modules


class LRHandler:
    def __init__(self, model_size, warmup, factor):
        self.factor = factor
        self.model_size = model_size
        self.warmup = warmup

    def get_lr(self, step):
        if step < 1:
            step = 1
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


class LEORN(nn.Module):
    def __init__(self):
        super(LEORN, self).__init__()

        self.textual_encoding = getattr(textual_modules, config.TAN.TEXTUAL_MODULE.NAME)(
            config.TAN.TEXTUAL_MODULE.PARAMS)
        self.object_interaction_layer = getattr(frame_modules, config.TAN.OBJECT_MODULE.NAME)(
            config.TAN.OBJECT_MODULE.PARAMS)
        self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS)
        self.prop_layer = getattr(prop_modules, config.TAN.PROP_MODULE.NAME)(config.TAN.PROP_MODULE.PARAMS)
        self.fusion_layer = getattr(fusion_modules, config.TAN.FUSION_MODULE.NAME)(config.TAN.FUSION_MODULE.PARAMS)
        self.map_layer = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS)
        self.pred_layer = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)

    def forward(self, textual_input, textual_mask, rcnn_input, rcnn_mask, rcnn_bbox):
        """

        :param textual_input: batch * num * embed_size
        :param textual_mask: batch * num
        :param rcnn_input: batch * frame * object * rcnn_input_size
        :param rcnn_mask: batch * frame * object
        :param rcnn_bbox: batch * frame * object * 4
        :return:
        """
        tex_encode, _ = self.textual_encoding(textual_input, textual_mask)
        if isinstance(tex_encode, list):
            object_text_encode = tex_encode[1]
            fusion_tex_encode = tex_encode[0]
        else:
            object_text_encode = fusion_tex_encode = tex_encode
        vis = self.object_interaction_layer(rcnn_input, object_text_encode, rcnn_mask, rcnn_bbox)
        vis_h = self.frame_layer(vis.transpose(1, 2))  # batch * 512 * 16
        map_h, map_mask = self.prop_layer(vis_h)  # batch * 512 * 16 * 16
        fused = self.fusion_layer(fusion_tex_encode, textual_mask, map_h, map_mask)
        fused_h = self.map_layer(fused, map_mask)
        prediction = self.pred_layer(fused_h) * map_mask

        if config.DEBUG:
            assert torch.sum(torch.isnan(vis).int()) == 0
            assert torch.sum(torch.isnan(vis_h).int()) == 0
            assert torch.sum(torch.isnan(map_h).int()) == 0
            assert torch.sum(torch.isnan(fused).int()) == 0
            assert torch.sum(torch.isnan(fused_h).int()) == 0
            assert torch.sum(torch.isnan(prediction).int()) == 0
        return prediction, map_mask

    def extract_features(self, textual_input, textual_mask, rcnn_input, rcnn_mask, rcnn_bbox):
        tex_encode, _ = self.textual_encoding(textual_input, textual_mask)
        if isinstance(tex_encode, list):
            object_text_encode = tex_encode[1]
            fusion_tex_encode = tex_encode[0]
        else:
            object_text_encode = fusion_tex_encode = tex_encode
        vis, features = self.object_interaction_layer(rcnn_input, object_text_encode, rcnn_mask, rcnn_bbox,
                                            feature_extraction=True)
        return features  # batch * frame * 36 * 1

    def adjust_lr(self, optimizer, t):
        # optimizer.param_groups[0]['lr'] = self.lr_handler.get_lr(t)
        pass


class LEORN_F(nn.Module):
    def __init__(self):
        super(LEORN_F, self).__init__()
        self.textual_encoding = getattr(textual_modules, config.TAN.TEXTUAL_MODULE.NAME)(
            config.TAN.TEXTUAL_MODULE.PARAMS)
        self.object_interaction_layer = getattr(frame_modules, config.TAN.OBJECT_MODULE.NAME)(
            config.TAN.OBJECT_MODULE.PARAMS)
        self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS)
        self.prop_layer = getattr(prop_modules, config.TAN.PROP_MODULE.NAME)(config.TAN.PROP_MODULE.PARAMS)
        self.fusion_layer = getattr(fusion_modules, config.TAN.FUSION_MODULE.NAME)(config.TAN.FUSION_MODULE.PARAMS)
        self.map_layer = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS)
        self.pred_layer = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)

    def forward(self, textual_input, textual_mask, visual_input, rcnn_input, rcnn_mask, rcnn_bbox):
        """

        :param textual_input: batch * num * embed_size
        :param textual_mask: batch * num
        :param rcnn_input: batch * frame * object * rcnn_input_size
        :param rcnn_mask: batch * frame * object
        :param rcnn_bbox: batch * frame * object * 4
        :return:
        """
        tex_encode, _ = self.textual_encoding(textual_input, textual_mask)
        if isinstance(tex_encode, list):
            object_text_encode = tex_encode[1]
            fusion_tex_encode = tex_encode[0]
        else:
            object_text_encode = fusion_tex_encode = tex_encode
        vis = self.object_interaction_layer(rcnn_input, object_text_encode, rcnn_mask, rcnn_bbox)
        vis_h = self.frame_layer([vis, visual_input])  # batch * 512 * 16
        map_h, map_mask = self.prop_layer(vis_h)  # batch * 512 * 16 * 16
        fused_h = self.fusion_layer(fusion_tex_encode, textual_mask, map_h, map_mask)
        fused_h = self.map_layer(fused_h, map_mask)
        prediction = self.pred_layer(fused_h) * map_mask

        return prediction, map_mask

    def extract_features(self, textual_input, textual_mask, visual_input):
        tex_encode = self.textual_encoding(textual_input)
        vis_h = self.frame_layer(visual_input.transpose(1, 2))  # batchsize * 512 * 16
        map_h, map_mask = self.prop_layer(vis_h)  # batchsize * 512 * 16 * 16
        fused_h = self.fusion_layer(tex_encode, textual_mask, map_h, map_mask)
        fused_h = self.map_layer(fused_h, map_mask)
        # prediction = self.pred_layer(fused_h)[:, :, 15:31, 0:16] * map_mask
        prediction = self.pred_layer(fused_h) * map_mask

        return fused_h, prediction

    def get_parameters(self):
        if config.TRAIN.FINE_TUNE:
            trans_params = list(map(id, self.object_interaction_layer.parameters()))
            trans_params += list(map(id, self.textual_encoding.parameters()))

            base_params = filter(lambda p: id(p) not in trans_params, self.parameters())
            params = [
                {'params': self.object_interaction_layer.parameters(), 'lr': 0.000001},
                {'params': self.textual_encoding.parameters(), 'lr': 0.0001},
                {'params': base_params}
            ]
            return params
        else:
            return self.parameters()

    def _load_params(self, state_dict, module_name, strict=True):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.split('.')[1] == module_name:
                new_state_dict[k[int(7 + len(module_name) + 1):]] = v
        getattr(self, module_name).load_state_dict(new_state_dict, strict=strict)

    def load_object_params(self, state_dict):
        self._load_params(state_dict, 'object_interaction_layer')

