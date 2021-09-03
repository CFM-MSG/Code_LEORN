import torch
from torch import nn


class FrameAvgPool(nn.Module):

    def __init__(self, cfg):
        super(FrameAvgPool, self).__init__()
        input_size = cfg.INPUT_SIZE  # 4096
        hidden_size = cfg.HIDDEN_SIZE  # 512
        kernel_size = cfg.KERNEL_SIZE  # 16
        stride = cfg.STRIDE
        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.avg_pool = nn.AvgPool1d(kernel_size, stride)

    def forward(self, visual_input):  # batchsize * 4096 * 256
        vis_h = torch.relu(self.vis_conv(visual_input))
        vis_h = self.avg_pool(vis_h)  # batchsize * 512 * 16
        return vis_h  # batchsize * 512 * 16


class MultiFeatureAvgPool_C(nn.Module):
    def __init__(self, cfg):
        super(MultiFeatureAvgPool_C, self).__init__()
        input_size = cfg.INPUT_SIZE  # 4096
        hidden_size = cfg.HIDDEN_SIZE  # 512
        kernel_size = cfg.KERNEL_SIZE  # 16
        stride = cfg.STRIDE
        # self.global_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.vis_conv = nn.Conv1d(hidden_size + input_size, hidden_size, 1, 1)
        self.avg_pool = nn.AvgPool1d(kernel_size, stride)
        # self.norm1 = nn.BatchNorm1d(hidden_size)
        self.norm = nn.BatchNorm1d(hidden_size)

    def forward(self, visual_input):  # batchsize * 4096 * 256
        assert isinstance(visual_input, list)
        rcnn_feature = visual_input[0].transpose(1, 2)
        global_feature = visual_input[1].transpose(1, 2)
        # global_feature = self.global_conv(global_feature)
        # global_feature = self.norm1(global_feature)
        # global_feature = torch.relu(global_feature)

        vis_h = torch.cat([rcnn_feature, global_feature], dim=1)
        vis_h = self.vis_conv(vis_h)
        vis_h = torch.relu(vis_h)
        vis_h = self.avg_pool(vis_h)  # batchsize * 512 * 16
        vis_h = self.norm(vis_h)

        return vis_h  # batchsize * 512 * 16


class MultiFeatureAvgPool(nn.Module):
    def __init__(self, cfg):
        super(MultiFeatureAvgPool, self).__init__()
        input_size = cfg.INPUT_SIZE  # 4096
        hidden_size = cfg.HIDDEN_SIZE  # 512
        kernel_size = cfg.KERNEL_SIZE  # 16
        stride = cfg.STRIDE
        self.global_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.vis_conv = nn.Conv1d(hidden_size + hidden_size, hidden_size, 1, 1)
        self.avg_pool = nn.AvgPool1d(kernel_size, stride)
        # self.norm1 = nn.BatchNorm1d(hidden_size)
        # self.norm2 = nn.BatchNorm1d(hidden_size)
        self.norm = nn.BatchNorm1d(hidden_size)
        # self.__init_fuse_conv__(hidden_size)

    def __init_fuse_conv__(self, hidden_size):
        weight1 = torch.eye(hidden_size, hidden_size)
        weight2 = torch.zeros(hidden_size, hidden_size)
        weight = torch.cat([weight1, weight2], dim=1).unsqueeze(2)
        weight = nn.Parameter(weight)
        bias = nn.Parameter(torch.zeros(hidden_size))
        self.vis_conv.weight = weight
        self.vis_conv.bias = bias

    def forward(self, visual_input):  # batchsize * 4096 * 256
        assert isinstance(visual_input, list)
        rcnn_feature = visual_input[0].transpose(1, 2)
        global_feature = visual_input[1].transpose(1, 2)
        global_feature = self.global_conv(global_feature)
        global_feature = torch.relu(global_feature)

        vis_h = torch.cat([rcnn_feature, global_feature], dim=1)
        vis_h = self.vis_conv(vis_h)
        vis_h = self.norm(vis_h)
        vis_h = torch.relu(vis_h)
        vis_h = self.avg_pool(vis_h)  # batchsize * 512 * 16

        return vis_h  # batchsize * 512 * 16

class MultiFeaturePoolAvg(nn.Module):
    def __init__(self, cfg):
        super(MultiFeatureAvgPool, self).__init__()
        input_size = cfg.INPUT_SIZE  # 4096
        hidden_size = cfg.HIDDEN_SIZE  # 512
        kernel_size = cfg.KERNEL_SIZE  # 16
        stride = cfg.STRIDE
        self.global_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.vis_conv = nn.Conv1d(hidden_size, hidden_size, 1, 1)
        self.avg_pool = nn.AvgPool1d(kernel_size, stride)
        self.fuse_conv = nn.Conv1d(hidden_size + hidden_size, hidden_size, 1, 1)
        self.__init_fuse_conv__(hidden_size)

    def __init_fuse_conv__(self, hidden_size):
        weight1 = torch.eye(hidden_size, hidden_size)
        weight2 = torch.zeros(hidden_size, hidden_size)
        weight = torch.cat([weight1, weight2], dim=1).unsqueeze(2)
        weight = nn.Parameter(weight)
        bias = nn.Parameter(torch.zeros(hidden_size))
        self.fuse_conv.weight = weight
        self.fuse_conv.bias = bias

    def forward(self, visual_input):  # batchsize * 4096 * 256
        assert isinstance(visual_input, list)
        rcnn_feature = visual_input[0].transpose(1, 2)
        global_feature = visual_input[1].transpose(1, 2)

        global_feature = self.global_conv(global_feature)
        global_feature = torch.relu(global_feature)
        global_feature = self.avg_pool(global_feature)

        vis_h = self.vis_conv(rcnn_feature)
        vis_h = torch.relu(vis_h)
        vis_h = self.avg_pool(vis_h)  # batchsize * 512 * 16

        vis_h = torch.cat([vis_h, global_feature], dim=1)
        vis_h = torch.relu(self.fuse_conv(vis_h))
        return vis_h  # batchsize * 512 * 16

class FrameMaxPool(nn.Module):

    def __init__(self, input_size, hidden_size, stride):
        super(FrameMaxPool, self).__init__()
        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.max_pool = nn.MaxPool1d(stride)

    def forward(self, visual_input):
        vis_h = torch.relu(self.vis_conv(visual_input))
        vis_h = self.max_pool(vis_h)
        return vis_h


class SequentialFrameAttentionPool(nn.Module):

    def __init__(self, cfg):
        super(SequentialFrameAttentionPool, self).__init__()
        input_size = cfg.INPUT_SIZE  # 4096
        self.hidden_size = cfg.HIDDEN_SIZE  # 512
        kernel_size = cfg.KERNEL_SIZE  # 16
        self.stride = cfg.STRIDE  # 16
        self.sqn = cfg.SQN_NUM
        # self.sqn = 2
        att_hidden_size = 256

        self.vis_conv = nn.Conv1d(input_size, self.hidden_size, 1, 1)
        self.avg_pool = nn.AvgPool1d(kernel_size, self.stride)

        self.global_emb_fn = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.sqn)])
        self.guide_emb_fn = nn.Sequential(*[
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU()
        ])

        self.att_fn1 = nn.Linear(self.hidden_size, att_hidden_size)
        self.att_fn2 = nn.Linear(self.hidden_size, att_hidden_size)
        self.att_fn3 = nn.Linear(att_hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        # self.drop = nn.Dropout()

        self.vis_out_conv = nn.Conv1d(self.hidden_size * self.sqn, self.hidden_size, 1, 1)

    def forward(self, visual_input):
        B, _, v_len = visual_input.shape
        vis_h = torch.relu(self.vis_conv(visual_input))

        avg_vis = self.avg_pool(vis_h)  # batchsize * 512 * 16

        seg_list = []
        att_seg_list = []
        for i in range(v_len // self.stride):
            vis_seg = vis_h[:, :, self.stride * i: self.stride * (i + 1)].transpose(1, 2)  # batchsize * 16 * 512
            avg_seg = avg_vis[:, :, i]
            prev_se = avg_seg.new_zeros(B, self.hidden_size)

            sqn_list = []
            att_list = []
            for m in range(self.sqn):
                v_n = self.global_emb_fn[m](avg_seg)
                g_n = torch.relu(self.guide_emb_fn(torch.cat([v_n, prev_se], dim=1)))  # batchsize * 512

                att = torch.tanh(self.att_fn1(g_n).unsqueeze(1).expand(-1, self.stride, -1) + self.att_fn2(vis_seg))
                att = self.att_fn3(att)

                att = self.softmax(att)  # batchsize * 16 * 1
                # TODO 使用sigmoid还是softmax
                # att = torch.sigmoid(att) * 2 - 1

                prev_se = torch.sum(vis_seg * att, dim=1)  # batchsize * 512
                sqn_list.append(prev_se)
                att_list.append(att)

            vis_new = torch.cat(sqn_list, dim=1)
            seg_list.append(vis_new)
            att_seg_list.append(torch.cat(att_list, dim=2))  # batchsize  * 16 * sqn

        vis_out = torch.relu(self.vis_out_conv(torch.stack(seg_list, dim=2)))
        att_out = torch.stack(att_seg_list, dim=1)  # batchsize * 16 * 16 * sqn

        return vis_out, att_out


class SequentialFrameWordAttentionPool(nn.Module):

    def __init__(self, cfg):
        super(SequentialFrameWordAttentionPool, self).__init__()
        input_size = cfg.INPUT_SIZE  # 4096
        self.hidden_size = cfg.HIDDEN_SIZE  # 512
        kernel_size = cfg.KERNEL_SIZE  # 16
        self.stride = cfg.STRIDE  # 16
        # self.sqn = cfg.SQN_NUM
        self.sqn = 3
        att_hidden_size = 256

        self.vis_conv = nn.Conv1d(input_size, self.hidden_size, 1, 1)
        self.avg_pool = nn.AvgPool1d(kernel_size, self.stride)

        self.global_emb_fn = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.sqn)])
        self.guide_emb_fn = nn.Sequential(*[
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU()
        ])

        self.att_fn1 = nn.Linear(self.hidden_size, att_hidden_size)
        self.att_fn2 = nn.Linear(self.hidden_size, att_hidden_size)
        self.att_fn3 = nn.Linear(att_hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        # self.drop = nn.Dropout()

        self.vis_out_conv = nn.Conv1d(self.hidden_size, self.hidden_size, 1, 1)

        self.text_linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, visual_input, text_feature):
        B, _, v_len = visual_input.shape
        vis_h = torch.relu(self.vis_conv(visual_input))

        avg_vis = self.avg_pool(vis_h)  # batchsize * 512 * 16

        text_att = self.text_linear(text_feature)  # batchsize * 512

        seg_list = []
        att_seg_list = []
        for i in range(v_len // self.stride):
            vis_seg = vis_h[:, :, self.stride * i: self.stride * (i + 1)].transpose(1, 2)  # batchsize * 16 * 512
            avg_seg = avg_vis[:, :, i].squeeze()
            prev_se = avg_seg.new_zeros(B, self.hidden_size)

            sqn_list = []
            att_list = []
            for m in range(self.sqn):
                v_n = self.global_emb_fn[m](avg_seg)
                g_n = torch.relu(self.guide_emb_fn(torch.cat([v_n, prev_se], dim=1)))  # batchsize * 512

                att = torch.tanh(self.att_fn1(g_n).unsqueeze(1).expand(-1, 16, -1) + self.att_fn2(vis_seg))
                att = self.att_fn3(att)
                att = self.softmax(att)  # batchsize * 16 * 1

                prev_se = torch.sum(vis_seg * att, dim=1)  # batchsize * 512
                sqn_list.append(prev_se)
                att_list.append(att)

            vis_for_att = torch.stack(sqn_list, dim=1)  # batch * sqn * hidden_size
            fuse_att = torch.softmax(torch.matmul(vis_for_att, text_att.unsqueeze(2)), dim=1)  # batch * sqn * 1

            vis_new = torch.sum(vis_for_att * fuse_att, dim=1)
            seg_list.append(vis_new)
            att_seg_list.append(torch.cat(att_list, dim=2))  # batchsize  * 16 * sqn
            # TODO 使用加权后的attention还是原始的attention

        vis_out = torch.relu(self.vis_out_conv(torch.stack(seg_list, dim=2)))
        att_out = torch.stack(att_seg_list, dim=1)  # batchsize * 16 * 16 * sqn

        return vis_out, att_out


class WordAttentionPool(nn.Module):

    def __init__(self, cfg):
        super(WordAttentionPool, self).__init__()
        input_size = cfg.INPUT_SIZE  # 4096
        hidden_size = cfg.HIDDEN_SIZE  # 512
        self.stride = cfg.STRIDE  # 16

        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.text_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, visual_input, text_feature):
        _, _, v_len = visual_input.shape  # batchsize * 4096 * 256

        vis_att = torch.relu(self.vis_conv(visual_input))  # batchsize * 512 * 256
        text_att = torch.relu(self.text_linear(text_feature))  # batch * 512

        att = torch.matmul(text_att.unsqueeze(1), vis_att).transpose(1, 2)  # batchsize * 256 * 1

        seg_list = []
        for i in range(v_len // self.stride):
            vis_seg = visual_input[:, :, self.stride * i: self.stride * (i + 1)].transpose(1,
                                                                                           2)  # batchsize * 16 * 4096
            att_seg = torch.softmax(att[:, self.stride * i: self.stride * (i + 1), :], dim=1)  # batchsize * 16 * 1
            vis_new = torch.sum(vis_seg * att_seg, dim=1)  # batchsize * 4096
            seg_list.append(vis_new)

        vis_out = torch.relu(self.vis_conv(torch.stack(seg_list, dim=2)))  # batchsize * 512 * 16

        return vis_out


class MovementFlowAvgPool(nn.Module):
    def __init__(self, cfg):
        super(MovementFlowAvgPool, self).__init__()
        input_size = cfg.INPUT_SIZE  # 4096
        hidden_size = cfg.HIDDEN_SIZE  # 512
        kernel_size = cfg.KERNEL_SIZE  # 16
        stride = cfg.STRIDE  # 16
        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.vis_flow_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.avg_pool = nn.AvgPool1d(kernel_size, stride)

        self.fusion_conv = nn.Conv1d(hidden_size * 2, hidden_size, 1, 1)

    def forward(self, visual_input):  # batchsize * 4096 * 256
        B, H, l = visual_input.size()
        vis_flow = torch.zeros(B, H, l).type_as(visual_input)
        for i in range(l - 1):
            vis_flow[:, :, i] = visual_input[:, :, i + 1] - visual_input[:, :, i]
        vis_flow[:, :, l - 1] = vis_flow[:, :, l - 2]
        vis_h = torch.relu(self.vis_conv(visual_input))
        vis_flow_h = torch.relu(self.vis_conv(vis_flow))
        vis_h = self.avg_pool(vis_h)  # batchsize * 512 * 16
        vis_flow_h = self.avg_pool(vis_flow_h)

        vis_h = torch.relu(self.fusion_conv(torch.cat([vis_h, vis_flow_h], dim=1)))

        return vis_h  # batchsize * 512 * 16
