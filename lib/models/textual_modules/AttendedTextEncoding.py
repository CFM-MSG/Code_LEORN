import copy
import torch
from torch import nn

from easydict import EasyDict as edict

from models.textual_modules.textualEncoding import WLTextualEncoding


def clones(module, N):
    """
    Produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class AttendedTextEncoding(nn.Module):
    def __init__(self, hidden_size):
        super(AttendedTextEncoding, self).__init__()
        self.sentence_linear = nn.Linear(hidden_size, hidden_size)
        self.att_linear1 = nn.Linear(hidden_size * 2, hidden_size // 2)
        self.att_linear2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, cap_emb, sentence_encode, mask=None):
        '''

        :param cap_emb: batch * sentence_len * hidden size
        :param sentence_encode: batch * hidden size
        :param mask: batch * sentence_len
        :return: batch * hidden size
        '''
        sentence = torch.relu(self.sentence_linear(sentence_encode))
        fusion_emb = torch.cat([cap_emb, sentence[:, None, :].expand(cap_emb.shape)], dim=2)
        att = torch.relu(self.att_linear1(fusion_emb))
        att = self.att_linear2(att)  # batch * sentence_len * 1
        if mask is not None:
            att = att.masked_fill(~(mask.bool()), float('-inf'))
        att = torch.softmax(att.transpose(1, 2), dim=2)  # batch * sentence_len
        attended_emb = (att @ cap_emb).squeeze(1)
        return attended_emb


class MultiAttTextEncoding(nn.Module):
    def __init__(self, cfg):
        super(MultiAttTextEncoding, self).__init__()
        semantic_num = cfg.SEMANTIC_NUM
        hidden_size = cfg.TXT_HIDDEN_SIZE
        self.textual_encoding = WLTextualEncoding(cfg)
        self.attended_layers = clones(AttendedTextEncoding(hidden_size), semantic_num)

    def forward(self, x, mask):
        '''
        :param x: text batch * seq_len * input_size
        :param mask: batch * seq_len
        :return:
        '''
        sentence_emb, cap_emb = self.textual_encoding(x, mask)
        out = [sentence_emb]
        for layer in self.attended_layers:
            out.append(layer(cap_emb, sentence_emb, mask))
        return out, cap_emb


if __name__ == '__main__':
    c = edict()
    c.SEMANTIC_NUM = 3
    c.TXT_HIDDEN_SIZE = 10
    encode = MultiAttTextEncoding(c)
    a = torch.rand(2, 5, 10)
    b = torch.rand(2, 10)
    c = encode(a, b)
    print(c)
