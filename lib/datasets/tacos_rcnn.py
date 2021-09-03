""" Dataset loader for the TACoS dataset """
import os
import json
import numpy as np

import h5py
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext

import datasets
from . import average_to_fixed_length
from core.eval import iou
from core.config import config


class TACoS_RCNN(data.Dataset):
    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    def __init__(self, split, rcnn_threshold=0.0, training=False):
        super(TACoS_RCNN, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split
        self.object_num = config.DATASET.OBJECT_NUM
        self.rcnn_threshold = rcnn_threshold
        self.negative_sample = config.DATASET.NEGATIVE_SAMPLE if 'NEGATIVE_SAMPLE' in config.DATASET and training else False

        # val_1.json is renamed as val.json, val_2.json is renamed as test.json
        with open(os.path.join(self.data_dir, '{}.json'.format(split)), 'r') as f:
            annotations = json.load(f)
        anno_pairs = []
        for vid, video_anno in annotations.items():
            duration = video_anno['num_frames'] / video_anno['fps']
            for timestamp, sentence in zip(video_anno['timestamps'], video_anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    anno_pairs.append(
                        {
                            'video': vid,
                            'duration': duration,
                            'times': [max(timestamp[0] / video_anno['fps'], 0),
                                      min(timestamp[1] / video_anno['fps'], duration)],
                            'description': sentence,
                        }
                    )
        self.annotations = anno_pairs
        self.pos_len = len(anno_pairs)

    def __getitem__(self, index):
        if self.negative_sample:
            if index % 2 == 0:
                negative = False
            else:
                negative = True
            index = index // 2
        else:
            negative = None

        if negative:
            video_id = self.annotations[index]['video']
            gt_s_time, gt_e_time = [0, 0]
            neg_index = np.random.choice(
                list(range(0, max(0, index - 5))) + list(range(min(self.pos_len, index + 5), self.pos_len)))
            sentence = self.annotations[neg_index]['description']
            duration = self.annotations[index]['duration']
        else:
            video_id = self.annotations[index]['video']
            gt_s_time, gt_e_time = self.annotations[index]['times']
            sentence = self.annotations[index]['description']
            duration = self.annotations[index]['duration']

        word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in sentence.split()], dtype=torch.long)
        word_vectors = self.word_embedding(word_idxs)

        # visual_input, visual_mask = self.get_video_features(video_id)
        # visual_input = sample_to_fixed_length(visual_input, random_sampling=config.DATASET.RANDOM_SAMPLING)
        # visual_input = average_to_fixed_length(visual_input)

        rcnn_input, rcnn_conf, rcnn_bbox = self.get_rcnn_features(video_id)
        # TODO mask
        rcnn_mask = rcnn_conf > self.rcnn_threshold
        # rcnn_mask = torch.ones_like(rcnn_conf).bool()
        # rcnn_input, rcnn_mask, rcnn_bbox = sample_to_fixed_length(rcnn_input, rcnn_mask, rcnn_bbox)
        rcnn_input = rcnn_input * rcnn_mask[:, :, None]
        rcnn_bbox = rcnn_bbox * rcnn_mask[:, :, None]

        num_clips = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
        if negative:
            overlaps = np.zeros([num_clips, num_clips])  # 16 * 16
        else:
            s_times = torch.arange(0, num_clips).float() * duration / num_clips
            e_times = torch.arange(1, num_clips + 1).float() * duration / num_clips
            overlaps = iou(torch.stack([s_times[:, None].expand(-1, num_clips),
                                        e_times[None, :].expand(num_clips, -1)], dim=2).view(-1, 2).tolist(),
                           torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips, num_clips)

        item = {
            # 'visual_input': visual_input,
            # 'vis_mask': visual_mask,
            'rcnn_input': rcnn_input,
            'rcnn_mask': rcnn_mask,
            'rcnn_bbox': rcnn_bbox,
            'anno_idx': index,
            'word_vectors': word_vectors,
            'duration': duration,
            'reg_gt': torch.tensor([gt_s_time, gt_e_time]),
            'txt_mask': torch.ones(word_vectors.shape[0], 1),
            'map_gt': torch.from_numpy(overlaps),
            'description': sentence
        }

        return item

    def __len__(self):
        return self.pos_len * 2 if self.negative_sample else self.pos_len

    def get_video_features(self, vid):
        assert config.DATASET.VIS_INPUT_TYPE == 'c3d'
        with h5py.File(os.path.join(self.data_dir, 'tall_c3d_features.hdf5'), 'r') as video_feature_bank:
            features = torch.from_numpy(video_feature_bank[vid][:])
        if config.DATASET.NORMALIZE:
            features = F.normalize(features, dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask

    def get_rcnn_features(self, vid):
        data_dir = '/mnt/ssd/std/wgm/2D-TAN'
        with h5py.File(os.path.join(data_dir, 'tacos_rcnn_features_256_{}.hdf5').format(self.split), 'r') as rcnn_feature_bank:
            data = torch.from_numpy(rcnn_feature_bank[vid][:])[:, :self.object_num, :]
            features = data[:, :, :2048]  # frame * 36 * 2048
            confidences = data[:, :, 2048]  # frame * 36
            bboxs = data[:, :, 2049:]  # frame * 36 * 4
        if config.DATASET.NORMALIZE:
            features = F.normalize(features, dim=2)
        return features, confidences, bboxs

    def get_rcnn_features1(self, vid):
        with h5py.File(os.path.join(self.data_dir, 'tacos_rcnn_features_untc.hdf5'), 'r') as rcnn_feature_bank:
            features = torch.from_numpy(rcnn_feature_bank['feats'][vid][:])[:, :self.object_num, :]  # frame * 36 * 2048
            confidences = torch.from_numpy(rcnn_feature_bank['confidence'][vid][:])[:, :self.object_num]  # frame * 36
            bboxs = torch.from_numpy(rcnn_feature_bank['bbox'][vid][:])[:, :self.object_num, :]  # frame * 36 * 4
        if config.DATASET.NORMALIZE:
            features = F.normalize(features, dim=2)
        return features, confidences, bboxs

    def get_collate_fn(self, training=False):
        return datasets.orcnn_collate_fn


class TACoS_RCNN_C3D(data.Dataset):
    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    def __init__(self, split, rcnn_threshold=0.0, training=False):
        super(TACoS_RCNN_C3D, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split
        self.object_num = config.DATASET.OBJECT_NUM
        self.rcnn_threshold = rcnn_threshold

        # val_1.json is renamed as val.json, val_2.json is renamed as test.json
        with open(os.path.join(self.data_dir, '{}.json'.format(split)), 'r') as f:
            annotations = json.load(f)
        anno_pairs = []
        for vid, video_anno in annotations.items():
            duration = video_anno['num_frames'] / video_anno['fps']
            for timestamp, sentence in zip(video_anno['timestamps'], video_anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    anno_pairs.append(
                        {
                            'video': vid,
                            'duration': duration,
                            'times': [max(timestamp[0] / video_anno['fps'], 0),
                                      min(timestamp[1] / video_anno['fps'], duration)],
                            'description': sentence,
                        }
                    )
        self.annotations = anno_pairs

    def __getitem__(self, index):
        video_id = self.annotations[index]['video']
        gt_s_time, gt_e_time = self.annotations[index]['times']
        sentence = self.annotations[index]['description']
        duration = self.annotations[index]['duration']

        word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in sentence.split()], dtype=torch.long)
        word_vectors = self.word_embedding(word_idxs)

        visual_input, visual_mask = self.get_video_features(video_id)
        # visual_input = sample_to_fixed_length(visual_input, random_sampling=config.DATASET.RANDOM_SAMPLING)
        visual_input = average_to_fixed_length(visual_input)

        rcnn_input, rcnn_conf, rcnn_bbox = self.get_rcnn_features(video_id)
        # TODO mask
        rcnn_mask = rcnn_conf > self.rcnn_threshold
        # rcnn_mask = torch.ones_like(rcnn_conf).bool()
        # rcnn_input, rcnn_mask, rcnn_bbox = sample_to_fixed_length(rcnn_input, rcnn_mask, rcnn_bbox)
        rcnn_input = rcnn_input * rcnn_mask[:, :, None]
        rcnn_bbox = rcnn_bbox * rcnn_mask[:, :, None]

        num_clips = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
        s_times = torch.arange(0, num_clips).float() * duration / num_clips
        e_times = torch.arange(1, num_clips + 1).float() * duration / num_clips
        overlaps = iou(torch.stack([s_times[:, None].expand(-1, num_clips),
                                    e_times[None, :].expand(num_clips, -1)], dim=2).view(-1, 2).tolist(),
                       torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips, num_clips)

        item = {
            'visual_input': visual_input,
            'vis_mask': visual_mask,
            'rcnn_input': rcnn_input,
            'rcnn_mask': rcnn_mask,
            'rcnn_bbox': rcnn_bbox,
            'anno_idx': index,
            'word_vectors': word_vectors,
            'duration': duration,
            'reg_gt': torch.tensor([gt_s_time, gt_e_time]),
            'txt_mask': torch.ones(word_vectors.shape[0], 1),
            'map_gt': torch.from_numpy(overlaps),
            'description': sentence
        }

        return item

    def __len__(self):
        return len(self.annotations)

    def get_video_features(self, vid):
        assert config.DATASET.VIS_INPUT_TYPE == 'c3d'
        with h5py.File(os.path.join(self.data_dir, 'tall_c3d_features.hdf5'), 'r') as video_feature_bank:
            features = torch.from_numpy(video_feature_bank[vid][:])
        if config.DATASET.NORMALIZE:
            features = F.normalize(features, dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask

    def get_rcnn_features(self, vid):
        with h5py.File(os.path.join(self.data_dir, 'tacos_rcnn_features_256.hdf5'), 'r') as rcnn_feature_bank:
            data = torch.from_numpy(rcnn_feature_bank[vid][:])[:, :self.object_num, :]
            features = data[:, :, :2048]  # frame * 36 * 2048
            confidences = data[:, :, 2048]  # frame * 36
            bboxs = data[:, :, 2049:]  # frame * 36 * 4
        if config.DATASET.NORMALIZE:
            features = F.normalize(features, dim=2)
        return features, confidences, bboxs

    def get_rcnn_features1(self, vid):
        with h5py.File(os.path.join(self.data_dir, 'tacos_rcnn_features_untc.hdf5'), 'r') as rcnn_feature_bank:
            features = torch.from_numpy(rcnn_feature_bank['feats'][vid][:])[:, :self.object_num, :]  # frame * 36 * 2048
            confidences = torch.from_numpy(rcnn_feature_bank['confidence'][vid][:])[:, :self.object_num]  # frame * 36
            bboxs = torch.from_numpy(rcnn_feature_bank['bbox'][vid][:])[:, :self.object_num, :]  # frame * 36 * 4
        if config.DATASET.NORMALIZE:
            features = F.normalize(features, dim=2)
        return features, confidences, bboxs

    def get_collate_fn(self):
        return datasets.frcnn_collate_fn