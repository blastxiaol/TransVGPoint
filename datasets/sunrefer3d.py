# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import re
# import cv2
import sys
import json
import torch
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
sys.path.append('.')

from PIL import Image
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils.word_utils import Corpus


def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []

    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

class Parameters(object):
    def __init__(self, idx):
        self.idx = idx
        self.target = None
        self.sentence = None
        self.vmax = None
        self.vmin = None
        self.vsize = None
        self.point_cloud_path = None
        self.image_path = None
        self.calib_path = None

class SUNREFER3D(data.Dataset):
    def __init__(self, split='train', max_query_len=50, bert_model='bert-base-uncased', vrange=[], use_rgb=False):
        self.query_len = max_query_len
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.vmax = np.array(vrange[:3])
        self.vmin = np.array(vrange[3:6])
        self.vsize = self.vmax - self.vmin
        self.use_rgb = use_rgb
        if split == 'train':
            self.data_path = os.path.join('data/sunrefer/train_sunrefer.json')
        else:
            self.data_path = os.path.join('data/sunrefer/val_sunrefer.json')
        self.sunrefer = json.load(open(self.data_path, 'r'))

    def pull_item(self, idx):
        bbox = np.array(self.sunrefer[idx]['object_box'], dtype=np.float32)
        bbox[3:6] = bbox[3:6] * 2
        bbox[6] = bbox[6] * -1
        ## bbox normalize
        bbox[:3] = norm(bbox[:3], self.vmin, self.vmax)
        bbox[3:6] = bbox[3:6] / self.vsize
        bbox[6] = norm(bbox[6], -np.pi, np.pi)

        phrase = self.sunrefer[idx]['sentence']
        points = np.load(self.sunrefer[idx]['point_cloud_path'])['pc']
        points[:, :3] = norm(points[:, :3], self.vmin, self.vmax)
        if self.use_rgb:
            points[:, 3:6] = points[:, 3:6] / 255.
        else:
            points = points[:, :3]

        points = torch.tensor(points).float()
        bbox = torch.tensor(bbox).float()
        return points, phrase, bbox

    def __len__(self):
        return len(self.sunrefer)

    def __getitem__(self, idx):
        para = Parameters(idx)
        para.target = np.array(self.sunrefer[idx]['object_box'], dtype=np.float32)
        para.target[3:6] = para.target[3:6] * 2
        para.target[6] = para.target[6] * -1
        para.target = para.target.tolist()
        para.sentence = self.sunrefer[idx]['sentence']
        para.point_cloud_path = self.sunrefer[idx]['point_cloud_path']
        para.image_path = self.sunrefer[idx]['image_path']
        para.calib_path = self.sunrefer[idx]['calib_path']
        para.vmax = self.vmax
        para.vmin = self.vmin
        para.vsize = self.vsize
        points, phrase, bbox = self.pull_item(idx)
        phrase = phrase.lower()

        ## encode phrase to bert input
        examples = read_examples(phrase, idx)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask
 
        return points, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32), para

def norm(x, vmin, vmax):
    x = (x-vmin) / (vmax - vmin)
    return x