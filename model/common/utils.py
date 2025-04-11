import random

import torch
from torch import nn, Tensor
from torch.nn.utils import rnn

import io
import json
import logging
import os
import pickle
import re
import shutil
import urllib
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urlparse
import numpy as np

def select_random_anchors(object_ids, need_distractors, instance_labels, excluded_labels={'wall', 'floor', 'ceiling'}):
    """
    Randomly selects 1 to 2 anchor objects for each batch sample. 
    If need_distractors is True, only one anchor is selected.

    Args:
    object_ids (torch.Tensor): The input tensor of object IDs with shape (B, N).
    need_distractors (torch.Tensor): Boolean tensor of shape (B,) indicating if distractors are needed.
    instance_labels (List[List[str]]): 2D list of shape (B, N) containing instance labels for each object.
    excluded_labels (set): Set of labels to be excluded from anchor selection.

    Returns:
    List[List[int]]: A list of lists, each containing indices of selected anchor objects for each batch.
    """
    B, N = object_ids.shape
    anchor_indices = []
    anchor_ids = []
    for b in range(B):
        anchor_ids.append([])
        # Create a list of valid indices (non-padding and not excluded labels)
        valid_indices = [i for i in range(N) if object_ids[b, i] != -1 and instance_labels[b][i] not in excluded_labels]
        if not valid_indices:
            raise
        num_anchors = 1 if need_distractors[b] else random.choices([1, 2], weights=[0.9, 0.1], k=1)[0]
        # Randomly select the anchors
        selected_anchors = random.sample(valid_indices, min(num_anchors, len(valid_indices)))
        anchor_indices.append(selected_anchors)
    for b in range(B):
        for d in range(len(anchor_indices[b])):
            anchor_ids[b].append(object_ids[b][anchor_indices[b][d]])
    return anchor_ids

def gather_same_instance_indices(object_ids, target_index, need_distractor, instance_labels):
    """
    Gathers indices of objects with the same instance label as the selected object,
    conditional on the need_distractor flag for each batch, for 2D list instance labels.

    Args:
    tensor (torch.Tensor): The input tensor of shape (B, N).
    selected_indices (torch.Tensor): Tensor of shape (B,) containing selected indices for each batch.
    need_distractor (torch.Tensor): Boolean tensor of shape (B,) indicating if distractors are needed.
    instance_labels (List[List[str]]): 2D list of shape (B, N) containing instance labels for each object.

    Returns:
    List[List[int]]: A list where each element is a list containing indices of objects with 
                     the same instance label as the selected object for each batch.
    """
    B, N = object_ids.shape
    same_label_indices = []
    distractor_ids = []
    for b in range(B):
        distractor_ids.append([])
        if need_distractor[b]:
            # Get the instance label of the selected index
            selected_label = instance_labels[b][target_index[b]]
            # Find all indices with the same instance label
            indices = [i for i, label in enumerate(instance_labels[b]) if label == selected_label and i != target_index[b]]
            same_label_indices.append(indices)
        else:
            same_label_indices.append([])  # Empty list for batches where distractor is not needed
    for b in range(B):
        for d in range(len(same_label_indices[b])):
            distractor_ids[b].append(object_ids[b][same_label_indices[b][d]])
            
    return distractor_ids

def select_random_target_with_label_constraint(object_ids, instance_labels, excluded_labels = {'wall', 'floor', 'ceiling'}):
    """
    Selects a random index for each batch, ensuring the selected object has at least one other object with the same instance label.

    Args:
    object_ids (torch.Tensor): The input tensor of object IDs with shape (B, N).
    instance_labels (List[List[str]]): 2D list of shape (B, N) containing instance labels for each object.

    Returns:
    torch.Tensor: A tensor of shape (B,) containing the selected indices.
    """
    B, N = object_ids.shape
    # Create a mask for valid (non-padding) object IDs
    valid_mask = object_ids != -1
    selected_indices = torch.empty(B, dtype=torch.long)
    for b in range(B):
        # Count the occurrences of each label in the batch
        label_counts = {}
        for i in range(N):
            if valid_mask[b, i]:
                label = instance_labels[b][i]
                if label not in excluded_labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
        # Identify valid indices (labels occurring more than once)
        valid_indices = [i for i in range(N) if valid_mask[b, i] and 
                         instance_labels[b][i] not in excluded_labels and 
                         label_counts.get(instance_labels[b][i], 0) > 1 and
                         label_counts.get(instance_labels[b][i], 0) < 6]
        if valid_indices:
            # Randomly select from the valid indices
            selected_indices[b] = valid_indices[torch.randint(0, len(valid_indices), (1,))]
        else:
            selected_indices[b] = 0
            print(instance_labels[b])
            #raise
    target_id = torch.empty(B, dtype=torch.long)
    for b in range(B):
        target_id[b] = object_ids[b][selected_indices[b]]
    return target_id, selected_indices

def rotation_aggregate(output):
    B, R, N, _ = output.shape
    """scaling_factors = torch.rand((B, R, 1, 1), device=self.device) * (1 - 0.33) + 0.33
    scaled_output = output * scaling_factors"""
    scaled_output = output
    return (scaled_output / R).sum(dim=1)

def break_up_pc(pc: Tensor) -> [Tensor, Tensor]:
    """
    Split the pointcloud into xyz positions and features tensors.
    This method is taken from VoteNet codebase (https://github.com/facebookresearch/votenet)
    @param pc: pointcloud [N, 3 + C]
    :return: the xyz tensor and the feature tensor
    """
    xyz = pc[..., 0:3].contiguous()
    features = (
        pc[..., 3:].transpose(1, 2).contiguous()
        if pc.size(-1) > 3 else None
    )
    return xyz, features

def create_mapping(input_dim, output_dim, dropout_rate):
    return nn.Sequential(nn.Linear(input_dim, output_dim),
                        nn.Dropout(dropout_rate),
                        nn.LayerNorm(output_dim))

def get_random_rotation_matrix(rotate_number, device):
    rotate_theta_arr = torch.Tensor([i*2.0*torch.pi/rotate_number for i in range(rotate_number)]).to(device)
    theta = rotate_theta_arr[torch.randint(0, rotate_number, (1,))]
    return get_rotation_matrix(theta, device)

def get_rotation_matrix(theta, device):
    rotate_matrix = torch.Tensor([[torch.cos(theta), -torch.sin(theta), 0.0],
                                [torch.sin(theta), torch.cos(theta),  0.0],
                                [0.0,           0.0,            1.0]]).to(device)
    return rotate_matrix

def get_augmented_color(rgb, contrast_range=(0.5, 1.5), noise_std_dev=0.02, device='cuda'):
    # RGB Augmentation
    contrast_factor = torch.empty(1).uniform_(contrast_range[0], contrast_range[1]).to(device)
    rgb = rgb * contrast_factor
    noise = torch.normal(mean=0., std=noise_std_dev, size=rgb.shape, device=device)
    rgb = rgb + noise
    rgb = torch.clamp(rgb, -1.0, 1.0)
    return rgb

def scale_to_unit_range(x):
    max_x = torch.max(x, dim=-1, keepdim=True).values
    min_x = torch.min(x, dim=-1, keepdim=True).values
    return x / (max_x - min_x + 1e-9)

def get_siamese_features(net, in_features, aggregator=None):
    """ Applies a network in a siamese way, to 'each' in_feature independently
    :param net: nn.Module, Feat-Dim to new-Feat-Dim
    :param in_features: B x  N-objects x Feat-Dim
    :param aggregator, (opt, None, torch.stack, or torch.cat)
    :return: B x N-objects x new-Feat-Dim
    """
    independent_dim = 1
    n_items = in_features.size(independent_dim)
    out_features = []
    for i in range(n_items):
        out_features.append(net(in_features[:, i]))
    if aggregator is not None:
        out_features = aggregator(out_features, dim=independent_dim)
    return out_features

def truncate_caption(caption: str) -> str:
    """Truncate captions at periods and newlines."""
    caption = caption.strip('\n')
    trunc_index = caption.find('\n') + 1
    if trunc_index <= 0:
        trunc_index = caption.find('.') + 1
    if trunc_index > 0:
        caption = caption[:trunc_index]
    return caption

def mask_token(inputs, tokenizer, mlm_probability, vocab_size=None, special_tokens_mask=None):
    """
    randomly mask some input tokens
    """
    indices_replaced = torch.bernoulli(torch.full(inputs.shape, mlm_probability)).bool()
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs

def build_one_instance_stage_2(tokenizer, captions, prompt='', padding=True):
    """
    </PC>: 32001
    \n: 29871, 13
    ###: 835
    Assistant: 4007, 22137
    : : 584
    tokenizer.bos_token_id
    """
    input_ids, target_ids = [], []
    texts = ''
    #text = '</PC> ' + prompt + '\n'
    text = prompt + '\n'
    texts += text
    one_input_id = tokenizer(text, add_special_tokens=False).input_ids
    input_ids += one_input_id
    target_ids += [-100] * len(one_input_id)
    
    text = 'Assistant: '
    one_input_id = [835] + tokenizer(text, add_special_tokens=False).input_ids
    input_ids += one_input_id  
    target_ids += [-100] * len(one_input_id)  # do not perform loss regression on human prompt
    
    text = captions + '\n'
    texts += text
    one_input_id = tokenizer(text, add_special_tokens=False).input_ids + [835]
    target_ids += one_input_id
    if padding:
        input_ids += one_input_id
    return input_ids, target_ids

def process_batch_stage_2(tokenizer, batch_of_captions, max_tgt_len, prompt='', padding=True):
    batch_input_ids, batch_target_ids = [], []
    for caption in batch_of_captions:
        one_input_ids, one_target_ids = build_one_instance_stage_2(tokenizer, caption, prompt, padding)
        batch_input_ids.append(torch.tensor(one_input_ids))
        batch_target_ids.append(torch.tensor(one_target_ids))
    # pads sequences to the length of the longest sequence in the batch.
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id) # ensures that padding tokens are used for inputs.
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100) # ensures that the padded areas do not contribute to loss calculation during training.
    if padding:
        assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def l2_loss(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """
  Args:
    u: (N, T_I_V_A.txt, D) tensor.
    v: (N, T_I_V_A.txt, D) tensor.
  Returns:
    l1_loss: (N,) tensor of summed L1 loss.
  """
  assert u.shape == v.shape, (u.shape, v.shape)
  return ((u - v) ** 2).sum(dim=-1) ** 0.5


def get_modality(path_list):
    _postfix = os.path.splitext(path_list[0])[-1]
    if _postfix == '.jpg':
        return 'image'
    else:
        raise NotImplementedError
