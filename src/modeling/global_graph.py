""" Global interacting graph
Also served as the self-attention module for encoder and decoders.
"""
import math
import numpy as np
import torch
from torch import nn
from utils_files import utils

class GlobalGraph(nn.Module):
    r"""
    Global graph

    It's actually a self-attention.
    """

    def __init__(self, hidden_size, attention_head_size=None, num_attention_heads=1):
        super(GlobalGraph, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads if attention_head_size is None else attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.num_qkv = 1

        self.query = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        self.key = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        self.value = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)

    def get_extended_attention_mask(self, attention_mask):
        """
        1 in attention_mask stands for doing attention, 0 for not doing attention.

        After this function, 1 turns to 0, 0 turns to -10000.0

        Because the -10000.0 will be fed into softmax and -10000.0 can be thought as 0 in softmax.
        """
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def transpose_for_scores(self, x):
        sz = x.size()[:-1] + (self.num_attention_heads,
                              self.attention_head_size)
        # (batch, max_vector_num, head, head_size)
        x = x.view(*sz)
        # (batch, head, max_vector_num, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, mapping=None, return_scores=False):
        # Self-attention: transform the (same) input into Q,K,V
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = nn.functional.linear(hidden_states, self.key.weight)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        if attention_mask is not None:
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        if utils.args.visualize and mapping is not None:
            for i, each in enumerate(attention_probs.tolist()):
                mapping[i]['attention_scores'] = np.array(each[0])
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_scores:
            assert attention_probs.shape[1] == 1
            attention_probs = torch.squeeze(attention_probs, dim=1)
            assert len(attention_probs.shape) == 3
            return context_layer, attention_probs
        return context_layer


class CrossAttention(GlobalGraph):
    def __init__(self, hidden_size, attention_head_size=None, num_attention_heads=1, key_hidden_size=None,
                 query_hidden_size=None):
        super(CrossAttention, self).__init__(hidden_size, attention_head_size, num_attention_heads)
        if query_hidden_size is not None:
            self.query = nn.Linear(query_hidden_size, self.all_head_size * self.num_qkv)
        if key_hidden_size is not None:
            self.key = nn.Linear(key_hidden_size, self.all_head_size * self.num_qkv)
            self.value = nn.Linear(key_hidden_size, self.all_head_size * self.num_qkv)

    def forward(self, hidden_states_query, hidden_states_key=None, attention_mask=None, mapping=None,
                return_scores=False):
        mixed_query_layer = self.query(hidden_states_query)
        mixed_key_layer = self.key(hidden_states_key)
        mixed_value_layer = self.value(hidden_states_key)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        if attention_mask is not None:
            assert hidden_states_query.shape[1] == attention_mask.shape[1] \
                   and hidden_states_key.shape[1] == attention_mask.shape[2]
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_scores:
            return context_layer, torch.squeeze(attention_probs, dim=1)
        return context_layer


class GlobalGraphRes(nn.Module):
    def __init__(self, hidden_size):
        super(GlobalGraphRes, self).__init__()
        self.global_graph = GlobalGraph(hidden_size, hidden_size // 2)
        self.global_graph2 = GlobalGraph(hidden_size, hidden_size // 2)

    def forward(self, hidden_states, attention_mask=None, mapping=None):
        # hidden_states = self.global_graph(hidden_states, attention_mask, mapping) \
        #                 + self.global_graph2(hidden_states, attention_mask, mapping)
        hidden_states = torch.cat([self.global_graph(hidden_states, attention_mask, mapping),
                                   self.global_graph2(hidden_states, attention_mask, mapping)], dim=-1)
        return hidden_states

