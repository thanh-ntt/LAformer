# LAfromer: https://arxiv.org/pdf/2302.13933.pdf
# Written by Mengmeng Liu 
# All Rights Reserved
from typing import Dict, List, Tuple, NamedTuple, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from utils_files import utils, config
from modeling.global_graph import GlobalGraph, CrossAttention, GlobalGraphRes

class LayerNorm(nn.Module):
    r"""
    Layer normalization.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = torch.nn.functional.relu(hidden_states)
        return hidden_states

class PointLevelSubGraph(nn.Module):

    def __init__(self, hidden_size, depth=None):
        super(PointLevelSubGraph, self).__init__()
        if depth is None:
            depth = args.sub_graph_depth
        self.layers = nn.ModuleList(
            [MLP(hidden_size, hidden_size // 2) for _ in range(depth)])

        self.layer_0 = MLP(args.vector_size, hidden_size)
        if 'point_level-4-3' in args.other_params:
            self.layer_0_again = MLP(hidden_size)
        self.GRU = nn.GRU(hidden_size, hidden_size, num_layers= 2, batch_first=True)

    def forward(self, hidden_states, lengths):
        hidden_states = self.layer_0(hidden_states)
        if 'point_level-4-3' in args.other_params:
            hidden_states = hidden_states + self.layer_0_again(hidden_states)
        output, hn = self.GRU(hidden_states)
        return hn[-1], None  # torch.cat(utils.de_merge_tensors(hidden_states, lengths))

class PointLevelSubGraph_lane(nn.Module):

    def __init__(self, hidden_size, depth=None):
        super(PointLevelSubGraph_lane, self).__init__()
        if depth is None:
            depth = 3
        self.layers = nn.ModuleList(
            [MLP(hidden_size, hidden_size // 2) for _ in range(depth)])

        self.layer_0 = MLP(args.vector_size, hidden_size)
        if 'point_level-4-3' in args.other_params:
            self.layer_0_again = MLP(hidden_size)
        self.GRU = nn.GRU(hidden_size, hidden_size, num_layers= 2, batch_first=True)

    def forward(self, hidden_states, lengths):
        hidden_states = self.layer_0(hidden_states)

        if 'point_level-4-3' in args.other_params:
            hidden_states = self.layer_0_again(hidden_states)
            
        output, hn = self.GRU(hidden_states)
        return hn[-1], None  # torch.cat(utils.de_merge_tensors(hidden_states, lengths))

class VectorNet(nn.Module):
    r"""
    VectorNet

    It has two main components, sub graph and global graph.

    Sub graph encodes a polyline as a single vector.
    """

    def __init__(self, args_: config.Args):
        super(VectorNet, self).__init__()
        global args
        args = args_
        hidden_size = args.hidden_size

        self.point_level_sub_graph = PointLevelSubGraph(hidden_size)
        self.point_level_sub_graph_lane = PointLevelSubGraph_lane(hidden_size)
        self.point_level_cross_attention = CrossAttention(hidden_size)
        if "nuscenes" in args.other_params:
            num_layers = 1
        else:
            num_layers = 3

        decoder_layer_A2L = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=hidden_size)
        self.laneGCN_A2L = nn.TransformerDecoder(decoder_layer_A2L, num_layers=num_layers)
        decoder_layer_L2A = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=hidden_size)
        self.laneGCN_L2A = nn.TransformerDecoder(decoder_layer_L2A, num_layers=num_layers)

    def forward(self, mapping: List[Dict], matrix: List[np.ndarray], polyline_spans: List[List[slice]],
                                 device, batch_size) -> Tuple[List[Tensor], Tensor]:
        """
        :param matrix: each value in list is vectors of all element (shape [-1, 128])
        :param polyline_spans: vectors of i_th element is matrix[polyline_spans[i]]
        :return: hidden states of all elements and hidden states of lanes
        """
        # get agent and lane tensor list
        all_agent_lists, all_lane_lists, batch_split_agent, batch_split_lane = [], [], [], []
        start_lane, end_lane = 0, 0
        start_agent, end_agent = 0, 0
        for i in range(batch_size):
            map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
            for j, polyline_span in enumerate(polyline_spans[i]):
                tensor = torch.tensor(matrix[i][polyline_span],
                                        device=device)
                if j < map_start_polyline_idx:
                    all_agent_lists.append(tensor)
                else:
                    all_lane_lists.append(tensor)
                if j == map_start_polyline_idx or map_start_polyline_idx == len(polyline_spans[i]):
                    end_agent += map_start_polyline_idx
                    batch_split_agent.append([start_agent, end_agent])
                    start_agent = end_agent
            end_lane += len(polyline_spans[i]) - map_start_polyline_idx
            batch_split_lane.append([start_lane, end_lane])
            start_lane = end_lane
        # point level sub graph in batch
        device = all_agent_lists[0].device
        all_agent_lists, lengths = utils.merge_tensors(all_agent_lists, device, args.vector_size)
        all_lane_lists, lengths_lane = utils.merge_tensors(all_lane_lists, device, args.vector_size)
        all_agent_states_unspilited, _ = self.point_level_sub_graph(all_agent_lists, lengths)
        all_lane_states_unspilited, _ = self.point_level_sub_graph_lane(all_lane_lists, lengths_lane)
        # Run laneGCN, get element states and lane_states
        agent_states_batch, lane_states_batch = [], [] # len(agent_states_batch) = len(lane_states_batch) = batch
        for i in range(batch_size):
            map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
            agents = all_agent_states_unspilited[batch_split_agent[i][0]:batch_split_agent[i][1]]
            lanes = all_lane_states_unspilited[batch_split_lane[i][0]:batch_split_lane[i][1]]
            agent_states_batch.append(agents)
            lane_states_batch.append(lanes)
        # print(f'[encoder] len(agent_states_batch): {len(agent_states_batch)}')
        # print(f'[encoder] agent_states_batch[0].shape: {agent_states_batch[0].shape}') # [4, 64] <= different value in different iterations
        # print(f'[encoder] agent_states_batch[1].shape: {agent_states_batch[1].shape}') # [27, 64] <= different value in different iterations
        # print(f'[encoder] len(lane_states_batch): {len(lane_states_batch)}')
        # print(f'[encoder] lane_states_batch[0].shape: {lane_states_batch[0].shape}') # [62, 64] <= different value in different iterations
        # print(f'[encoder] lane_states_batch[1].shape: {lane_states_batch[1].shape}') # [88, 64] <= different value in different iterations
        # agent_states_batch.shape = [batch, max_agent_states_length, feature]
        #   max_agent_states_length (lengths) varies between iterations
        agent_states_batch, lengths = utils.merge_tensors(agent_states_batch, device, args.hidden_size)
        # print(f'[encoder] (1) agent_states_batch.shape: {agent_states_batch.shape}')
        # lane_states_batch.shape = [batch, max_lane_states_length, feature]
        #   max_lane_states_length (lengths_lane) varies between iterations
        # print(f'len(lane_states_batch): {len(lane_states_batch)}')
        # for i in range(10):
        #     print(f'lane_states_batch[{i}].shape: {lane_states_batch[i].shape}')
        lane_states_batch, lengths_lane = utils.merge_tensors(lane_states_batch, device, args.hidden_size)
        # print(f'[encoder] (1) lane_states_batch.shape: {lane_states_batch.shape}')
        # print(f'[encoder] lengths_lane: {lengths_lane}')
        src_attention_mask_lane = torch.zeros([batch_size, lane_states_batch.shape[1]], device=device)
        src_attention_mask_agent = torch.zeros([batch_size, agent_states_batch.shape[1]], device=device)
        for i in range(batch_size):
            assert lengths[i] > 0
            assert lengths_lane[i] > 0
            src_attention_mask_lane[i, :lengths_lane[i]] = 1
            src_attention_mask_agent[i, :lengths[i]] = 1
        src_attention_mask_lane = src_attention_mask_lane == 0
        src_attention_mask_agent = src_attention_mask_agent == 0
        lane_states_batch = lane_states_batch.permute(1, 0, 2)  # [seq_len, batch, feature]
        agent_states_batch = agent_states_batch.permute(1, 0, 2)  # [seq_len, batch, feature]

        lanes_embed = lane_states_batch + self.laneGCN_A2L(lane_states_batch, agent_states_batch, \
                                            memory_key_padding_mask=src_attention_mask_agent, tgt_key_padding_mask=src_attention_mask_lane)
        agent_states_batch = agent_states_batch+ self.laneGCN_L2A(agent_states_batch, lanes_embed, \
                                            memory_key_padding_mask=src_attention_mask_lane, tgt_key_padding_mask=src_attention_mask_agent)
        agent_states_batch = agent_states_batch.permute(1, 0, 2)  # [batch, seq_len, feature]
        # print(f'[encoder] (2) agent_states_batch.shape: {agent_states_batch.shape}')
        lanes_embed = lanes_embed.permute(1, 0, 2)  # [batch, seq_len, feature]
        # print(f'[encoder] (2) lanes_embed.shape: {lanes_embed.shape}')
        element_states_batch = []
        for i in range(batch_size):
            element_states_batch.append(torch.cat([agent_states_batch[i], lanes_embed[i]], dim=0))
        # print(f'[encoder] len(element_states_batch): {len(element_states_batch)}')
        # print(f'[encoder] element_states_batch[0].shape: {element_states_batch[0].shape}') # <= different value in different iterations
        # print(f'[encoder] element_states_batch[1].shape: {element_states_batch[1].shape}') # <= different value in different iterations
        # print(f'[encoder] lanes_embed.shape: {lanes_embed.shape}')
        # print(f'[encoder] lanes_embed.shape[1]: {lanes_embed.shape[1]}')
        # len(element_states_batch) = batch
        #   element_states_batch[i].shape = [max_agent_states_length + max_lane_states_length, feature]
        # lanes_embed.shape = [batch, max_lane_states_length, feature]
        #   max_lane_states_length varies between iterations
        #   max_agent_states_length varies between iterations
        return element_states_batch, lanes_embed  # h_i, c_j

