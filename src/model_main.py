# LAfromer: https://arxiv.org/pdf/2302.13933.pdf
# Written by Mengmeng Liu 
# All Rights Reserved
from typing import Dict, List
import torch
from torch import nn
from modeling.vectornet import VectorNet
from modeling.global_graph import CrossAttention, GlobalGraphRes
from modeling.laplace_decoder import  GRUDecoder
from utils_files import utils, config

class ModelMain(nn.Module):

    def __init__(self, args_: config.Args):
        super(ModelMain, self).__init__()
        global args
        args = args_
        hidden_size = args.hidden_size

        self.encoder = VectorNet(args)
        self.self_attention = GlobalGraphRes(hidden_size)
        self.decoder = GRUDecoder(args, self)

    def forward(self, mapping: List[Dict], device):
        """main forward call, will invoke encoder (vectornet), global interaction graph, & decoder

        Args:
            mapping: a dictionary of all pre-processed input
        """
        vector_matrix = utils.get_from_mapping(mapping, 'matrix')
        # vectors of i_th element is matrix[polyline_spans[i]]
        polyline_spans = utils.get_from_mapping(mapping, 'polyline_spans')
        batch_size = len(vector_matrix)
        utils.batch_origin_init(mapping)

        # Encoder
        # self.encoder.forward (section 3.2)
        #   ...
        #   agents_lanes_embed_list: h_i = Concat[h_i, c_j]
        agents_lanes_embed_list, lane_states_batch = self.encoder.forward(mapping, vector_matrix, polyline_spans, device, batch_size)
        # Global interacting graph
        inputs, inputs_lengths = utils.merge_tensors(agents_lanes_embed_list, device=device)
        # print(f'[main] inputs.shape: {inputs.shape}')
        # print(f'[main] len(lane_states_batch): {len(lane_states_batch)}')
        # print(f'[main] lane_states_batch[0].shape: {lane_states_batch[0].shape}')
        # print(f'[main] lane_states_batch[1].shape: {lane_states_batch[1].shape}')
        lane_states_batch, lane_states_length = utils.merge_tensors(lane_states_batch, device=device)
        # print(f'[main] lane_states_batch.shape: {lane_states_batch.shape}')
        max_poly_num = max(inputs_lengths)
        attention_mask = torch.zeros([batch_size, max_poly_num, max_poly_num], device=device)
        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1)
        global_hidden_states = self.self_attention(inputs, attention_mask, mapping)
        # print(f'[main] global_hidden_states.shape: {global_hidden_states.shape}')

        # Decoder
        return self.decoder(mapping, batch_size, lane_states_batch, lane_states_length, inputs, global_hidden_states, device)

    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict_rename_key = {}
        # to be compatible for the class structure of old models
        for key in state_dict.keys():
            if key.startswith('point_level_') or key.startswith('laneGCN_'):
                state_dict_rename_key['encoder.'+key] = state_dict[key]
            else:
                state_dict_rename_key[key] = state_dict[key]
        super(ModelMain, self).load_state_dict(state_dict_rename_key, strict)