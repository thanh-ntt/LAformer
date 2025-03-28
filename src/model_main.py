# LAfromer: https://arxiv.org/pdf/2302.13933.pdf
# Written by Mengmeng Liu 
# All Rights Reserved
from typing import Dict, List
import torch
from torch import nn

from modeling.goal_prediction import GoalPrediction
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
        # TODO: question - why changing this global_graph variable name cause performance to reduce from 1.189 -> 1.6???
        self.global_graph = GlobalGraphRes(hidden_size)
        self.goal_prediction = GoalPrediction(args)
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

        # Encoder (section 3.2)
        # agents_lanes_embed: [Batch, SeqLen (agents+lanes), Dims]
        #       agents_lanes_embed: h_i = Concat[h_i, c_j]
        # lanes_embed: [Batch, SeqLen(lanes), Dims]
        agents_lanes_embed, lanes_embed = self.encoder.forward(mapping, vector_matrix, polyline_spans, device, batch_size)

        # Global Interaction Graph (after Agent2Lane & Lane2Agent in encoder)
        max_poly_num = agents_lanes_embed.shape[1]
        attention_mask = torch.zeros([batch_size, max_poly_num, max_poly_num], device=device)
        for i in range(batch_size):
            attention_mask[i][:max_poly_num][:max_poly_num].fill_(1)
        # SelfAtt{hi}
        global_embed = self.global_graph(agents_lanes_embed, attention_mask, mapping) # [Batch, SeqLen (agents+lanes), Dims]
        # print(f'[main] global_embed.shape: {global_embed.shape}')

        loss = torch.zeros(batch_size, device=device)

        if "step_lane_score" in args.other_params:
            dense_lane_topk = self.goal_prediction(mapping, lanes_embed, agents_lanes_embed, global_embed, device,
                                                    loss)  # [N, dense*mink, hidden_size + 1]
            print(f'[main] dense_lane_topk.shape: {dense_lane_topk.shape}')

        return self.decoder(mapping, batch_size, lanes_embed, agents_lanes_embed, global_embed, dense_lane_topk, device, loss)

    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict_rename_key = {}
        # to be compatible for the class structure of old models
        for key in state_dict.keys():
            if key.startswith('point_level_') or key.startswith('laneGCN_'):
                state_dict_rename_key['encoder.'+key] = state_dict[key]
            else:
                state_dict_rename_key[key] = state_dict[key]
        super(ModelMain, self).load_state_dict(state_dict_rename_key, strict)