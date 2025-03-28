import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from utils_files import utils, config
from modeling.vectornet import MLP

class DecoderResCat(nn.Module):
    """
    DecoderResCat is a class name that likely stands for "Decoder with Residual Concatenation".
    This class implements a neural network module that uses a Multi-Layer Perceptron (MLP)
    to process input features and then concatenates the original input features with the output
    of the MLP before passing them through a fully connected layer.

    This concatenation can be seen as a form of residual connection,
    which helps in preserving the original input information while adding the learned transformations from the MLP.
    """
    def __init__(self, hidden_size, in_features, out_features=60):
        super(DecoderResCat, self).__init__()
        self.mlp = MLP(in_features, hidden_size)
        self.fc = nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states):
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.fc(hidden_states)
        return hidden_states

class GoalPrediction(nn.Module):
    def __init__(self, args: config.Args):
        super(GoalPrediction, self).__init__()
        decoder_layer_dense_label = nn.TransformerDecoderLayer(d_model=args.hidden_size, nhead=32,
                                                               dim_feedforward=args.hidden_size)
        self.dense_label_cross_attention = nn.TransformerDecoder(decoder_layer_dense_label, num_layers=1)
        self.dense_lane_decoder = DecoderResCat(args.hidden_size, args.hidden_size * 3,
                                                out_features=args.future_frame_num)
        self.args = args

        self.lane_segment_in_topk_num = 0
        self.invalid_lane_segment_in_topk_num = 0
        self.prediction_num = 0
        self.gt_invalid_num = 0

    def compute_dense_lane_scores(self, lane_features, agents_lanes_embed, global_embed, device):
        """predict score of the j-th lane segment at future time step t

            Question: Why does compute_dense_lane_scores() return Tensor shape [seq_len, N, H]?
                This fn returns a scalar (predicted score) of j-th lane segment at time step t
                    N * seq_len (j-th lane segment)
                    H * (time step t)

        Returns:
            tensor: [seq_len, batch, future_steps]
                seq_len varies between iterations
                future_steps: t_f (in section 3.3)
        """
        batch_size, lane_seq_length = lane_features.shape[0], lane_features.shape[1]
        print(f'[goal_prediction.compute_dense_lane_scores] lane_features.shape: {lane_features.shape}, agents_lanes_embed.shape: {agents_lanes_embed.shape}, global_embed.shape: {global_embed.shape}')

        src_attention_mask_lane = torch.zeros([batch_size, lane_features.shape[1]], device=device) # [N, max_len]
        for i in range(batch_size):
            src_attention_mask_lane[i, :lane_seq_length] = 1
        src_attention_mask_lane = src_attention_mask_lane == 0
        lane_features = lane_features.permute(1, 0, 2) # [max_len, N, feature]

        # Scaled dot product attention block
        # h_{i,att}: global_embed_att = cross_attention(Q: h_i, K,V: C)
        # Q: lanes_embed
        # K, V: element_hidden_states
        # A_{i,j} = softmax(...) = Scaled Dot Product Attention (K,V,Q above)
        # Question: where are the linear projections (of both K,V and Q)?
        #   => inside nn.TransformerDecoderLayer (it's already in the original "Attention is all you need!" paper)
        # lane_encoding_batch.shape = [max_len, N, feature]
        # lane_states_batch_attention.shape = [max_num_lanes, batch_size, hidden_size]
        # TODO: why the element-wise addiction '+'?
        lane_states_batch_attention = lane_features + self.dense_label_cross_attention(
            lane_features, agents_lanes_embed.unsqueeze(0), tgt_key_padding_mask=src_attention_mask_lane)
        # print(f'lane_states_batch_attention.shape: {lane_states_batch_attention.shape}')

        # \theta = 2-layer MLP to process:
        #   1. h_i: agent motion encoding
        #       Why need `expand(lane_states_batch.shape), lane_states_batch, lane_states_batch_attention], dim=-1)`?
        #       => To re-shape global_embed into lane_states_batch.shape
        #           ^ for concatenation operation `torch.cat`
        #       TODO: what if we use full h_i instead of just the h_i of last sequence?
        #   2. c_j: lane_states_batch: hidden states of lanes (lane encoding)
        #   3. A_{i,j}: lane_states_batch_attention: the predicted score of the j-th lane segment at t
        #
        # dense_lane_scores.shape = [max_num_lanes, batch_size, t_f]
        #   t_f: future_steps / future_frame_num (default value = 12)
        print(
            f'[goal_prediction] global_embed.shape: {global_embed.shape}\n\t lane_features.shape: {lane_features.shape}\n\t lane_states_batch_attention.shape: {lane_states_batch_attention.shape}')
        tmp_tensor = torch.cat([global_embed.unsqueeze(0).expand(
            lane_features.shape), lane_features, lane_states_batch_attention], dim=-1)
        print(f'[goal_prediction] tmp_tensor.shape: {tmp_tensor.shape}')
        dense_lane_scores = self.dense_lane_decoder(torch.cat([global_embed.unsqueeze(0).expand(
            lane_features.shape), lane_features, lane_states_batch_attention], dim=-1))  # [max_len, N, H]
        print(f'[goal_prediction] dense_lane_scores.shape: {dense_lane_scores.shape}')

        # Lane-scoring head
        # s^_{j,t} = softmax(\theta{ h_i, c_j, A_{i,j} }) <= this does not change shape of the tensor
        dense_lane_scores = F.log_softmax(dense_lane_scores, dim=0)
        # print(f'(2) dense_lane_scores.shape: {dense_lane_scores.shape}')
        return dense_lane_scores  # [seq_len, batch, future_steps] = [max_len, N, H]

    def forward(self, mapping: List[Dict], lanes_embed, agents_lanes_embed, global_embed, device, loss):
        batch_size = len(mapping)
        assert batch_size == lanes_embed.shape[0] == agents_lanes_embed.shape[0] == global_embed.shape[0], \
            "First dimension should be batch size"
        assert lanes_embed.shape[-1] == agents_lanes_embed.shape[-1] == global_embed.shape[-1], \
            "Last dimension should be model hidden size"

        # TODO: why the decoder only cares about the embedding of the 1st sequence in inputs & hidden_states?
        #   (?) seems like it discard all previous time steps?
        local_embed = agents_lanes_embed[:, 0, :]  # [batch_size, hidden_size]
        global_embed = global_embed[:, 0, :] # [batch_size, hidden_size] TODO: what if we use original global_embed?

        lane_seq_length = lanes_embed.shape[1]
        assert lane_seq_length > 0, "Empty lane sequence"
        future_frame_num = self.args.future_frame_num  # H

        # dense_lane_pred: prediction about the probability of each lane segment index that the agent will go to
        #       ^ that's why size = [N*H, max_len] <= max_len is # lane segments
        dense_lane_pred = self.compute_dense_lane_scores(lanes_embed, local_embed, global_embed, device) # [max_len, N, H]
        assert dense_lane_pred.shape == (lane_seq_length, batch_size, future_frame_num)

        dense_lane_pred = dense_lane_pred.permute(1, 0, 2) # [N, max_len, H]
        # lanes_embed = lanes_embed.permute(1, 0, 2) # [N, max_len, feature]
        print(f'[goal_prediction] (2) lanes_embed.shape: {lanes_embed.shape}')
        dense_lane_pred =  dense_lane_pred.permute(0, 2, 1) # [N, H, max_len]
        dense_lane_pred = dense_lane_pred.contiguous().view(-1, lane_seq_length)  # [N*H, max_len]
        print(f'[goal_prediction] dense_lane_pred.shape: {dense_lane_pred.shape}')

        # TODO: only use during training (now move out of `if self.args.do_train` for debugging)
        # dense_lane_targets: GT lane segment index
        #   ^ index of what? => self.subdivided_lane_traj_rel
        dense_lane_targets = torch.zeros([batch_size, future_frame_num], device=device, dtype=torch.long)
        for i in range(batch_size):
            dense_lane_targets[i, :] = torch.tensor(np.array(mapping[i]['dense_lane_labels']), dtype=torch.long,
                                                    device=device)
        dense_lane_targets = dense_lane_targets.view(-1)  # [N*H] - GT lane segment in each N*H frame

        if self.args.do_train:
            # compute L_lane loss (cross-entropy)
            lane_loss_weight = self.args.lane_loss_weight  # \lambda_1
            loss += lane_loss_weight*F.nll_loss(dense_lane_pred, dense_lane_targets, reduction='none').\
                    view(batch_size, future_frame_num).sum(dim=1) # cross-entropy loss L_lane

        mink = self.args.topk
        dense_lane_topk = torch.zeros((dense_lane_pred.shape[0], mink, self.args.hidden_size), device=device) # [N*H, mink, hidden_size]
        dense_lane_topk_scores = torch.zeros((dense_lane_pred.shape[0], mink), device=device)   # [N*H, mink]
        print(f'[goal_prediction] dense_lane_topk_scores.shape: {dense_lane_topk_scores.shape}')

        for i in range(dense_lane_topk_scores.shape[0]): # for each i in N*H (batch_size * future_frame_num)
            batch_idx = i // future_frame_num
            k = min(mink, lane_seq_length)
            _, topk_idxs = torch.topk(dense_lane_pred[i], k) # select top k=2 (or 4) lane segments to guide decoder
            # topk_idxs = top_k_indices()
            # print(f'[goal_prediction] topk_idxs.shape: {topk_idxs.shape}')
            # print(f'[goal_prediction] topk_idxs: {topk_idxs}')
            dense_lane_topk[i][:k] = lanes_embed[batch_idx, topk_idxs] # [N*H, mink, hidden_size]
            dense_lane_topk_scores[i][:k] = dense_lane_pred[i][topk_idxs] # [N*H, mink]

            self.lane_segment_in_topk_num += topk_idxs.size(0)

        # print(f'prediction_num: {self.prediction_num}, gt_invalid_num: {self.gt_invalid_num}, % = {self.gt_invalid_num / max(self.prediction_num, 1)}')
        # print(f'lane_segment_in_topk_num: {self.lane_segment_in_topk_num}, invalid_lane_segment_in_topk_num: {self.invalid_lane_segment_in_topk_num}, % = {self.invalid_lane_segment_in_topk_num / self.lane_segment_in_topk_num}')

        # obtain candidate lane encodings C = ConCat{c_{1:k}, s^_{1:k}}^{t_f}_{t=1}
        dense_lane_topk = torch.cat([dense_lane_topk, dense_lane_topk_scores.unsqueeze(-1)], dim=-1) # [N*H, mink, hidden_size + 1]
        dense_lane_topk = dense_lane_topk.view(batch_size, future_frame_num*mink, self.args.hidden_size + 1) # [N, sense*mink, hidden_size + 1]
        return dense_lane_topk # [N, H*mink, hidden_size + 1]
