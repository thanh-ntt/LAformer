# LAfromer: https://arxiv.org/pdf/2302.13933.pdf
# Written by Mengmeng Liu 
# All Rights Reserved
import math
from pprint import pprint
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict, List, Tuple, NamedTuple, Any
import numpy as np
from utils_files import utils, config
from utils_files.utils import init_weights, get_random_ints, compute_angle_diff
from modeling.vectornet import *
from modeling.motion_refinement import trajectory_refinement
from utils_files.loss import *

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

class GRUDecoder(nn.Module):
    def __init__(self, args: config.Args, vectornet) -> None:
        super(GRUDecoder, self).__init__()
        min_scale: float = 1e-3
        self.input_size = args.hidden_size
        self.hidden_size = args.hidden_size
        self.future_steps = args.future_frame_num
        self.num_modes = args.mode_num
        self.min_scale = min_scale
        self.args = args
        self.future_frame_num = args.future_frame_num
        self.z_size = args.z_size
        self.smothl1 = torch.nn.SmoothL1Loss(reduction='none')
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=False,
                          dropout=0,
                          bidirectional=False)
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        self.scale = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        self.pi = nn.Sequential(
                nn.Linear(self.hidden_size*2, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, 1))
        self.aggregate_global_z = nn.Sequential(
            nn.Linear(self.hidden_size + 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True))
        self.reg_loss = LaplaceNLLLoss(reduction='none')
        # self.reg_loss = GaussianNLLLoss(reduction='none')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='none')
        if "step_lane_score" in args.other_params:
            self.multihead_proj_global = nn.Sequential(
                                        nn.Linear(self.hidden_size*2, self.num_modes * self.hidden_size),
                                        nn.LayerNorm(self.num_modes * self.hidden_size),
                                        nn.ReLU(inplace=True))  
            decoder_layer_dense_label = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=32, dim_feedforward=self.hidden_size)
            self.dense_label_cross_attention = nn.TransformerDecoder(decoder_layer_dense_label, num_layers=1)
            self.dense_lane_decoder = DecoderResCat(self.hidden_size, self.hidden_size * 3, out_features=self.future_frame_num)
            self.proj_topk = MLP(self.hidden_size+1, self.hidden_size)
            decoder_layer_aggregation = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=32, dim_feedforward=self.hidden_size)
            self.aggregation_cross_att= nn.TransformerDecoder(decoder_layer_aggregation, num_layers=1)
        else:
            self.multihead_proj_global = nn.Sequential(
                                        nn.Linear(self.hidden_size, self.num_modes * self.hidden_size),
                                        nn.LayerNorm(self.num_modes * self.hidden_size),
                                        nn.ReLU(inplace=True))
        self.apply(init_weights)   
        if "stage_two" in args.other_params:
            if args.do_train:
                model_recover = torch.load(args.other_params['stage-two-train_recover'])
                vectornet.decoder = self
                utils.load_model(vectornet, model_recover)
                # # self must be vectornet
                # for p in vectornet.parameters():
                #     p.requires_grad = False
            self.trajectory_refinement = trajectory_refinement(args)

        self.lane_segment_in_topk_num = 0
        self.invalid_lane_segment_in_topk_num = 0
        self.prediction_num = 0
        self.gt_invalid_num = 0

    def dense_lane_aware(self, mapping: List[Dict], lane_features, element_hidden_states, global_embed, device, loss):
        """future_frame_num lane aware
        Args:
            mapping (list): data mapping
            lane_features (tensor): [batch, lane_seq_len, feature]
                hidden states of lanes (lane encoding c_j)
            element_hidden_states (tensor): [batch, feature]
            global_embed (tensor): [batch, feature]
                global_embed is the output of the Global Interaction Graph
                global_embed contains both agent & lane states
                ^ check model_main.py's forward fn
                    GlobalGraph has nn.Linear for all K,Q,V => linear projection

        Tensor shape explanation:
            feature: hidden_size (size of latent vector)
            N: batch size
            H: t_f * 2 = number of future time steps (default = 6s * 2Hz)
            seq_len: max_len / max_num_lanes / maximum number of lane states from encoder
                Each lane has different number (e.g. 62, 88, ...); Each iteration has different
                 maximum number of lane states from encoder (e.g. 290, 289, 292, ...)

        Returns:
            tensor: candidate lane encodings C = ConCat{c_{1:k}, s^_{1:k}}^{t_f}_{t=1}
        """
        def compute_dense_lane_scores():
            """predict score of the j-th lane segment at future time step t
            TODO: loss re-weighting

                Question: Why does compute_dense_lane_scores() return Tensor shape [seq_len, N, H]?
                    This fn returns a scalar (predicted score) of j-th lane segment at time step t
                        N * seq_len (j-th lane segment)
                        H * (time step t)

            Returns:
                tensor: [seq_len, batch, future_steps]
                    seq_len varies between iterations
                    future_steps: t_f (in section 3.3)
            """
            # Scaled dot product attention block
            # h_{i,att}: global_embed_att = cross_attention(Q: h_i, K,V: C)
            # Q: lane_features = lane encoding c_j
            # K, V: element_hidden_states = agent motion encoding h_i
            # A_{i,j} = softmax(...) = Scaled Dot Product Attention (K,V,Q above)
            # Question: where are the linear projections (of both K,V and Q)?
            #   => inside nn.TransformerDecoderLayer (it's already in the original "Attention is all you need!" paper)
            # lane_encoding_batch.shape = [max_len, N, feature]
            # lane_states_batch_attention.shape = [max_num_lanes, batch_size, hidden_size]
            # TODO: why the element-wise addiction '+'?
            lane_states_batch_attention = lane_features + self.dense_label_cross_attention(
                lane_features, element_hidden_states.unsqueeze(0), tgt_key_padding_mask=src_attention_mask_lane)
            # print(f'lane_states_batch_attention.shape: {lane_states_batch_attention.shape}')

            # \theta = 2-layer MLP to process:
            #   1. h_i: agent motion encoding
            #       Why need `expand(lane_states_batch.shape), lane_states_batch, lane_states_batch_attention], dim=-1)`?
            #       => To re-shape global_embed into lane_states_batch.shape
            #           ^ for concatenation operation `torch.cat`
            #   2. c_j: lane_states_batch: hidden states of lanes (lane encoding)
            #   3. A_{i,j}: lane_states_batch_attention: the predicted score of the j-th lane segment at t
            #
            # dense_lane_scores.shape = [max_num_lanes, batch_size, t_f]
            #   t_f: future_steps / future_frame_num (default value = 12)
            print(f'[decoder] global_embed.shape: {global_embed.shape}\n\t lane_features.shape: {lane_features.shape}\n\t lane_states_batch_attention.shape: {lane_states_batch_attention.shape}')
            tmp_tensor = torch.cat([global_embed.unsqueeze(0).expand(
                lane_features.shape), lane_features, lane_states_batch_attention], dim=-1)
            print(f'[decoder] tmp_tensor.shape: {tmp_tensor.shape}')
            dense_lane_scores = self.dense_lane_decoder(torch.cat([global_embed.unsqueeze(0).expand(
                lane_features.shape), lane_features, lane_states_batch_attention], dim=-1)) # [max_len, N, H]
            print(f'[decoder] dense_lane_scores.shape: {dense_lane_scores.shape}')

            # Lane-scoring head
            # s^_{j,t} = softmax(\theta{ h_i, c_j, A_{i,j} }) <= this does not change shape of the tensor
            dense_lane_scores = F.log_softmax(dense_lane_scores, dim=0)
            # print(f'(2) dense_lane_scores.shape: {dense_lane_scores.shape}')
            return dense_lane_scores # [seq_len, batch, future_steps] = [max_len, N, H]

        def top_k_indices() -> Tensor:
            """
            Get the top k indices in pred_score, (optional) apply penalty
            """
            # i: index of dense_lane_pred and dense_lane_targets along 1st dimension - shape [N*H, max_len]
            #     i in range (0, N*H)
            # batch_idx = i // future_frame_num (H)

            # why can lane_meta be shorter than pred_score? => pred_score is computed from lane_states_batch (merged tensor)
            subdivided_lane_to_lane_meta = utils.get_from_mapping(mapping, 'subdivided_lane_to_lane_meta')
            assert dense_lane_pred[i].shape[0] >= len(subdivided_lane_to_lane_meta[batch_idx])

            pred_score_processed = torch.tensor(dense_lane_pred[i], device=device)
            ego_angle_abs = - (utils.get_from_mapping(mapping, 'angle')[batch_idx] - math.pi / 2)

            self.prediction_num += 1
            for lane_idx in range(0, len(subdivided_lane_to_lane_meta[batch_idx])):
                lane_angle, _, layer = subdivided_lane_to_lane_meta[batch_idx][lane_idx]
                # if layer == 'lane' and compute_angle_diff(lane_angle, ego_angle_abs) > (math.pi * 4 / 5):
                # velocity = utils.get_from_mapping(mapping, 'past_traj')[batch_idx][-1, 2]
                if compute_angle_diff(lane_angle, ego_angle_abs) > (math.pi * 3 / 4):
                # if compute_angle_diff(lane_angle, ego_angle_abs) > (math.pi * 3 / 4) and i % future_frame_num < 4:
                # if (compute_angle_diff(lane_angle, ego_angle_abs) > (math.pi * 3 / 4)
                #         and i % future_frame_num < 4 and velocity > 4.167):
                    pred_score_processed[lane_idx] = - math.inf

                    if lane_idx == dense_lane_targets[i]:
                        self.gt_invalid_num += 1

            _, topk_indices = torch.topk(dense_lane_pred[i], k)
            _, valid_topk_indices = torch.topk(pred_score_processed, k)
            if not torch.equal(topk_indices, valid_topk_indices):
                for lane_idx in topk_indices:
                    if lane_idx not in valid_topk_indices:
                        self.invalid_lane_segment_in_topk_num += 1

            return valid_topk_indices

        lane_seq_length = lane_features.shape[1]
        assert lane_seq_length > 0
        batch_size = len(mapping)
        src_attention_mask_lane = torch.zeros([batch_size, lane_features.shape[1]], device=device) # [N, max_len]
        for i in range(batch_size):
            src_attention_mask_lane[i, :lane_seq_length] = 1
        src_attention_mask_lane = src_attention_mask_lane == 0
        lane_features = lane_features.permute(1, 0, 2) # [max_len, N, feature]

        # dense_lane_pred: prediction about the probability of each lane segment index that the agent will go to
        #       ^ that's why size = [N*H, max_len] <= max_len is # lane segments
        dense_lane_pred = compute_dense_lane_scores() # [max_len, N, H] - log(true_pred_score)
        dense_lane_pred = dense_lane_pred.permute(1, 0, 2) # [N, max_len, H]
        lane_features = lane_features.permute(1, 0, 2) # [N, max_len, feature]
        dense_lane_pred =  dense_lane_pred.permute(0, 2, 1) # [N, H, max_len]
        dense_lane_pred = dense_lane_pred.contiguous().view(-1, lane_seq_length)  # [N*H, max_len]

        future_frame_num = self.future_frame_num  # H

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
        dense_lane_topk = torch.zeros((dense_lane_pred.shape[0], mink, self.hidden_size), device=device) # [N*H, mink, hidden_size]
        dense_lane_topk_scores = torch.zeros((dense_lane_pred.shape[0], mink), device=device)   # [N*H, mink]

        for i in range(dense_lane_topk_scores.shape[0]): # for each i in N*H (batch_size * future_frame_num)
            batch_idx = i // future_frame_num
            k = min(mink, lane_seq_length)
            _, topk_idxs = torch.topk(dense_lane_pred[i], k) # select top k=2 (or 4) lane segments to guide decoder
            # topk_idxs = top_k_indices()
            dense_lane_topk[i][:k] = lane_features[batch_idx, topk_idxs] # [N*H, mink, hidden_size]
            dense_lane_topk_scores[i][:k] = dense_lane_pred[i][topk_idxs] # [N*H, mink]

            self.lane_segment_in_topk_num += topk_idxs.size(0)

        # print(f'prediction_num: {self.prediction_num}, gt_invalid_num: {self.gt_invalid_num}, % = {self.gt_invalid_num / max(self.prediction_num, 1)}')
        # print(f'lane_segment_in_topk_num: {self.lane_segment_in_topk_num}, invalid_lane_segment_in_topk_num: {self.invalid_lane_segment_in_topk_num}, % = {self.invalid_lane_segment_in_topk_num / self.lane_segment_in_topk_num}')

        # obtain candidate lane encodings C = ConCat{c_{1:k}, s^_{1:k}}^{t_f}_{t=1}
        dense_lane_topk = torch.cat([dense_lane_topk, dense_lane_topk_scores.unsqueeze(-1)], dim=-1) # [N*H, mink, hidden_size + 1]
        dense_lane_topk = dense_lane_topk.view(batch_size, future_frame_num*mink, self.hidden_size + 1) # [N, sense*mink, hidden_size + 1]
        return dense_lane_topk # [N, H*mink, hidden_size + 1]

    def forward(self, mapping: List[Dict], batch_size, lanes_embed, agents_lanes_embed: Tensor, global_embed: Tensor,
                device) -> Tuple[torch.Tensor, torch.Tensor]:
        """lane-aware estimation + multimodal conditional decoder
        Args:
            lanes_embed: hidden states of lanes
                [batch, lane_seq_len, feature]
                    batch: train_batch_size
                    lane_seq_len: max_vector_num
                    feature: hidden_size
                ^ c_j
                This is not encoded by global graph

            agents_lanes_embed: hidden states of agents before encoding by global graph
                [batch, seq_len, feature]
                    batch: train_batch_size
                    seq_len: max_agent_states_length + max_vector_num
                    feature: hidden_size
                ^ h_i (BEFORE global graph)
                This actually also contains the hidden states of lanes - encoder code:
                    `element_states_batch.append(torch.cat([agent_states_batch[i], lane_states_batch[i]], dim=0))`
                    ^ Explained in section 3.2:
                        Afterward, the GIG further explores self-attention and skip-connection to learn the interactions among agents.
                        Namely, h_i = ConCat[h_i, c_j] for j ∈ {1, ..., N_lane}

            global_embed: hidden states (embedding) of agent & lane after Global Interaction Graph (GIG)
                [batch, seq_len, feature]
                    batch: train_batch_size
                    seq_len: max_agent_states_length + max_vector_num
                    feature: hidden_size
                ^ h_i (AFTER global graph)

        Other variables:
            local_embed: hidden states of agents before encoding by global graph
                [batch_size, hidden_size]
                code: `local_embed = inputs[:, 0, :]`
                This slicing operation extracts the first element (along the sequence length dimension)
                from the inputs tensor for each batch, resulting in a tensor local_embed with the shape [batch_size, hidden_size].
                Essentially, it selects the first hidden state from the inputs tensor for each batch.
            global_embed: hidden states of agents after encoding by global graph
                [batch_size, hidden_size]
        """
        labels = utils.get_from_mapping(mapping, 'labels')
        labels_is_valid = utils.get_from_mapping(mapping, 'labels_is_valid')
        loss = torch.zeros(batch_size, device=device)
        DE = np.zeros([batch_size, self.future_steps])
        # TODO: why the decoder only cares about the embedding of the 1st sequence in inputs & hidden_states?
        #   (?) seems like it discard all previous time steps?
        local_embed = agents_lanes_embed[:, 0, :]  # [batch_size, hidden_size]
        global_embed = global_embed[:, 0, :] # [batch_size, hidden_size]
        if "step_lane_score" in self.args.other_params:
            dense_lane_topk = self.dense_lane_aware(mapping, lanes_embed, local_embed, global_embed, device,
                                                    loss)  # [N, dense*mink, hidden_size + 1]
            dense_lane_topk = dense_lane_topk.permute(1, 0, 2)  # [dense*mink, N, hidden_size + 1]
            # TODO: (paper) "and the candidate lane encodings C as the key and value vectors" => Why need projection proj_topk?
            dense_lane_topk = self.proj_topk(dense_lane_topk) # [dense*mink, N, hidden_size]
            # h_{i,att}: global_embed_att = cross_attention(Q: h_i, K,V: C)
            # Q: global_embed = the target agent’s past trajectory encoding h_i
            # K, V: dense_lane_topk = candidate lane encodings = C
            #   ^ (C = ConCat{c_{1:k}, s^_{1:k}}^{t_f}_{t=1}) obtained in self.dense_lane_aware
            global_embed_att = global_embed + self.aggregation_cross_att(global_embed.unsqueeze(0), dense_lane_topk).squeeze(0) # [N, D]
            global_embed = torch.cat([global_embed, global_embed_att], dim=-1) # [N, 2*D]
        local_embed = local_embed.repeat(self.num_modes, 1, 1)  # [F, N, D]
        global_embed = self.multihead_proj_global(global_embed).view(-1, self.num_modes, self.hidden_size)  # [N, F, D]
        batch_size = global_embed.shape[0]
        global_embed = global_embed.transpose(0, 1)  # [F, N, D] 
        # if "stage_two" in self.args.other_params:
        pi = self.pi(torch.cat((local_embed, global_embed), dim=-1)).squeeze(-1).t()  # [N, F]
        global_embed = global_embed.reshape(-1, self.input_size)  # [F x N, D]

        z_size = self.z_size
        # the latent vector z sampled from a multivariate normal distribution, served as an additional inputs
        #   => why need this z? => to model the multi-modal distribution of the output trajectories
        z = torch.randn(self.num_modes*batch_size,  z_size, device=device) # [F*N, 5]
        global_embed = torch.cat([global_embed, z], dim=-1)  # [F x N, D+z_size]
        global_embed = self.aggregate_global_z(global_embed)  # [F x N, D]
        
        global_embed = global_embed.expand(self.future_steps, *global_embed.shape)  # [H, F x N, D]
        local_embed = local_embed.reshape(-1, self.input_size).unsqueeze(0)  # [1, F x N, D]
        out, _ = self.gru(global_embed, local_embed)
        out = out.transpose(0, 1)  # [F x N, H, D]
        loc = self.loc(out).view(self.num_modes, -1, self.future_steps, 2)  # [F, N, H, 2]
        scale = F.elu_(self.scale(out), alpha=1.0)+ 1.0 + self.min_scale  # [F x N, H, 2]
        scale = scale.view(self.num_modes, -1, self.future_steps, 2)  # [F, N, H, 2]
        if "stage_two" in self.args.other_params:
            past_traj = utils.get_from_mapping(mapping, 'past_traj') #[N, T, 2]]
            past_traj = torch.tensor(np.array(past_traj), dtype=torch.float32, device=device)
            past_traj = past_traj[:,:,:2]
            past_traj = past_traj.expand(self.num_modes, *past_traj.shape)  # [F, N, T, 2]
            full_traj = torch.cat((past_traj, loc), dim=2) # [F, N, H+T, 2]
            loc_delta, _ = self.trajectory_refinement(out, full_traj, global_embed, local_embed) #  [N, F, H], [F, N, H, 2]
        if "stage_two" in self.args.other_params:
            return self.laplace_decoder_loss((loc, loc_delta, past_traj), scale, pi, labels_is_valid, loss, DE, device, labels, mapping)
        else:
            return self.laplace_decoder_loss(loc, scale, pi, labels_is_valid, loss, DE, device, labels, mapping)


    def laplace_decoder_loss(self, loc, scale, pi, labels_is_valid, loss, DE, device, labels, mapping=None):
        if "stage_two" in self.args.other_params:
            original_loc, loc_delta, past_traj = loc
            loc = original_loc + loc_delta
        y_hat = torch.cat((loc, scale), dim=-1)
        batch_size = y_hat.shape[1]
        labels = torch.tensor(np.array(labels), device = device)
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - labels, p=2, dim=-1) ).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(y_hat.shape[1])]
        if "stage_two" in self.args.other_params and self.args.do_train:
            loc_delta_best = loc_delta[best_mode, torch.arange(y_hat.shape[1])]
            delta_label = labels-original_loc[best_mode, torch.arange(y_hat.shape[1])]
            reg_delta_loss = torch.norm(loc_delta_best-delta_label, p=2, dim=-1) # [N, H]
            reg_loss = self.reg_loss(y_hat_best, labels).sum(dim=-1) + 5*reg_delta_loss  # [N, H]
            loss += get_angle_diff(labels, y_hat_best[:, :, :2], past_traj)*2
            soft_target = F.softmax(-l2_norm/ self.future_steps, dim=0).t().detach() #[N, F]
            cls_loss = self.cls_loss(pi, soft_target)
        else:
            reg_loss = self.reg_loss(y_hat_best, labels).sum(dim=-1)
            soft_target = F.softmax(-l2_norm/ self.future_steps, dim=0).t().detach() #[N, F]
            cls_loss = self.cls_loss(pi, soft_target)
        if self.args.do_train:
            for i in range(batch_size):
                if self.args.do_train:
                    assert labels_is_valid[i][-1]
                loss_ = reg_loss[i]
                loss_ = loss_ * torch.tensor(labels_is_valid[i], device=device, dtype=torch.float).view(self.future_steps, 1)
                if labels_is_valid[i].sum() > utils.eps:
                    loss[i] += loss_.sum() / labels_is_valid[i].sum()
                loss[i] += cls_loss[i]
        if self.args.do_eval:
            outputs = loc.permute(1, 0, 2, 3).detach()
            pred_probs = F.softmax(pi, dim=-1).cpu().detach().numpy()
            for i in range(batch_size):
                if self.args.visualize:
                    labels = utils.get_from_mapping(mapping, 'labels')
                    labels = np.array(labels)
                    utils.visualize_gifs(
                        mapping[i], self.args.future_frame_num,
                        labels[i], outputs[i].cpu().numpy())
                outputs[i] = utils.to_origin_coordinate(outputs[i], i)
                if "vis_nuscenes" in self.args.other_params:
                    from utils_files import vis_nuscenes
                    vis_nuscenes.generate_nuscenes_gif(mapping[i], self.args.future_frame_num, outputs[i].cpu().numpy())
            outputs = outputs.cpu().numpy()
            return outputs, pred_probs, None

        return loss.mean(), DE, None






