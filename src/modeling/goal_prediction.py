from torch import nn
from typing import Dict, List, Tuple

class GoalPrediction(nn.Module):
    def __init__(self, hidden_size):
        super(GoalPrediction, self).__init__()

    def forward(self, mapping: List[Dict], lane_features, agents_lanes_embed, global_embed, device, loss):
        pass