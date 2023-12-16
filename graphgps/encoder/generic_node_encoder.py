import math
import torch
import torch.nn.functional as F

from graphgps.encoder.kernel_pos_encoder import RWSENodeEncoder
from graphgps.encoder.laplace_pos_encoder import LapPENodeEncoder
from torch_geometric.graphgym.register import register_node_encoder

@register_node_encoder('GNE')
class GenericNodeEncoder(torch.nn.Module):
    def __init__(self, dim_embed, expand_x=True):
        super().__init__()
        self.pe_encoder = LapPENodeEncoder(dim_embed, expand_x)
        self.se_encoder = RWSENodeEncoder(dim_embed, expand_x)
        self.cum_loss_pe = 0.
        self.cum_loss_se = 0.
        self.last_action = 1 # 0 -> PE, 1 -> SE
        self.steps = 0
        self.num_restarts = 0

    def record_loss(self, loss):
        if self.last_action == 0:
            # LapPE
            self.cum_loss_pe += 0.1 * (loss - self.cum_loss_pe)
        else:
            # RWSE
            self.cum_loss_se += 0.1 * (loss - self.cum_loss_se)

    def forward(self, batch):
        # Restart when we've made enough steps
        # Exponential backoff before restart
        # Explore both LapPE and RWSE for the same
        # number of steps initially
        if self.steps >= 2 ** (self.num_restarts + 5):
            self.num_restarts += 1
            if torch.rand(1).item() < math.exp(-self.steps):
                # Exploration
                # Less chance to explore as we progress
                self.action = 1 - self.action
            else:
                # Decide next action based on current cumulative loss
                # Exploitation
                if self.cum_loss_pe < self.cum_loss_se:
                    self.last_action = 0
                else:
                    self.last_action = 1
            # print(f"{self.last_action=}")
            # print("LapPE", round(self.cum_loss_pe, 2))
            # print("RWSE", round(self.cum_loss_se, 2))

        if self.last_action == 0:
            # LapPE
            batch = self.pe_encoder(batch)
        else:
            # RWSE
            batch = self.se_encoder(batch)

        # Update number of steps
        self.steps += 1

        return batch
