"""The existing combination's single-cell path, without reference modules."""
from torch import nn


class CycleResidualSingleCell(nn.Module):
    def __init__(self, combination):
        super().__init__()
        # Reuse the modules themselves, including their initialization and names.
        self.cell_encoder = combination.cell_encoder
        self.ori_head = combination.ori_head

    def forward(self, feature):
        return self.ori_head(self.cell_encoder(feature)).view(-1)
