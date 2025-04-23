from torch import nn

class SlotEncoder(nn.Module):
    def __init__(self, backbone='resnet18', num_slots=6, slot_dim=64):
        pass

    def forward(self, image, proprio):
        return None  # [num_slots, slot_dim]
        