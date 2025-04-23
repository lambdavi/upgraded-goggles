import timm
import torch
import torch.nn as nn

class VisionProprioPolicy(nn.Module):
    def __init__(self, image_shape, proprio_dim, action_dim, vit_type="vit_base_patch16_224"):
        super().__init__()
        
        self.vit = timm.create_model(vit_type, pretrained=True, num_classes=0)
        self.vit_proj = nn.Linear(self.vit.num_features, 256)

        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

        self.actor_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * 2),  # mean + log_std
        )

    def forward(self, image, proprio):
        # Encode
        vision_feat = self.vit(image)             # [B, D]
        vision_feat = self.vit_proj(vision_feat)  # [B, 256]
        proprio_feat = self.proprio_proj(proprio) # [B, 256]

        x = torch.cat([vision_feat, proprio_feat], dim=-1)  # [B, 512]
        x = self.actor_head(x)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        std = log_std.clamp(-5, 2).exp()
        return mean, std

    def sample_action(self, image, proprio):
        mean, std = self(image, proprio)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob
