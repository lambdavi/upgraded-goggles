import torch
import random
import numpy as np

from collections import deque


class ReplayBuffer:
    def __init__(self, size):
        self.storage = deque(maxlen=size)

    def store(self, img, prop, action, reward, next_img, next_prop):
        self.storage.append((img, prop, action, reward, next_img, next_prop))

    def sample(self, batch_size=64):
        batch = random.sample(self.storage, batch_size)
        imgs, props, actions, rewards, next_imgs, next_props = zip(*batch)
        return {
            "img": torch.stack(imgs),
            "prop": torch.stack(props),
            "action": torch.stack(actions),
            "reward": torch.tensor(rewards).float(),
            "next_img": torch.stack(next_imgs),
            "next_prop": torch.stack(next_props),
        }
