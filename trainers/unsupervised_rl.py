import torch
class EntropyTrainer:
    def __init__(self, agent, env, buffer, image_preprocess, device):
        self.agent = agent
        self.env = env
        self.buffer = buffer
        self.image_preprocess = image_preprocess
        self.device = device

    def run_episode(self):
        obs = self.env.reset()
        image = obs[0]["sensor_data"]["base_camera"]["rgb"].to(torch.float).permute(0, 3, 1, 2)  # B, C, H, W
        image = self.image_preprocess(image).to(self.device) 
        proprio = obs[0]["agent"]["qpos"].to(self.device)  # B, 9

        done = False

        while not done:
            action, log_prob = self.agent.sample_action(image, proprio)
            next_obs, _, done, _, _ = self.env.step(action.detach().cpu())
            next_image = next_obs[0]["sensor_data"]["base_camera"]["rgb"].to(torch.float).permute(0, 3, 1, 2) # B, C, H, W
            next_image = self.image_preprocess(next_image).to(self.device)
            next_proprio = next_obs[0]["agent"]["qpos"].to(self.device)

            reward = log_prob.detach().cpu().item()

            self.buffer.store(image, proprio, action, reward, next_image, next_proprio)
            image, proprio = next_image, next_proprio

        self.update_policy()

    def update_policy(self):
        for _ in range(self.update_steps):
            batch = self.buffer.sample()
            self.agent.update(batch)
