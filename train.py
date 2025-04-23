import torch
import numpy as np
from agents.policy import VisionProprioPolicy
from trainers.unsupervised_rl import EntropyTrainer
from data.replay_buffer import ReplayBuffer
from utils.vision import vision_tfm
from utils.maniskill_env import make_maniskill_env

def main():
    # === CONFIGS ===
    env_id = "PickCube-v1"
    obs_mode = "state_dict+rgb"
    control_mode = "pd_joint_delta_pos"
    render_mode = "rgb_array"
    # === HYPERPARAMS ===
    num_envs = 1
    image_size = (224, 224)
    proprio_dim = 9
    action_dim = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buffer_size = int(1e5)
    num_episodes = 5000

    # === ENV ===
    env = make_maniskill_env(env_id, obs_mode, control_mode, num_envs)
    print(f"env: {env_id}, obs_mode: {obs_mode}, control_mode: {control_mode}")
    print(f"action space: {env.action_space}")
    print(f"observation space: {env.observation_space}")
    obs = env.reset(seed=0)

    # === AGENT ===
    agent = VisionProprioPolicy(
        image_shape=(3, *image_size),
        proprio_dim=proprio_dim,
        action_dim=action_dim
    ).to(device)

    # === BUFFER ===
    buffer = ReplayBuffer(size=buffer_size)

    # === TRAINER ===
    trainer = EntropyTrainer(
        agent=agent,
        env=env,
        buffer=buffer,
        image_preprocess=vision_tfm,
        device=device,
    )

    # === TRAIN LOOP ===
    for ep in range(num_episodes):
        trainer.run_episode()
        if ep % 10 == 0:
            print(f"[ep {ep}] buffer size: {len(buffer.storage)}")
        if ep % 100 == 0:
            torch.save(agent.state_dict(), f"checkpoints/agent_ep{ep}.pth")

if __name__ == "__main__":
    main()
