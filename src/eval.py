from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import isaaclab_tasks
import gymnasium as gym
import torch
from config import PPOConfig
from env import IsaacLabEnv
from model import Actor
from utils.normalization import RunningMeanStd
from isaaclab_tasks.utils import load_cfg_from_registry


def eval():
    cfg = PPOConfig()
    cfg.num_envs = 16

    # create env with render_mode for video
    env_cfg = load_cfg_from_registry(cfg.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.viewer.resolution = (640, 480)
    env_cfg.viewer.eye = (1.5, 1.5, 1.5)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)

    env = gym.make(cfg.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder="videos",
        episode_trigger=lambda ep: True,
        video_length=300,
    )

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    obs_dim = obs.shape[-1]
    act_dim = env.action_space.shape[-1]

    checkpoint = torch.load("final_policy.pt", map_location=cfg.device)

    actor = Actor(obs_dim, act_dim).to(cfg.device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    # restore obs normalization
    obs_rms = RunningMeanStd(shape=(obs_dim,), device=cfg.device)
    obs_rms.mean = checkpoint["obs_rms_mean"]
    obs_rms.var = checkpoint["obs_rms_var"]

    for step in range(500):
        obs_norm = obs_rms.normalize(obs)

        with torch.no_grad():
            mean = actor(obs_norm)

        obs_dict, reward, terminated, truncated, info = env.step(mean)
        obs = obs_dict["policy"]

    env.close()
    simulation_app.close()
    print("Video saved to videos/")


if __name__ == "__main__":
    eval()