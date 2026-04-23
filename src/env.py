import gymnasium as gym
from isaaclab_tasks.utils import load_cfg_from_registry


class IsaacLabEnv:
    def __init__(self, cfg):
        env_cfg = load_cfg_from_registry(cfg.task, "env_cfg_entry_point")
        env_cfg.scene.num_envs = cfg.num_envs

        self.env = gym.make(cfg.task, cfg=env_cfg)

        self.num_envs = cfg.num_envs
        self.device = cfg.device

        # get dims from actual reset output
        obs_dict, _ = self.env.reset()
        self._initial_obs = obs_dict["policy"]
        self.obs_dim = self._initial_obs.shape[-1]
        self.act_dim = self.env.action_space.shape[-1]

    def reset(self):
        if self._initial_obs is not None:
            obs = self._initial_obs
            self._initial_obs = None
            return obs
        obs_dict, _ = self.env.reset()
        return obs_dict["policy"]

    def step(self, actions):
        obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)
        dones = terminated | truncated
        return obs_dict["policy"], rewards, dones, infos

    def close(self):
        self.env.close()