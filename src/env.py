import gymnasium as gym

class IsaacLabEnv:
    def __init__(self, cfg):
        self.env = gym.make(
            cfg.task,
            num_envs=cfg.num_envs,
            device=cfg.device,
        )

        self.num_envs = cfg.num_envs
        self.obs_dim = self.env.observation_space["policy"].shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.device = cfg.device

    def reset(self):
        obs_dict, _ = self.env.reset()
        return obs_dict["policy"]

    def step(self, actions):
        obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)
        dones = terminated | truncated
        return obs_dict["policy"], rewards, dones, infos

    def close(self):
        self.env.close()