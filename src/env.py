from isaaclab_tasks.direct.cartpole.cartpole_env import CartpoleEnv, CartpoleEnvCfg


class IsaacLabEnv:
    def __init__(self, cfg):
        env_cfg = CartpoleEnvCfg()
        env_cfg.scene.num_envs = cfg.num_envs

        self.env = CartpoleEnv(cfg=env_cfg)

        self.num_envs = cfg.num_envs
        self.obs_dim = env_cfg.observation_space
        self.act_dim = env_cfg.action_space
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