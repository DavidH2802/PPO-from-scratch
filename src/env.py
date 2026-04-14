import isaacgym
import isaacgymenvs

class IsaacEnv:
    def __init__(self, cfg):
        self.env = isaacgymenvs.make(
            seed=cfg.seed,
            task=cfg.task,
            num_envs=cfg.num_envs,
            sim_device=cfg.device,
            rl_device=cfg.device,
            headless=True,
        )

        self.num_envs = cfg.num_envs
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.device = cfg.device

    def reset(self):
        obs_dict = self.env.reset()
        return obs_dict["obs"]

    def step(self, actions):
        obs_dict, rewards, dones, infos = self.env.step(actions)
        return obs_dict["obs"], rewards, dones, infos