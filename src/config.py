from dataclasses import dataclass

@dataclass
class PPOConfig:
    # env
    task: str = "Isaac-Reach-Franka-v0"
    num_envs: int = 4096

    # ppo
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs_per_update: int = 4
    minibatch_size: int = 2048
    horizon: int = 32
    
    # misc
    max_iterations: int = 500
    device: str = "cuda"