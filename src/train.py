from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab_tasks
import torch
from config import PPOConfig
from env import IsaacLabEnv
from agent import PPOAgent
from utils.logger import Logger
from utils.normalization import RunningMeanStd


def train():
    cfg = PPOConfig()

    env = IsaacLabEnv(cfg)
    agent = PPOAgent(env.obs_dim, env.act_dim, cfg)
    logger = Logger(log_dir=f"runs/{cfg.task}")
    obs_rms = RunningMeanStd(shape=(env.obs_dim,), device=cfg.device)
    rew_rms = RunningMeanStd(shape=(), device=cfg.device)

    obs = env.reset()

    for iteration in range(cfg.max_iterations):
        obs = agent.collect_rollout(env, obs)

        obs_rms.update(agent.obs_buf.reshape(-1, env.obs_dim))
        agent.obs_buf = obs_rms.normalize(agent.obs_buf)

        raw_mean_reward = agent.rew_buf.sum(dim=0).mean().item()

        rew_rms.update(agent.rew_buf.reshape(-1))
        agent.rew_buf = rew_rms.normalize(agent.rew_buf)

        obs = obs_rms.normalize(obs)

        advantages, returns = agent.compute_gae(obs)
        losses = agent.update(advantages, returns)

        logger.log({
            "reward/mean": raw_mean_reward,
            "loss/policy": losses["policy_loss"],
            "loss/value": losses["value_loss"],
        }, step=iteration)

        print(
            f"iteration {iteration:4d} | "
            f"reward {raw_mean_reward:8.2f} | "
            f"policy_loss {losses['policy_loss']:7.4f} | "
            f"value_loss {losses['value_loss']:7.4f}"
        )

        if (iteration + 1) % 50 == 0:
            torch.save({
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "optimizer": agent.optimizer.state_dict(),
                "obs_rms_mean": obs_rms.mean,
                "obs_rms_var": obs_rms.var,
                "iteration": iteration,
            }, f"checkpoint_{iteration + 1}.pt")

    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "obs_rms_mean": obs_rms.mean,
        "obs_rms_var": obs_rms.var,
    }, "final_policy.pt")

    logger.close()
    env.close()
    simulation_app.close()
    print("done")


if __name__ == "__main__":
    train()