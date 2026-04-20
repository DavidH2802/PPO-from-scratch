import isaacgym
import torch
from config import PPOConfig
from env import IsaacEnv
from agent import PPOAgent


def train():
    cfg = PPOConfig()

    env = IsaacEnv(cfg)
    agent = PPOAgent(env.obs_dim, env.act_dim, cfg)

    obs = env.reset()

    for iteration in range(cfg.max_iterations):
        obs = agent.collect_rollout(env, obs)

        advantages, returns = agent.compute_gae(obs)

        losses = agent.update(advantages, returns)

        mean_reward = agent.rew_buf.sum(dim=0).mean().item()
        print(
            f"iteration {iteration:4d} | "
            f"reward {mean_reward:8.2f} | "
            f"policy_loss {losses['policy_loss']:7.4f} | "
            f"value_loss {losses['value_loss']:7.4f}"
        )

        if (iteration + 1) % 50 == 0:
            torch.save({
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "optimizer": agent.optimizer.state_dict(),
                "iteration": iteration,
            }, f"checkpoint_{iteration + 1}.pt")

    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
    }, "final_policy.pt")

    print("done")


if __name__ == "__main__":
    train()